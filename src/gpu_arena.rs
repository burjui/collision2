use std::{marker::PhantomData, mem::size_of, ops::Range};

use bytemuck::NoUninit;
use wgpu::{Buffer, BufferSlice, COPY_BUFFER_ALIGNMENT, Device, Queue};

pub struct GpuArena {
    buffer: Buffer,
    start: u64,
}

impl GpuArena {
    pub fn new(size: usize, label: &str, usage: wgpu::BufferUsages, device: &Device) -> Self {
        assert!(size > 0);
        let required_size = u64::try_from(size).unwrap();
        let padded_size = required_size.div_ceil(COPY_BUFFER_ALIGNMENT) * COPY_BUFFER_ALIGNMENT;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: padded_size,
            usage,
            mapped_at_creation: false,
        });
        Self { buffer, start: 0 }
    }

    pub fn new_slice<T>(
        length: usize,
        label: &str,
        usage: wgpu::BufferUsages,
        device: &Device,
    ) -> (GpuArena, GpuSlice<T>) {
        let mut arena = GpuArena::new(length * size_of::<T>(), label, usage, device);
        let slice = arena.allocate(length);
        (arena, slice)
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn allocate<T>(&mut self, length: usize) -> GpuSlice<T> {
        let length: u64 = (length * size_of::<T>()).try_into().unwrap();
        let range = self.start..self.start + length;
        assert!(range.end <= self.buffer.size(), "Arena is out of space");
        self.start = range.end;
        GpuSlice::new(self.buffer.clone(), range)
    }
}

#[derive(Clone)]
pub struct GpuSlice<T> {
    buffer: Buffer,
    range: Range<u64>,
    _marker: PhantomData<T>,
}

impl<T> GpuSlice<T> {
    fn new(buffer: Buffer, range: Range<u64>) -> Self {
        Self {
            buffer,
            range,
            _marker: PhantomData,
        }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn len(&self) -> usize {
        self.size() / size_of::<T>()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn size(&self) -> usize {
        (self.range.end - self.range.start).try_into().unwrap()
    }

    pub fn slice_all(&self) -> BufferSlice<'_> {
        self.buffer.slice(self.range.clone())
    }

    pub fn slice(&self, range: Range<usize>) -> BufferSlice<'_> {
        let slice_size = range.len() * size_of::<T>();
        let start = self.range.start + u64::try_from(range.start * size_of::<T>()).unwrap();
        let end = start + u64::try_from(slice_size).unwrap();
        assert!(
            start >= self.range.start && end <= self.range.end,
            "Range {start}..{end} is out of bounds ({}..{})",
            self.range.start,
            self.range.end
        );
        self.buffer.slice(start..end)
    }

    pub fn write(&self, queue: &Queue, data: &[T])
    where
        T: NoUninit,
    {
        let size = self.range.end - self.range.start;
        let data_size = u64::try_from(size_of_val(data)).unwrap();
        assert!(data_size <= size);
        let mut view = queue.write_buffer_with(&self.buffer, self.range.start, data_size.try_into().unwrap()).unwrap();
        view.as_mut().copy_from_slice(bytemuck::cast_slice(data));
    }

    pub fn cast<U>(&self) -> GpuSlice<U> {
        assert!(size_of::<U>() == size_of::<T>());
        GpuSlice::new(self.buffer.clone(), self.range.clone())
    }
}
