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
        let start: u64 = self.start.try_into().unwrap();
        let length: u64 = (length * size_of::<T>()).try_into().unwrap();
        let end = start + length;
        assert!(end <= self.buffer.size(), "Arena is out of space");
        self.start = end;
        GpuSlice::new(self.buffer.clone(), start..end)
    }
}

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

    pub fn len(&self) -> usize {
        (self.range.end - self.range.start).try_into().unwrap()
    }

    pub fn slice(&self, range: Range<usize>) -> BufferSlice<'_> {
        let start: u64 = range.start.try_into().unwrap();
        let end: u64 = range.end.try_into().unwrap();
        let slice_length = end - start;
        let self_length = self.range.end - self.range.start;
        assert!(
            slice_length <= self_length,
            "Range {start}..{end} is out of bounds ({}..{})",
            self.range.start,
            self.range.end
        );
        self.buffer.slice(self.range.start + start..self.range.start + end)
    }

    pub fn enque_write(&self, queue: &Queue, data: &[T])
    where
        T: NoUninit,
    {
        let size = self.range.end - self.range.start;
        let data_size = u64::try_from(size_of_val(data)).unwrap();
        assert!(data_size <= size);
        let mut view = queue.write_buffer_with(&self.buffer, self.range.start, data_size.try_into().unwrap()).unwrap();
        view.as_mut().copy_from_slice(bytemuck::cast_slice(data));
    }
}
