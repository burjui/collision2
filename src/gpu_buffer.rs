use std::{marker::PhantomData, mem::size_of, ops::RangeBounds};

use bytemuck::NoUninit;
use wgpu::{Buffer, BufferSlice, COPY_BUFFER_ALIGNMENT, Device, Queue};

#[derive(Clone)]
pub struct GpuBuffer<T> {
    buffer: Buffer,
    _marker: PhantomData<T>,
}

impl<T> GpuBuffer<T> {
    pub fn new(length: usize, label: &str, usage: wgpu::BufferUsages, device: &Device) -> Self {
        let required_size = u64::try_from(length * size_of::<T>()).unwrap();
        let padded_size = required_size.div_ceil(COPY_BUFFER_ALIGNMENT) * COPY_BUFFER_ALIGNMENT;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: padded_size,
            usage,
            mapped_at_creation: false,
        });
        Self {
            buffer,
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

    pub fn slice(&self, bounds: impl RangeBounds<usize>) -> BufferSlice<'_> {
        let start = match bounds.start_bound() {
            std::ops::Bound::Included(&start) => start,
            std::ops::Bound::Excluded(&start) => start + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            std::ops::Bound::Included(&end) => end,
            std::ops::Bound::Excluded(&end) => end - 1,
            std::ops::Bound::Unbounded => self.len(),
        };
        let slice_start = u64::try_from(start * size_of::<T>()).unwrap();
        let slice_end = u64::try_from(end * size_of::<T>()).unwrap();
        assert!(
            slice_start <= slice_end && slice_end <= self.buffer.size(),
            "Bounds {slice_start}..{slice_end} out of range {}..{}",
            0,
            self.buffer.size()
        );
        self.buffer.slice(slice_start..slice_end)
    }

    pub fn write(&self, queue: &Queue, data: &[T])
    where
        T: NoUninit,
    {
        let data_size = u64::try_from(size_of_val(data)).unwrap();
        assert!(data_size <= self.buffer.size());
        let mut view = queue.write_buffer_with(&self.buffer, 0, data_size.try_into().unwrap()).unwrap();
        view.as_mut().copy_from_slice(bytemuck::cast_slice(data));
    }

    fn size(&self) -> usize {
        self.buffer.size().try_into().unwrap()
    }
}
