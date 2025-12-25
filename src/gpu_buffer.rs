use std::{
    marker::PhantomData,
    mem::size_of,
    ops::{Bound, RangeBounds},
};

use bytemuck::{NoUninit, Pod};
use crossbeam::sync::WaitGroup;
use wgpu::{Buffer, BufferSlice, COPY_BUFFER_ALIGNMENT, Device, MapMode, PollType, Queue};

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
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start + 1,
            Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            Bound::Included(&end) => end + 1,
            Bound::Excluded(&end) => end,
            Bound::Unbounded => self.len(),
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

    pub fn write(&self, queue: &Queue, src: &[T])
    where
        T: NoUninit,
    {
        let data_size = u64::try_from(size_of_val(src)).unwrap();
        assert!(data_size <= self.buffer.size());
        let mut view = queue.write_buffer_with(&self.buffer, 0, data_size.try_into().unwrap()).unwrap();
        view.as_mut().copy_from_slice(bytemuck::cast_slice(src));
    }

    fn size(&self) -> usize {
        self.buffer.size().try_into().unwrap()
    }

    pub fn read(&self, device: &Device, dst: &mut [T])
    where
        T: Pod,
    {
        let dst_size = u64::try_from(size_of_val(dst)).unwrap();
        assert!(dst_size >= self.buffer.size());

        let wait_group = WaitGroup::new();
        let wg = wait_group.clone();
        self.buffer.map_async(MapMode::Read, .., |result| {
            result.unwrap();
            drop(wg);
        });
        device.poll(PollType::Poll).unwrap();
        wait_group.wait();

        let view = self.buffer.get_mapped_range(0..dst_size);
        dst.copy_from_slice(bytemuck::cast_slice(&view));
        drop(view);
        self.buffer.unmap();
    }
}
