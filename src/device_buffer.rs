use std::{
    marker::PhantomData,
    mem::size_of,
    ops::{Bound, Range, RangeBounds},
};

use bytemuck::{AnyBitPattern, NoUninit, Pod};
use itertools::Itertools;
use wgpu::{
    Buffer, BufferAsyncError, BufferBinding, BufferSize, COPY_BUFFER_ALIGNMENT, CommandEncoder, Device, MapMode, Queue,
    WasmNotSend, util::DeviceExt,
};

/// Typesafe handle to a wgpu buffer
#[derive(Clone)]
pub struct DeviceBuffer<T> {
    buffer: Buffer,
    _marker: PhantomData<T>,
}

impl<T> DeviceBuffer<T> {
    pub fn new(device: &Device, length: u32, label: &str, usage: wgpu::BufferUsages) -> Self {
        let required_size = u64::from(length) * u64::try_from(size_of::<T>()).unwrap();
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

    pub fn from_data(device: &Device, data: &[T], label: &str, usage: wgpu::BufferUsages) -> Self
    where
        T: NoUninit,
    {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage,
        });
        Self {
            buffer,
            _marker: PhantomData,
        }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn as_entire_buffer_binding(&self) -> BufferBinding<'_> {
        self.buffer.as_entire_buffer_binding()
    }

    pub fn len(&self) -> usize {
        self.size() / size_of::<T>()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn write(&self, queue: &Queue, data: &[T])
    where
        T: NoUninit,
    {
        let data_size: BufferSize = u64::try_from(size_of_val(data)).unwrap().try_into().unwrap();
        let mut view = queue.write_buffer_with(&self.buffer, 0, data_size).unwrap();
        view.copy_from_slice(bytemuck::cast_slice(data));
    }

    pub fn write_iter(&self, queue: &Queue, iter: impl Iterator<Item = T>)
    where
        T: NoUninit + AnyBitPattern,
    {
        let buffer_size: BufferSize = self.buffer.size().try_into().unwrap();
        let data = iter.collect_vec();
        let mut view = queue.write_buffer_with(&self.buffer, 0, buffer_size).unwrap();
        let mut slice = view.slice(0..data.len() * size_of::<T>());
        slice.copy_from_slice(bytemuck::cast_slice(&data));
    }

    fn size(&self) -> usize {
        self.buffer.size().try_into().unwrap()
    }

    pub fn read(&self, size: usize, callback: impl FnOnce(Result<Vec<T>, BufferAsyncError>) + WasmNotSend + 'static)
    where
        T: Pod,
    {
        let dst_size = u64::try_from(size * size_of::<T>()).unwrap();
        let copy_size = dst_size.min(self.buffer.size());
        let buffer = self.buffer.clone();
        self.buffer.map_async(MapMode::Read, .., move |result| {
            callback(result.map(|_| {
                let view = buffer.get_mapped_range(0..copy_size);
                let data = bytemuck::cast_slice(&view).to_vec();
                drop(view);
                buffer.unmap();
                data
            }));
        });
    }

    pub fn copy(
        &self,
        dst_bounds: impl RangeBounds<usize>,
        src: &DeviceBuffer<T>,
        src_bounds: impl RangeBounds<usize>,
        encoder: &mut CommandEncoder,
    ) {
        let dst_range = self.range(dst_bounds);
        let src_range = src.range(src_bounds);
        encoder.copy_buffer_to_buffer(
            &src.buffer,
            src_range.start,
            &self.buffer,
            dst_range.start,
            src_range.end - src_range.start,
        );
    }

    fn range(&self, bounds: impl RangeBounds<usize>) -> Range<u64> {
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
        let start = u64::try_from(start * size_of::<T>()).unwrap();
        let end = u64::try_from(end * size_of::<T>()).unwrap();
        assert!(
            start <= end && end <= self.buffer.size(),
            "Bounds {start}..{end} out of range {}..{}",
            0,
            self.buffer.size()
        );
        start..end
    }
}
