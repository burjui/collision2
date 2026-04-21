use std::{
    marker::PhantomData,
    mem::size_of,
    ops::{Bound, RangeBounds},
};

use bytemuck::{AnyBitPattern, NoUninit, Pod};
use wgpu::{
    Buffer, BufferAsyncError, BufferSize, BufferSlice, COPY_BUFFER_ALIGNMENT, Device, MapMode, Queue, WasmNotSend,
    util::DeviceExt,
};

/// Typesafe handle to a wgpu buffer
#[derive(Clone)]
pub struct TypedBuffer<T> {
    buffer: Buffer,
    _marker: PhantomData<T>,
}

impl<T> TypedBuffer<T> {
    pub fn new(device: &Device, length: usize, label: &str, usage: wgpu::BufferUsages) -> Self {
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

    pub fn write(&self, queue: &Queue, data: &[T])
    where
        T: NoUninit,
    {
        let data_size: BufferSize = u64::try_from(size_of_val(data)).unwrap().try_into().unwrap();
        let mut view = queue.write_buffer_with(&self.buffer, 0, data_size).unwrap();
        view.as_mut().copy_from_slice(bytemuck::cast_slice(data));
    }

    pub fn write_iter(&self, queue: &Queue, data: impl Iterator<Item = T>)
    where
        T: NoUninit + AnyBitPattern,
    {
        let buffer_size: BufferSize = self.buffer.size().try_into().unwrap();
        let mut view = queue.write_buffer_with(&self.buffer, 0, buffer_size).unwrap();
        data.zip(bytemuck::cast_slice_mut(view.as_mut()).iter_mut()).for_each(|(src, dst)| *dst = src);
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
}
