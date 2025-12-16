use std::{marker::PhantomData, mem::size_of};

use bytemuck::NoUninit;
use wgpu::{Buffer, COPY_BUFFER_ALIGNMENT, Device, Queue};

pub struct WgpuBuffer<T> {
    buffer: Buffer,
    length: usize,
    _marker: PhantomData<T>,
}

impl<T> WgpuBuffer<T> {
    pub fn new(device: &Device, label: &str, length: usize, usage: wgpu::BufferUsages) -> Self {
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
            length,
            _marker: PhantomData,
        }
    }

    pub fn inner(&self) -> &Buffer {
        &self.buffer
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn enque_write(&self, queue: &Queue, data: &[T])
    where
        T: NoUninit,
    {
        let data_size = u64::try_from(size_of_val(data)).unwrap();
        let mut view = queue.write_buffer_with(&self.buffer, 0, data_size.try_into().unwrap()).unwrap();
        view.as_mut().copy_from_slice(bytemuck::cast_slice(data));
    }
}
