use std::time::Duration;

use wgpu::{BufferUsages, CommandEncoder, Device, QuerySet, QueryType};

use crate::gpu_buffer::GpuBuffer;

pub struct PassDurationMeasurer {
    device: Device,
    query_set: QuerySet,
    query_buffer: GpuBuffer<u64>,
    query_readback_buffer: GpuBuffer<u64>,
}

impl PassDurationMeasurer {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("integrator duration query set"),
                ty: QueryType::Timestamp,
                count: 2,
            }),
            query_buffer: GpuBuffer::new(
                2,
                "integrator duration query buffer",
                BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                device,
            ),
            query_readback_buffer: GpuBuffer::new(
                2,
                "integrator duration query readback buffer",
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                device,
            ),
        }
    }

    pub fn start(&self, encoder: &mut CommandEncoder) -> PassDurationMeasurementStart {
        encoder.write_timestamp(&self.query_set, 0);

        PassDurationMeasurementStart {
            device: self.device.clone(),
            query_set: self.query_set.clone(),
            query_buffer: self.query_buffer.clone(),
            query_readback_buffer: self.query_readback_buffer.clone(),
        }
    }
}

pub struct PassDurationMeasurementStart {
    device: Device,
    query_set: QuerySet,
    query_buffer: GpuBuffer<u64>,
    query_readback_buffer: GpuBuffer<u64>,
}

impl PassDurationMeasurementStart {
    pub fn finish(self, encoder: &mut CommandEncoder) -> PassDurationMeasurementResult {
        encoder.write_timestamp(&self.query_set, 1);
        encoder.resolve_query_set(&self.query_set, 0..2, self.query_buffer.buffer(), 0);
        encoder.copy_buffer_to_buffer(self.query_buffer.buffer(), 0, self.query_readback_buffer.buffer(), 0, None);
        PassDurationMeasurementResult {
            device: self.device,
            query_readback_buffer: self.query_readback_buffer,
        }
    }
}

pub struct PassDurationMeasurementResult {
    device: Device,
    query_readback_buffer: GpuBuffer<u64>,
}

impl PassDurationMeasurementResult {
    pub fn duration(self) -> Duration {
        let mut timestamps = [0u64; 2];
        self.query_readback_buffer.read(&self.device, &mut timestamps);
        Duration::from_nanos(timestamps[1] - timestamps[0])
    }
}
