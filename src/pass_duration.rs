use std::time::Duration;

use wgpu::{
    BufferUsages, CommandEncoder, ComputePassTimestampWrites, Device, QuerySet, QuerySetDescriptor, QueryType,
    RenderPassTimestampWrites,
};

use crate::gpu_buffer::GpuBuffer;

#[must_use]
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
            query_set: device.create_query_set(&QuerySetDescriptor {
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

    pub fn compute_pass_timestamp_writes(&self) -> ComputePassTimestampWrites<'_> {
        ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }
    }

    pub fn render_pass_timestamp_writes(&self) -> RenderPassTimestampWrites<'_> {
        RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }
    }

    pub fn measure(&self, encoder: &mut CommandEncoder, mut f: impl FnMut(&mut CommandEncoder)) {
        encoder.write_timestamp(&self.query_set, 0);
        f(encoder);
        encoder.write_timestamp(&self.query_set, 1);
    }

    pub fn update(&self, encoder: &mut CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..2, self.query_buffer.buffer(), 0);
        encoder.copy_buffer_to_buffer(self.query_buffer.buffer(), 0, self.query_readback_buffer.buffer(), 0, None);
    }

    pub fn duration(&self) -> Duration {
        let mut timestamps = [0u64; 2];
        self.query_readback_buffer.read(&self.device, &mut timestamps);
        Duration::from_nanos(timestamps[1] - timestamps[0])
    }
}
