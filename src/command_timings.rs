use std::time::Duration;

use wgpu::{
    BufferUsages, CommandEncoder, ComputePass, Device, QuerySet, QuerySetDescriptor, QueryType, RenderPass, WasmNotSend,
};

use crate::device_buffer::DeviceBuffer;

#[derive(Clone)]
pub struct CommandTimings {
    device: Device,
    capacity: u32,
    query_set: QuerySet,
    query_buffer: DeviceBuffer<u64>,
    requests: Vec<&'static str>,
    requests_readback: Vec<&'static str>,
}

impl CommandTimings {
    pub fn new(device: &Device, capacity: u32) -> CommandTimings {
        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some("CommandTimings query set"),
            ty: QueryType::Timestamp,
            count: capacity * 2,
        });
        let query_buffer = DeviceBuffer::new(
            device,
            capacity * 2,
            "CommandTimings query buffer",
            BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
        );
        Self {
            device: device.clone(),
            capacity,
            query_set,
            query_buffer,
            requests: Vec::new(),
            requests_readback: Vec::new(),
        }
    }

    pub fn measure(&mut self, encoder: &mut CommandEncoder, label: &'static str, f: impl FnOnce(&mut CommandEncoder)) {
        let slot = self.request_slot(label);
        encoder.write_timestamp(&self.query_set, slot);
        f(encoder);
        encoder.write_timestamp(&self.query_set, slot + 1);
    }

    pub fn measure_compute(
        &mut self,
        compute_pass: &mut ComputePass,
        label: &'static str,
        f: impl FnOnce(&mut ComputePass),
    ) {
        let slot = self.request_slot(label);
        compute_pass.write_timestamp(&self.query_set, slot);
        f(compute_pass);
        compute_pass.write_timestamp(&self.query_set, slot + 1);
    }

    pub fn measure_render(
        &mut self,
        render_pass: &mut RenderPass,
        label: &'static str,
        f: impl FnOnce(&mut RenderPass),
    ) {
        let slot = self.request_slot(label);
        render_pass.write_timestamp(&self.query_set, slot);
        f(render_pass);
        render_pass.write_timestamp(&self.query_set, slot + 1);
    }

    pub fn resolve(&mut self, encoder: &mut CommandEncoder, timestamp_period: f32) -> CommandTimingsReader {
        encoder.resolve_query_set(&self.query_set, 0..self.request_slot_count(), self.query_buffer.buffer(), 0);

        let query_readback_buffer = DeviceBuffer::new(
            &self.device,
            self.capacity * 2,
            "CommandTimings query readback buffer",
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );
        encoder.copy_buffer_to_buffer(self.query_buffer.buffer(), 0, query_readback_buffer.buffer(), 0, None);
        self.requests_readback.clear();
        self.requests_readback.extend(self.requests.drain(..));
        CommandTimingsReader {
            query_readback_buffer: query_readback_buffer,
            requests_readback: self.requests_readback.clone(),
            timestamp_period,
        }
    }

    fn request_slot(&mut self, label: &'static str) -> u32 {
        assert!(self.requests.len() < usize::try_from(self.capacity).unwrap());
        let slot = self.request_slot_count();
        self.requests.push(label);
        slot
    }

    fn request_slot_count(&self) -> u32 {
        u32::try_from(self.requests.len() * 2).unwrap()
    }
}

pub struct CommandTimingsReader {
    query_readback_buffer: DeviceBuffer<u64>,
    requests_readback: Vec<&'static str>,
    timestamp_period: f32,
}

impl CommandTimingsReader {
    pub fn read(&self, callback: impl FnOnce(Vec<(&'static str, Duration)>) + WasmNotSend + 'static + Clone) {
        let labels = self.requests_readback.clone();
        let timestamp_period = self.timestamp_period;
        self.query_readback_buffer.read(self.requests_readback.len() * 2, move |result| {
            let timestamps = result.unwrap();
            callback(
                labels
                    .into_iter()
                    .zip(timestamps.chunks(2).map(|chunk| {
                        let nanos = (chunk[1] - chunk[0]) as f32 * timestamp_period;
                        Duration::from_nanos(nanos as u64)
                    }))
                    .collect(),
            );
        });
    }
}
