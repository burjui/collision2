use std::time::Duration;

use wgpu::{
    BufferUsages, CommandEncoder, ComputePass, Device, QuerySet, QuerySetDescriptor, QueryType, RenderPass, WasmNotSend,
};

use crate::device_buffer::DeviceBuffer;

#[derive(Clone)]
pub struct CommandTimings {
    capacity: u32,
    query_set: QuerySet,
    query_buffer: DeviceBuffer<u64>,
    query_readback_buffer: DeviceBuffer<u64>,
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
        let query_readback_buffer = DeviceBuffer::new(
            device,
            capacity * 2,
            "CommandTimings query readback buffer",
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );
        Self {
            capacity,
            query_set,
            query_buffer,
            query_readback_buffer,
            requests: Vec::new(),
            requests_readback: Vec::new(),
        }
    }

    pub fn measure(&mut self, encoder: &mut CommandEncoder, label: &'static str, f: impl FnOnce(&mut CommandEncoder)) {
        assert!(self.requests.len() < usize::try_from(self.capacity).unwrap());
        let slot = self.request_slot_count();
        self.requests.push(label);
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
        assert!(self.requests.len() < usize::try_from(self.capacity).unwrap());
        let slot = self.request_slot_count();
        self.requests.push(label);
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
        assert!(self.requests.len() < usize::try_from(self.capacity).unwrap());
        let slot = self.request_slot_count();
        self.requests.push(label);
        render_pass.write_timestamp(&self.query_set, slot);
        f(render_pass);
        render_pass.write_timestamp(&self.query_set, slot + 1);
    }

    pub fn resolve(&mut self, encoder: &mut CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..self.request_slot_count(), self.query_buffer.buffer(), 0);
        encoder.copy_buffer_to_buffer(self.query_buffer.buffer(), 0, self.query_readback_buffer.buffer(), 0, None);
        self.requests_readback.clear();
        self.requests_readback.extend(self.requests.drain(..));
    }

    pub fn read(&self, callback: impl FnOnce(Vec<(&'static str, Duration)>) + WasmNotSend + 'static + Clone) {
        let labels = self.requests_readback.clone();
        self.query_readback_buffer.read(self.requests_readback.len() * 2, |result| {
            let timestamps = result.unwrap();
            callback(
                labels
                    .into_iter()
                    .zip(timestamps.chunks(2).map(|chunk| Duration::from_nanos(chunk[1] - chunk[0])))
                    .collect(),
            );
        });
    }

    fn request_slot_count(&self) -> u32 {
        u32::try_from(self.requests.len() * 2).unwrap()
    }
}
