use std::time::Duration;

use wgpu::{
    BufferUsages, CommandEncoder, ComputePass, Device, QuerySet, QuerySetDescriptor, QueryType, RenderPass, WasmNotSend,
};

use crate::device_buffer::DeviceBuffer;

pub struct PassPipeline<'a> {
    device: &'a Device,
    measurers: Vec<CommandDurationMeasurer>,
}

impl<'a> PassPipeline<'a> {
    pub fn new(device: &'a Device) -> PassPipeline<'a> {
        Self {
            device,
            measurers: Vec::new(),
        }
    }

    pub fn measure_compute(&mut self, compute_pass: &mut ComputePass, label: &str, f: impl FnOnce(&mut ComputePass)) {
        let measurer = CommandDurationMeasurer::new(&self.device, label);
        measurer.measure_compute(compute_pass, |compute_pass| f(compute_pass));
        self.measurers.push(measurer);
    }

    pub fn add_render(&mut self, render_pass: &mut RenderPass, label: &str, f: impl FnOnce(&mut RenderPass)) {
        let measurer = CommandDurationMeasurer::new(&self.device, label);
        measurer.measure_render(render_pass, |render_pass| f(render_pass));
        self.measurers.push(measurer);
    }

    pub fn finish(self, encoder: &mut CommandEncoder) -> CommanPipelineTimings {
        for measurer in &self.measurers {
            measurer.resolve(encoder);
        }
        CommanPipelineTimings {
            measurers: self.measurers,
        }
    }
}

// TODO: avoid allocations

pub struct CommanPipelineTimings {
    measurers: Vec<CommandDurationMeasurer>,
}

impl CommanPipelineTimings {
    pub fn read(self, callback: impl FnOnce(Vec<(String, Duration)>) + WasmNotSend + 'static + Clone) {
        let (tx, rx) = crossbeam::channel::bounded(self.measurers.len());
        for measurer in self.measurers {
            let label = measurer.label.clone();
            measurer.read({
                let tx = tx.clone();
                move |duration| {
                    tx.send((label, duration)).unwrap();
                }
            });
        }
        drop(tx);
        let timings = rx.iter().collect();
        callback(timings);
    }
}

#[must_use]
struct CommandDurationMeasurer {
    label: String,
    query_set: QuerySet,
    query_buffer: DeviceBuffer<u64>,
    query_readback_buffer: DeviceBuffer<u64>,
}

impl CommandDurationMeasurer {
    fn new(device: &Device, label: &str) -> Self {
        Self {
            label: label.to_string(),
            query_set: device.create_query_set(&QuerySetDescriptor {
                label: Some(&format!("{}: query set", label)),
                ty: QueryType::Timestamp,
                count: 2,
            }),
            query_buffer: DeviceBuffer::new(
                device,
                2,
                &format!("{}: query buffer", label),
                BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            ),
            query_readback_buffer: DeviceBuffer::new(
                device,
                2,
                &format!("{}: query readback buffer", label),
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            ),
        }
    }

    fn measure_compute(&self, compute_pass: &mut ComputePass, f: impl FnOnce(&mut ComputePass)) {
        compute_pass.write_timestamp(&self.query_set, 0);
        f(compute_pass);
        compute_pass.write_timestamp(&self.query_set, 1);
    }

    fn measure_render(&self, render_pass: &mut RenderPass, f: impl FnOnce(&mut RenderPass)) {
        render_pass.write_timestamp(&self.query_set, 0);
        f(render_pass);
        render_pass.write_timestamp(&self.query_set, 1);
    }

    fn resolve(&self, encoder: &mut CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..2, self.query_buffer.buffer(), 0);
        encoder.copy_buffer_to_buffer(self.query_buffer.buffer(), 0, self.query_readback_buffer.buffer(), 0, None);
    }

    fn read(&self, callback: impl FnOnce(Duration) + WasmNotSend + 'static) {
        self.query_readback_buffer.read(2, |result| {
            let timestamps = result.unwrap();
            let duration = Duration::from_nanos(timestamps[1] - timestamps[0]);
            callback(duration);
        });
    }
}
