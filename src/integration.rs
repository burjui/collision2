use wgpu::{CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, Device, Queue, SubmissionIndex};

use crate::{
    gpu_buffer::GpuBuffer,
    shaders::{
        integration::{
            ComputeMass, ComputePosition, ComputeVelocity, WORKGROUP_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries,
            WgpuBindGroup0EntriesParams, compute::create_cs_main_pipeline_embed_source,
        },
        shape,
    },
};

pub struct GpuIntegrator {
    pipeline: ComputePipeline,
}

impl GpuIntegrator {
    pub fn new(device: &Device) -> Self {
        Self {
            pipeline: create_cs_main_pipeline_embed_source(device),
        }
    }

    pub fn compute(
        &self,
        device: &Device,
        queue: &Queue,
        dt: &GpuBuffer<f32>,
        positions: &GpuBuffer<ComputePosition>,
        velocities: &GpuBuffer<ComputeVelocity>,
        masses: &GpuBuffer<ComputeMass>,
        flags: &GpuBuffer<shape::FlagsInput>,
    ) -> SubmissionIndex {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                dt: dt.buffer().as_entire_buffer_binding(),
                mass: masses.buffer().as_entire_buffer_binding(),
                velocity: velocities.buffer().as_entire_buffer_binding(),
                position: positions.buffer().as_entire_buffer_binding(),
                flags: flags.buffer().as_entire_buffer_binding(),
            }),
        );

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        compute_pass.set_pipeline(&self.pipeline);
        bind_group.set(&mut compute_pass);
        let total_workgroups = u32::try_from(positions.len()).unwrap().div_ceil(WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(total_workgroups.min(65535), total_workgroups.div_ceil(65535), 1);
        drop(compute_pass);
        queue.submit(Some(encoder.finish()))
    }
}
