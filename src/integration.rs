use wgpu::{CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, Device, Queue, SubmissionIndex};

use crate::{
    gpu_buffer::GpuBuffer,
    shaders::{
        common::{AABB, Flags, Mass, Velocity},
        integration::{
            WORKGROUP_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            compute::create_cs_main_pipeline_embed_source,
        },
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
        flags: &GpuBuffer<Flags>,
        aabbs: &GpuBuffer<AABB>,
        velocities: &GpuBuffer<Velocity>,
        masses: &GpuBuffer<Mass>,
        processed: &GpuBuffer<u32>,
    ) -> SubmissionIndex {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                dt: dt.buffer().as_entire_buffer_binding(),
                flags: flags.buffer().as_entire_buffer_binding(),
                aabbs: aabbs.buffer().as_entire_buffer_binding(),
                velocities: velocities.buffer().as_entire_buffer_binding(),
                masses: masses.buffer().as_entire_buffer_binding(),
                processed: processed.buffer().as_entire_buffer_binding(),
            }),
        );

        processed.write(queue, &[0]);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        compute_pass.set_pipeline(&self.pipeline);
        bind_group.set(&mut compute_pass);
        let total_workgroups = u32::try_from(aabbs.len()).unwrap().div_ceil(WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(total_workgroups.min(65535), total_workgroups.div_ceil(65535), 1);
        drop(compute_pass);
        queue.submit(Some(encoder.finish()))
    }
}
