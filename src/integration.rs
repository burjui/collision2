use wgpu::{CommandEncoder, ComputePassDescriptor, ComputePipeline, Device};

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
    bind_group: WgpuBindGroup0,
    object_count: usize,
}

impl GpuIntegrator {
    pub fn new(
        device: &Device,
        dt: GpuBuffer<f32>,
        flags: GpuBuffer<Flags>,
        aabbs: GpuBuffer<AABB>,
        velocities: GpuBuffer<Velocity>,
        masses: GpuBuffer<Mass>,
    ) -> Self {
        let pipeline = create_cs_main_pipeline_embed_source(device);
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                dt: dt.buffer().as_entire_buffer_binding(),
                flags: flags.buffer().as_entire_buffer_binding(),
                aabbs: aabbs.buffer().as_entire_buffer_binding(),
                velocities: velocities.buffer().as_entire_buffer_binding(),
                masses: masses.buffer().as_entire_buffer_binding(),
            }),
        );
        Self {
            pipeline,
            bind_group,
            object_count: flags.len(),
        }
    }

    pub fn compute(&self, encoder: &mut CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(&mut compute_pass);
        let total_workgroups = u32::try_from(self.object_count).unwrap().div_ceil(WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(total_workgroups.min(65535), total_workgroups.div_ceil(65535), 1);
        drop(compute_pass);
    }
}
