use wgpu::{ComputePipeline, Device};

use crate::{
    buffer_sets::BroadPhaseBuffers,
    compute_stage::ComputeStage,
    shaders::reset_grid_aabb::{
        WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
        compute::create_reset_grid_aabb_pipeline_embed_source,
    },
};

pub struct ResetGridAABB {
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
}

impl ResetGridAABB {
    pub fn new(device: &Device, broad_phase_buffers: &BroadPhaseBuffers) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                grid_min_x: broad_phase_buffers.grid_min_x.as_entire_buffer_binding(),
                grid_max_x: broad_phase_buffers.grid_max_x.as_entire_buffer_binding(),
                grid_min_y: broad_phase_buffers.grid_min_y.as_entire_buffer_binding(),
                grid_max_y: broad_phase_buffers.grid_max_y.as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_reset_grid_aabb_pipeline_embed_source(device);
        Self { bind_group, pipeline }
    }
}

impl ComputeStage for ResetGridAABB {
    const LABEL: &'static str = "Reset grid AABB";

    fn compute_impl(&self, compute_pass: &mut wgpu::ComputePass) {
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }
}
