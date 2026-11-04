use wgpu::{ComputePipeline, Device};

use crate::{
    device_buffer::DeviceBuffer,
    shaders::reset_grid_aabb::{
        WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
        compute::create_reset_grid_aabb_pipeline_embed_source,
    },
    util::ComputeStage,
};

pub struct ResetGridAABB {
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
}

pub struct ResetGridAABBParams<'a> {
    pub grid_min_x: &'a DeviceBuffer<f32>,
    pub grid_max_x: &'a DeviceBuffer<f32>,
    pub grid_min_y: &'a DeviceBuffer<f32>,
    pub grid_max_y: &'a DeviceBuffer<f32>,
}

impl ResetGridAABB {
    pub fn new(
        device: &Device,
        ResetGridAABBParams {
            grid_min_x,
            grid_max_x,
            grid_min_y,
            grid_max_y,
        }: ResetGridAABBParams<'_>,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                grid_min_x: grid_min_x.as_entire_buffer_binding(),
                grid_max_x: grid_max_x.as_entire_buffer_binding(),
                grid_min_y: grid_min_y.as_entire_buffer_binding(),
                grid_max_y: grid_max_y.as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_reset_grid_aabb_pipeline_embed_source(device);
        Self { bind_group, pipeline }
    }
}

impl ComputeStage for ResetGridAABB {
    const LABEL: &'static str = "Reset grid AABB";

    fn compute(&self, compute_pass: &mut wgpu::ComputePass) {
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }
}
