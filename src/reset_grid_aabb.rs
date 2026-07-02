use wgpu::{ComputePipeline, Device};

use crate::{
    device_buffer::DeviceBuffer,
    shaders::{
        common::AABB,
        reset_grid_aabb::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            compute::create_reset_grid_aabb_pipeline_embed_source,
        },
    },
};

pub struct ResetGridAABB {
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
}

impl ResetGridAABB {
    pub fn new(
        device: &Device,
        first_aabb: DeviceBuffer<AABB>,
        grid_min_x: DeviceBuffer<f32>,
        grid_min_y: DeviceBuffer<f32>,
        grid_max_x: DeviceBuffer<f32>,
        grid_max_y: DeviceBuffer<f32>,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                first_aabb: first_aabb.buffer().as_entire_buffer_binding(),
                grid_min_x: grid_min_x.buffer().as_entire_buffer_binding(),
                grid_min_y: grid_min_y.buffer().as_entire_buffer_binding(),
                grid_max_x: grid_max_x.buffer().as_entire_buffer_binding(),
                grid_max_y: grid_max_y.buffer().as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_reset_grid_aabb_pipeline_embed_source(device);
        Self { bind_group, pipeline }
    }

    pub fn compute(&self, compute_pass: &mut wgpu::ComputePass) {
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }
}
