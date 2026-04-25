use wgpu::{ComputePipeline, Device};

use crate::{
    shaders::{
        calculate_grid_size::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            compute::create_calculate_grid_size_pipeline_embed_source,
        },
        common::GridSize,
    },
    typed_buffer::TypedBuffer,
};

pub struct CalculateGridSize {
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
}

impl CalculateGridSize {
    pub fn new(
        device: &Device,
        grid_min_x: TypedBuffer<f32>,
        grid_min_y: TypedBuffer<f32>,
        grid_max_x: TypedBuffer<f32>,
        grid_max_y: TypedBuffer<f32>,
        cell_size: TypedBuffer<f32>,
        grid_size: TypedBuffer<GridSize>,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                grid_min_x: grid_min_x.buffer().as_entire_buffer_binding(),
                grid_min_y: grid_min_y.buffer().as_entire_buffer_binding(),
                grid_max_x: grid_max_x.buffer().as_entire_buffer_binding(),
                grid_max_y: grid_max_y.buffer().as_entire_buffer_binding(),
                cell_size: cell_size.buffer().as_entire_buffer_binding(),
                grid_size: grid_size.buffer().as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_calculate_grid_size_pipeline_embed_source(device);
        Self { bind_group, pipeline }
    }

    pub fn compute(&self, compute_pass: &mut wgpu::ComputePass) {
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }
}
