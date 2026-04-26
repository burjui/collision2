use wgpu::{ComputePipeline, Device};

use crate::{
    shaders::{
        calculate_cell_iteration_dispatch_dimensions::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            compute::create_calculate_cell_iteration_dispatch_dimensions_pipeline_embed_source,
        },
        common::{DispatchIndirectArgs, GridSize},
    },
    typed_buffer::TypedBuffer,
};

pub struct CalculateCellIterationDispatchDimensions {
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
}

impl CalculateCellIterationDispatchDimensions {
    pub fn new(
        device: &Device,
        grid_size: TypedBuffer<GridSize>,
        cell_offsets_dispatch_dimensions: TypedBuffer<DispatchIndirectArgs>,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                grid_size: grid_size.buffer().as_entire_buffer_binding(),
                cell_offsets_dispatch_dimensions: cell_offsets_dispatch_dimensions.buffer().as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_calculate_cell_iteration_dispatch_dimensions_pipeline_embed_source(device);
        Self { bind_group, pipeline }
    }

    pub fn compute(&self, compute_pass: &mut wgpu::ComputePass) {
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }
}
