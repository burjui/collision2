use wgpu::{ComputePipeline, Device};

use crate::{
    shaders::{
        common::{DispatchIndirectArgs, GridSize},
        reset_cell_object_count::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            compute::create_reset_cell_object_count_pipeline_embed_source,
        },
    },
    typed_buffer::TypedBuffer,
};

pub struct ResetCellObjectCount {
    dispatch_dimensions: TypedBuffer<DispatchIndirectArgs>,
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
}

impl ResetCellObjectCount {
    pub fn new(
        device: &Device,
        dispatch_dimensions: TypedBuffer<DispatchIndirectArgs>,
        grid_size: TypedBuffer<GridSize>,
        cell_object_count: TypedBuffer<u32>,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                grid_size: grid_size.buffer().as_entire_buffer_binding(),
                cell_object_count: cell_object_count.buffer().as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_reset_cell_object_count_pipeline_embed_source(device);
        Self {
            dispatch_dimensions,
            bind_group,
            pipeline,
        }
    }

    pub fn compute(&self, compute_pass: &mut wgpu::ComputePass) {
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);
        compute_pass.dispatch_workgroups_indirect(self.dispatch_dimensions.buffer(), 0);
    }
}
