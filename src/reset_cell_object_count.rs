use wgpu::{ComputePipeline, Device};

use crate::{
    device_buffer::DeviceBuffer,
    shaders::{
        common::{DispatchIndirectArgs, GridSize},
        reset_cell_object_count::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            compute::create_reset_cell_object_count_pipeline_embed_source,
        },
    },
};

pub struct ResetCellObjectCount {
    dispatch_dimensions: DeviceBuffer<DispatchIndirectArgs>,
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
}

impl ResetCellObjectCount {
    pub fn new(
        device: &Device,
        dispatch_dimensions: DeviceBuffer<DispatchIndirectArgs>,
        grid_size: DeviceBuffer<GridSize>,
        cell_object_count: DeviceBuffer<u32>,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                grid_size: grid_size.as_entire_buffer_binding(),
                cell_object_count: cell_object_count.as_entire_buffer_binding(),
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
