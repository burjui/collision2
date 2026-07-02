use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    device_buffer::DeviceBuffer,
    shaders::{
        calculate_cell_offsets::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            compute::create_calculate_cell_offsets_pipeline_embed_source,
        },
        common::{DispatchIndirectArgs, GridSize},
    },
};

pub struct CalculateCellOffsets {
    dispatch_dimensions: DeviceBuffer<DispatchIndirectArgs>,
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
}

impl CalculateCellOffsets {
    pub fn new(
        device: &Device,
        dispatch_dimensions: DeviceBuffer<DispatchIndirectArgs>,
        current_cell_offset: DeviceBuffer<u32>,
        grid_size: DeviceBuffer<GridSize>,
        cell_object_count: DeviceBuffer<u32>,
        cell_offsets: DeviceBuffer<u32>,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                grid_size: grid_size.buffer().as_entire_buffer_binding(),
                cell_object_count: cell_object_count.buffer().as_entire_buffer_binding(),
                current_cell_offset: current_cell_offset.buffer().as_entire_buffer_binding(),
                cell_offsets: cell_offsets.buffer().as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_calculate_cell_offsets_pipeline_embed_source(device);
        Self {
            dispatch_dimensions,
            bind_group,
            pipeline,
        }
    }

    pub fn compute(&self, compute_pass: &mut ComputePass) {
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);
        compute_pass.dispatch_workgroups_indirect(self.dispatch_dimensions.buffer(), 0);
    }
}
