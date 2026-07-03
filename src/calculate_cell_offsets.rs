use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    device_buffer::DeviceBuffer,
    shaders::{
        calculate_cell_offsets::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            compute::create_calculate_cell_offsets_pipeline_embed_source,
        },
        common::DispatchIndirectArgs,
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
        grid_min_x: DeviceBuffer<f32>,
        grid_max_x: DeviceBuffer<f32>,
        grid_min_y: DeviceBuffer<f32>,
        grid_max_y: DeviceBuffer<f32>,
        cell_size: DeviceBuffer<f32>,
        cell_object_count: DeviceBuffer<u32>,
        cell_offsets: DeviceBuffer<u32>,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                grid_min_x: grid_min_x.as_entire_buffer_binding(),
                grid_max_x: grid_max_x.as_entire_buffer_binding(),
                grid_min_y: grid_min_y.as_entire_buffer_binding(),
                grid_max_y: grid_max_y.as_entire_buffer_binding(),
                cell_size: cell_size.as_entire_buffer_binding(),
                cell_object_count: cell_object_count.as_entire_buffer_binding(),
                current_cell_offset: current_cell_offset.as_entire_buffer_binding(),
                cell_offsets: cell_offsets.as_entire_buffer_binding(),
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
