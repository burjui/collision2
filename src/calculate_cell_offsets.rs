use wgpu::{ComputePass, ComputePipeline, Device, Queue};

use crate::{
    shaders::{
        calculate_cell_offsets::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            compute::create_calculate_cell_offsets_pipeline_embed_source,
        },
        common::{DispatchIndirectArgs, GridSize},
    },
    typed_buffer::TypedBuffer,
};

pub struct CalculateCellOffsets {
    dispatch_dimensions: TypedBuffer<DispatchIndirectArgs>,
    current_cell_offset: TypedBuffer<u32>,
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
}

impl CalculateCellOffsets {
    pub fn new(
        device: &Device,
        dispatch_dimensions: TypedBuffer<DispatchIndirectArgs>,
        current_cell_offset: TypedBuffer<u32>,
        grid_size: TypedBuffer<GridSize>,
        cell_object_count: TypedBuffer<u32>,
        cell_offsets: TypedBuffer<u32>,
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
            current_cell_offset,
            bind_group,
            pipeline,
        }
    }

    pub fn compute(&self, compute_pass: &mut ComputePass, queue: &Queue) {
        self.current_cell_offset.write(queue, &[0]);
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);
        compute_pass.dispatch_workgroups_indirect(self.dispatch_dimensions.buffer(), 0);
    }
}
