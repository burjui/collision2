use std::mem::size_of;

use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    buffer_sets::BroadPhaseBuffers,
    compute_stage::ComputeStage,
    device_buffer::DeviceBuffer,
    shaders::{
        calculate_cell_offsets::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            compute::create_calculate_cell_offsets_pipeline_embed_source,
        },
        calculate_cell_offsets_dispatch_dimensions::N_CELL_INDIRECT_DISPATCHES,
        common::{DispatchIndirectArgs, MAX_DISPATCH_DIMENSION, WORKGROUP_SIZE},
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
        broad_phase_buffers: &BroadPhaseBuffers,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                grid_min_x: broad_phase_buffers.grid_min_x.as_entire_buffer_binding(),
                grid_min_y: broad_phase_buffers.grid_min_y.as_entire_buffer_binding(),
                grid_size_x: broad_phase_buffers.grid_size_x.as_entire_buffer_binding(),
                grid_size_y: broad_phase_buffers.grid_size_y.as_entire_buffer_binding(),
                cell_object_count: broad_phase_buffers.cell_object_count.as_entire_buffer_binding(),
                current_cell_offset: broad_phase_buffers.current_cell_offset.as_entire_buffer_binding(),
                cell_offsets: broad_phase_buffers.cell_offsets.as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_calculate_cell_offsets_pipeline_embed_source(device);
        Self {
            dispatch_dimensions,
            bind_group,
            pipeline,
        }
    }
}

impl ComputeStage for CalculateCellOffsets {
    const LABEL: &'static str = "Calculate cell offsets";

    fn compute_impl(&self, compute_pass: &mut ComputePass) {
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);

        let dispatch_indirect_args_size: u64 = size_of::<DispatchIndirectArgs>().try_into().unwrap();
        for i in 0..N_CELL_INDIRECT_DISPATCHES {
            let thread_offset = i * MAX_DISPATCH_DIMENSION * WORKGROUP_SIZE;
            compute_pass.set_immediates(0, bytemuck::cast_slice(&[thread_offset]));
            compute_pass.dispatch_workgroups_indirect(
                self.dispatch_dimensions.buffer(),
                u64::from(i) * dispatch_indirect_args_size,
            );
        }
    }
}
