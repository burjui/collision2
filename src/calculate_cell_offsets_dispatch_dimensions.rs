use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    buffer_sets::BroadPhaseBuffers,
    compute_stage::ComputeStage,
    device_buffer::DeviceBuffer,
    shaders::{
        calculate_cell_offsets_dispatch_dimensions::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            compute::create_calculate_cell_offsets_dispatch_dimensions_pipeline_embed_source,
        },
        common::DispatchIndirectArgs,
    },
};

pub struct CellIterationDispatchDimensions {
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
}

impl CellIterationDispatchDimensions {
    pub fn new(
        device: &Device,
        broad_phase_buffers: &BroadPhaseBuffers,
        cell_offsets_dispatch_dimensions: DeviceBuffer<DispatchIndirectArgs>,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                particle_radius: broad_phase_buffers.particle_radius.as_entire_buffer_binding(),
                grid_min_x: broad_phase_buffers.grid_min_x.as_entire_buffer_binding(),
                grid_max_x: broad_phase_buffers.grid_max_x.as_entire_buffer_binding(),
                grid_min_y: broad_phase_buffers.grid_min_y.as_entire_buffer_binding(),
                grid_max_y: broad_phase_buffers.grid_max_y.as_entire_buffer_binding(),
                grid_size_x: broad_phase_buffers.grid_size_x.as_entire_buffer_binding(),
                grid_size_y: broad_phase_buffers.grid_size_y.as_entire_buffer_binding(),
                cell_offsets_dispatch_dimensions: cell_offsets_dispatch_dimensions.as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_calculate_cell_offsets_dispatch_dimensions_pipeline_embed_source(device);
        Self { bind_group, pipeline }
    }
}

impl ComputeStage for CellIterationDispatchDimensions {
    const LABEL: &'static str = "Calculate cell offsets dispatch dimensions";

    fn compute_impl(&self, compute_pass: &mut ComputePass) {
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }
}
