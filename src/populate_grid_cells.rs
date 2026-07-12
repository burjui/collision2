use wgpu::{ComputePipeline, Device};

use crate::{
    buffer_sets::BroadPhaseBuffers,
    compute_stage::ComputeStage,
    device_buffer::DeviceBuffer,
    shaders::populate_grid_cells::{
        WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
        compute::create_populate_object_cells_pipeline_embed_source,
    },
    util::dispatch_compute,
};

pub struct PopulateGridCells {
    object_count: u32,
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
}

impl PopulateGridCells {
    pub fn new(
        device: &Device,
        object_count: u32,
        object_count_buffer: DeviceBuffer<u32>,
        broad_phase_buffers: &BroadPhaseBuffers,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                object_count: object_count_buffer.as_entire_buffer_binding(),
                grid_size_x: broad_phase_buffers.grid_size_x.as_entire_buffer_binding(),
                object_cells: broad_phase_buffers.object_cells.as_entire_buffer_binding(),
                cell_offsets: broad_phase_buffers.cell_offsets.as_entire_buffer_binding(),
                cells: broad_phase_buffers.cells.as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_populate_object_cells_pipeline_embed_source(device);
        Self {
            object_count,
            bind_group,
            pipeline,
        }
    }
}

impl ComputeStage for PopulateGridCells {
    const LABEL: &'static str = "Populate grid cells";

    fn compute_impl(&self, compute_pass: &mut wgpu::ComputePass) {
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);
        dispatch_compute(compute_pass, self.object_count);
    }
}
