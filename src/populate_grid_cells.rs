use wgpu::{ComputePipeline, Device};

use crate::{
    device_buffer::DeviceBuffer,
    shaders::{
        common::CellPosition,
        populate_grid_cells::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            compute::create_populate_object_cells_pipeline_embed_source,
        },
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
        grid_size_x: DeviceBuffer<u32>,
        object_cells: DeviceBuffer<CellPosition>,
        cell_offsets: DeviceBuffer<u32>,
        cells: DeviceBuffer<u32>,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                object_count: object_count_buffer.as_entire_buffer_binding(),
                grid_size_x: grid_size_x.as_entire_buffer_binding(),
                object_cells: object_cells.as_entire_buffer_binding(),
                cell_offsets: cell_offsets.as_entire_buffer_binding(),
                cells: cells.as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_populate_object_cells_pipeline_embed_source(device);
        Self {
            object_count,
            bind_group,
            pipeline,
        }
    }

    pub fn compute(&self, compute_pass: &mut wgpu::ComputePass) {
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);
        dispatch_compute(compute_pass, self.object_count);
    }
}
