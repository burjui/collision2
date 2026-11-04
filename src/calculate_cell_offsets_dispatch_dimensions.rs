use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    device_buffer::DeviceBuffer,
    shaders::{
        calculate_cell_offsets_dispatch_dimensions::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            compute::create_calculate_cell_offsets_dispatch_dimensions_pipeline_embed_source,
        },
        common::DispatchIndirectArgs,
    },
    util::ComputeStage,
};

pub struct CellIterationDispatchDimensions {
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
}

pub struct CellIterationDispatchDimensionsParams<'a> {
    pub dispatch_dimensions: &'a DeviceBuffer<DispatchIndirectArgs>,
    pub particle_radius: &'a DeviceBuffer<f32>,
    pub grid_min_x: &'a DeviceBuffer<f32>,
    pub grid_max_x: &'a DeviceBuffer<f32>,
    pub grid_min_y: &'a DeviceBuffer<f32>,
    pub grid_max_y: &'a DeviceBuffer<f32>,
    pub grid_size_x: &'a DeviceBuffer<u32>,
    pub grid_size_y: &'a DeviceBuffer<u32>,
}

impl CellIterationDispatchDimensions {
    pub fn new(
        device: &Device,
        CellIterationDispatchDimensionsParams {
            dispatch_dimensions,
            particle_radius,
            grid_min_x,
            grid_max_x,
            grid_min_y,
            grid_max_y,
            grid_size_x,
            grid_size_y,
        }: CellIterationDispatchDimensionsParams<'_>,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                particle_radius: particle_radius.as_entire_buffer_binding(),
                grid_min_x: grid_min_x.as_entire_buffer_binding(),
                grid_max_x: grid_max_x.as_entire_buffer_binding(),
                grid_min_y: grid_min_y.as_entire_buffer_binding(),
                grid_max_y: grid_max_y.as_entire_buffer_binding(),
                grid_size_x: grid_size_x.as_entire_buffer_binding(),
                grid_size_y: grid_size_y.as_entire_buffer_binding(),
                cell_offsets_dispatch_dimensions: dispatch_dimensions.as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_calculate_cell_offsets_dispatch_dimensions_pipeline_embed_source(device);
        Self { bind_group, pipeline }
    }
}

impl ComputeStage for CellIterationDispatchDimensions {
    const LABEL: &'static str = "Calculate cell offsets dispatch dimensions";

    fn compute(&self, compute_pass: &mut ComputePass) {
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }
}
