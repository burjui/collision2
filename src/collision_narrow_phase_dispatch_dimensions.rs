use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    gpu_buffer::TypedBuffer,
    shaders::{
        collision_narrow_phase_dispatch_dimensions::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            compute::create_calculate_narrow_phase_dispatch_dimensions_pipeline_embed_source,
        },
        common::DispatchIndirectArgs,
    },
};

pub struct NarrowPhaseDispatchIndirectArgsCalculator {
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
}

impl NarrowPhaseDispatchIndirectArgsCalculator {
    pub fn new(
        device: &Device,
        candidate_count: TypedBuffer<u32>,
        dispatch_dimensions: TypedBuffer<DispatchIndirectArgs>,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                candidate_count: candidate_count.buffer().as_entire_buffer_binding(),
                narrow_phase_dispatch_dimensions: dispatch_dimensions.buffer().as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_calculate_narrow_phase_dispatch_dimensions_pipeline_embed_source(device);
        Self { bind_group, pipeline }
    }

    pub fn compute(&self, compute_pass: &mut ComputePass) {
        let pipeline = self.pipeline.clone();
        compute_pass.set_pipeline(&pipeline);
        self.bind_group.set(compute_pass);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }
}
