use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    phase_state::PhaseState,
    shaders::{
        collision_narrow_phase::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams, WgpuBindGroup1, WgpuBindGroup1Entries,
            WgpuBindGroup1EntriesParams, WgpuBindGroup2, WgpuBindGroup2Entries, WgpuBindGroup2EntriesParams,
            compute::create_narrow_phase_pipeline_embed_source,
        },
        common::{CollisionCandidate, DispatchIndirectArgs, Mass},
    },
    typed_buffer::TypedBuffer,
    util::PhaseStateCache,
};

pub struct NarrowPhase {
    dispatch_dimensions: TypedBuffer<DispatchIndirectArgs>,
    input_bind_group: WgpuBindGroup1,
    output_bind_group: WgpuBindGroup2,
    pipeline: ComputePipeline,
    masses: TypedBuffer<Mass>,
    phase_state_cache: PhaseStateCache<WgpuBindGroup0>,
}

impl NarrowPhase {
    pub fn new(
        device: &Device,
        dispatch_dimensions: TypedBuffer<DispatchIndirectArgs>,
        candidates: TypedBuffer<CollisionCandidate>,
        candidate_count: TypedBuffer<u32>,
        masses: TypedBuffer<Mass>,
        collision_forces: TypedBuffer<u32>,
    ) -> Self {
        let input_bind_group = WgpuBindGroup1::from_bindings(
            device,
            WgpuBindGroup1Entries::new(WgpuBindGroup1EntriesParams {
                candidates: candidates.buffer().as_entire_buffer_binding(),
                candidate_count: candidate_count.buffer().as_entire_buffer_binding(),
            }),
        );
        let output_bind_group = WgpuBindGroup2::from_bindings(
            device,
            WgpuBindGroup2Entries::new(WgpuBindGroup2EntriesParams {
                collision_forces: collision_forces.buffer().as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_narrow_phase_pipeline_embed_source(device);
        let phase_state_cache = PhaseStateCache::new();
        Self {
            dispatch_dimensions,
            input_bind_group,
            output_bind_group,
            pipeline,
            masses,
            phase_state_cache,
        }
    }

    pub fn prepare(&mut self, device: &Device, phase_state_index: usize, phase_state: &PhaseState) {
        self.phase_state_cache.update(phase_state_index, || {
            WgpuBindGroup0::from_bindings(
                device,
                WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                    aabbs: phase_state.aabbs().buffer().as_entire_buffer_binding(),
                    velocities: phase_state.velocities().buffer().as_entire_buffer_binding(),
                    masses: self.masses.buffer().as_entire_buffer_binding(),
                }),
            )
        });
    }

    pub fn compute(&self, compute_pass: &mut ComputePass) {
        let pipeline = self.pipeline.clone();
        let phase_state_bind_group = self.phase_state_cache.get_current();
        compute_pass.set_pipeline(&pipeline);
        phase_state_bind_group.set(compute_pass);
        self.input_bind_group.set(compute_pass);
        self.output_bind_group.set(compute_pass);
        compute_pass.dispatch_workgroups_indirect(self.dispatch_dimensions.buffer(), 0);
    }
}
