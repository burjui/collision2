use std::array::from_fn;

use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    gpu_buffer::TypedBuffer,
    phase_state::{PhaseState, PhaseStateRing},
    shaders::{
        collision_narrow_phase::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams, WgpuBindGroup1, WgpuBindGroup1Entries,
            WgpuBindGroup1EntriesParams, WgpuBindGroup2, WgpuBindGroup2Entries, WgpuBindGroup2EntriesParams,
            compute::create_narrow_phase_pipeline_embed_source,
        },
        common::{CollisionCandidate, DispatchIndirectArgs, Mass},
    },
};

pub struct NarrowPhase {
    dispatch_dimensions: TypedBuffer<DispatchIndirectArgs>,
    input_bind_group: WgpuBindGroup1,
    output_bind_group: WgpuBindGroup2,
    pipeline: ComputePipeline,
    masses: TypedBuffer<Mass>,
    phase_state_bind_groups: [Option<WgpuBindGroup0>; PhaseStateRing::CAPACITY],
    phase_state_index: Option<usize>,
}

impl NarrowPhase {
    pub fn new(
        device: &Device,
        dispatch_dimensions: TypedBuffer<DispatchIndirectArgs>,
        candidates: TypedBuffer<CollisionCandidate>,
        candidate_count: TypedBuffer<u32>,
        masses: TypedBuffer<Mass>,
        collision_forces_x: TypedBuffer<u32>,
        collision_forces_y: TypedBuffer<u32>,
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
                collision_forces_x: collision_forces_x.buffer().as_entire_buffer_binding(),
                collision_forces_y: collision_forces_y.buffer().as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_narrow_phase_pipeline_embed_source(device);
        Self {
            dispatch_dimensions,
            input_bind_group,
            output_bind_group,
            pipeline,
            masses,
            phase_state_bind_groups: from_fn(|_| None),
            phase_state_index: None,
        }
    }

    pub fn prepare(&mut self, phase_state_index: usize, device: &Device, phase_state: &PhaseState) {
        self.phase_state_bind_groups[phase_state_index].get_or_insert_with(|| {
            WgpuBindGroup0::from_bindings(
                device,
                WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                    aabbs: phase_state.aabbs().buffer().as_entire_buffer_binding(),
                    velocities: phase_state.velocities().buffer().as_entire_buffer_binding(),
                    masses: self.masses.buffer().as_entire_buffer_binding(),
                }),
            )
        });
        self.phase_state_index = Some(phase_state_index);
    }

    pub fn compute(&self, compute_pass: &mut ComputePass) {
        let pipeline = self.pipeline.clone();
        let phase_state_index = self.phase_state_index.expect("prepare() must be called every frame");
        let phase_state_bind_group = self.phase_state_bind_groups[phase_state_index].as_ref().unwrap();
        compute_pass.set_pipeline(&pipeline);
        phase_state_bind_group.set(compute_pass);
        self.input_bind_group.set(compute_pass);
        self.output_bind_group.set(compute_pass);
        compute_pass.dispatch_workgroups_indirect(self.dispatch_dimensions.buffer(), 0);
    }
}
