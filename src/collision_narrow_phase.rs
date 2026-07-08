use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    device_buffer::DeviceBuffer,
    phase_state::{PhaseState, PhaseStateRingConfig},
    shaders::{
        collision_narrow_phase::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams, WgpuBindGroup1, WgpuBindGroup1Entries,
            WgpuBindGroup1EntriesParams, WgpuBindGroup2, WgpuBindGroup2Entries, WgpuBindGroup2EntriesParams,
            WgpuBindGroup3, WgpuBindGroup3Entries, WgpuBindGroup3EntriesParams,
            compute::create_narrow_phase_pipeline_embed_source,
        },
        common::{CollisionCandidate, DispatchIndirectArgs, Mass},
    },
    util::PhaseStateCache,
};

pub struct NarrowPhase {
    dispatch_dimensions: DeviceBuffer<DispatchIndirectArgs>,
    constants_bind_group: WgpuBindGroup0,
    input_bind_group: WgpuBindGroup2,
    output_bind_group: WgpuBindGroup3,
    pipeline: ComputePipeline,
    masses: DeviceBuffer<Mass>,
    phase_state_cache: PhaseStateCache<WgpuBindGroup1>,
}

impl NarrowPhase {
    pub fn new(
        device: &Device,
        dispatch_dimensions: DeviceBuffer<DispatchIndirectArgs>,
        stiffness: DeviceBuffer<f32>,
        restitution: DeviceBuffer<f32>,
        candidates: DeviceBuffer<CollisionCandidate>,
        candidate_count: DeviceBuffer<u32>,
        masses: DeviceBuffer<Mass>,
        collision_forces: DeviceBuffer<u32>,
        phase_state_ring_config: PhaseStateRingConfig,
    ) -> Self {
        let constants_bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                stiffness: stiffness.as_entire_buffer_binding(),
                restitution: restitution.as_entire_buffer_binding(),
            }),
        );
        let input_bind_group = WgpuBindGroup2::from_bindings(
            device,
            WgpuBindGroup2Entries::new(WgpuBindGroup2EntriesParams {
                candidates: candidates.as_entire_buffer_binding(),
                candidate_count: candidate_count.as_entire_buffer_binding(),
            }),
        );
        let output_bind_group = WgpuBindGroup3::from_bindings(
            device,
            WgpuBindGroup3Entries::new(WgpuBindGroup3EntriesParams {
                collision_forces: collision_forces.as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_narrow_phase_pipeline_embed_source(device);
        let phase_state_cache = PhaseStateCache::new(phase_state_ring_config);
        Self {
            dispatch_dimensions,
            constants_bind_group,
            input_bind_group,
            output_bind_group,
            pipeline,
            masses,
            phase_state_cache,
        }
    }

    pub fn prepare(&mut self, device: &Device, phase_state_index: usize, phase_state: &PhaseState) {
        self.phase_state_cache.update(phase_state_index, || {
            WgpuBindGroup1::from_bindings(
                device,
                WgpuBindGroup1Entries::new(WgpuBindGroup1EntriesParams {
                    aabbs: phase_state.aabbs().as_entire_buffer_binding(),
                    velocities: phase_state.velocities().as_entire_buffer_binding(),
                    masses: self.masses.as_entire_buffer_binding(),
                }),
            )
        });
    }

    pub fn compute(&self, compute_pass: &mut ComputePass) {
        let pipeline = self.pipeline.clone();
        let phase_state_bind_group = self.phase_state_cache.get_current();
        compute_pass.set_pipeline(&pipeline);
        phase_state_bind_group.set(compute_pass);
        self.constants_bind_group.set(compute_pass);
        self.input_bind_group.set(compute_pass);
        self.output_bind_group.set(compute_pass);
        compute_pass.dispatch_workgroups_indirect(self.dispatch_dimensions.buffer(), 0);
    }
}
