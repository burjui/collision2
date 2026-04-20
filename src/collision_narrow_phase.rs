use std::array::from_fn;

use wgpu::{BufferUsages, ComputePass, ComputePipeline, Device, Queue};

use crate::{
    gpu_buffer::TypedBuffer,
    phase_state::{PhaseState, PhaseStateRing},
    shaders::{
        collision_narrow_phase::{
            BATCH_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams, WgpuBindGroup1,
            WgpuBindGroup1Entries, WgpuBindGroup1EntriesParams, compute::create_narrow_phase_pipeline_embed_source,
        },
        common::{AABB, CollisionCandidate, Flags, Force, Mass, Velocity},
    },
};

pub struct BroadPhase {
    output_bind_group: WgpuBindGroup1,
    pipeline: ComputePipeline,
    candidates: TypedBuffer<CollisionCandidate>,
    candidate_count: TypedBuffer<u32>,
    masses: TypedBuffer<Mass>,
    phase_state_bind_groups: [Option<WgpuBindGroup0>; PhaseStateRing::CAPACITY],
    phase_state_index: Option<usize>,
}

impl BroadPhase {
    pub fn new(
        device: &Device,
        candidates: TypedBuffer<CollisionCandidate>,
        candidate_count: TypedBuffer<u32>,
        masses: TypedBuffer<Mass>,
        forces: TypedBuffer<Force>,
    ) -> Self {
        let interaction_count = TypedBuffer::from_data(device, &[0], "interaction count", BufferUsages::STORAGE);
        let output_bind_group = WgpuBindGroup1::from_bindings(
            device,
            WgpuBindGroup1Entries::new(WgpuBindGroup1EntriesParams {
                interaction_count: interaction_count.buffer().as_entire_buffer_binding(),
                forces: forces.buffer().as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_narrow_phase_pipeline_embed_source(device);
        Self {
            output_bind_group,
            pipeline,
            candidates,
            candidate_count,
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
                    candidates: self.candidates.buffer().as_entire_buffer_binding(),
                    candidate_count: self.candidate_count.buffer().as_entire_buffer_binding(),
                    aabbs: phase_state.aabbs().buffer().as_entire_buffer_binding(),
                    velocities: phase_state.velocities().buffer().as_entire_buffer_binding(),
                    masses: self.masses.buffer().as_entire_buffer_binding(),
                }),
            )
        });
        self.phase_state_index = Some(phase_state_index);
    }

    pub fn compute(&self, queue: &Queue, compute_pass: &mut ComputePass) {
        // let pipeline = self.pipeline.clone();
        // let batch_count = self.object_count.div_ceil(BATCH_SIZE);
        // let phase_state_index = self.phase_state_index.expect("prepare() must be called every frame");
        // let phase_state_bind_group = self.phase_state_bind_groups[phase_state_index].as_ref().unwrap();
        // self.candidate_count.write(queue, &[0]);
        // compute_pass.set_pipeline(&pipeline);
        // self.candidate_bind_group.set(compute_pass);
        // phase_state_bind_group.set(compute_pass);
        // let (x, y, z) = dispatch_dimensions(batch_count, WORKGROUP_SIZE);
        // compute_pass.dispatch_workgroups_indirect(x, y, z);
    }
}
