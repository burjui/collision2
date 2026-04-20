use std::array::from_fn;

use wgpu::{BufferUsages, ComputePass, ComputePipeline, Device, Queue};

use crate::{
    gpu_buffer::TypedBuffer,
    phase_state::{PhaseState, PhaseStateRing},
    shaders::{
        collision_broad_phase::{
            BATCH_SIZE, WORKGROUP_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            WgpuBindGroup1, WgpuBindGroup1Entries, WgpuBindGroup1EntriesParams,
            compute::create_broad_phase_pipeline_embed_source,
        },
        common::{BvhNode, CollisionCandidate},
    },
    util::dispatch_dimensions,
};

pub struct BroadPhase {
    object_count: u32,
    candidate_bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
    candidate_count: TypedBuffer<u32>,
    phase_state_bind_groups: [Option<WgpuBindGroup1>; PhaseStateRing::CAPACITY],
    phase_state_index: Option<usize>,
}

impl BroadPhase {
    pub fn new(
        device: &Device,
        object_count: usize,
        candidates: TypedBuffer<CollisionCandidate>,
        candidate_count: TypedBuffer<u32>,
        nodes: TypedBuffer<BvhNode>,
    ) -> Self {
        let object_count_buffer =
            TypedBuffer::from_data(device, &[object_count], "object count", BufferUsages::UNIFORM);
        let max_candidates_buffer = TypedBuffer::from_data(device, &[0], "max candidates", BufferUsages::UNIFORM);
        let candidate_bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                object_count: object_count_buffer.buffer().as_entire_buffer_binding(),
                max_candidates: max_candidates_buffer.buffer().as_entire_buffer_binding(),
                candidates: candidates.buffer().as_entire_buffer_binding(),
                candidate_count: candidate_count.buffer().as_entire_buffer_binding(),
                nodes: nodes.buffer().as_entire_buffer_binding(),
            }),
        );

        let pipeline = create_broad_phase_pipeline_embed_source(device);
        Self {
            object_count: object_count.try_into().unwrap(),
            candidate_bind_group,
            pipeline,
            candidate_count,
            phase_state_bind_groups: from_fn(|_| None),
            phase_state_index: None,
        }
    }

    pub fn prepare(&mut self, phase_state_index: usize, device: &Device, phase_state: &PhaseState) {
        self.phase_state_bind_groups[phase_state_index].get_or_insert_with(|| {
            WgpuBindGroup1::from_bindings(
                device,
                WgpuBindGroup1Entries::new(WgpuBindGroup1EntriesParams {
                    aabbs: phase_state.aabbs().buffer().as_entire_buffer_binding(),
                    flags: phase_state.flags().buffer().as_entire_buffer_binding(),
                }),
            )
        });
        self.phase_state_index = Some(phase_state_index);
    }

    pub fn compute(&self, queue: &Queue, compute_pass: &mut ComputePass) {
        let pipeline = self.pipeline.clone();
        let batch_count = self.object_count.div_ceil(BATCH_SIZE);
        let phase_state_index = self.phase_state_index.expect("prepare() must be called every frame");
        let phase_state_bind_group = self.phase_state_bind_groups[phase_state_index].as_ref().unwrap();
        self.candidate_count.write(queue, &[0]);
        compute_pass.set_pipeline(&pipeline);
        self.candidate_bind_group.set(compute_pass);
        phase_state_bind_group.set(compute_pass);
        let (x, y, z) = dispatch_dimensions(batch_count, WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(x, y, z);
    }
}
