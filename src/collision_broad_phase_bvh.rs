use wgpu::{BufferUsages, ComputePass, ComputePipeline, Device, Queue};

use crate::{
    phase_state::PhaseState,
    shaders::{
        collision_broad_phase_bvh::{
            WORKGROUP_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams, WgpuBindGroup1,
            WgpuBindGroup1Entries, WgpuBindGroup1EntriesParams, compute::create_broad_phase_pipeline_embed_source,
        },
        common::{BvhNode, CollisionCandidate, MAX_CANDIDATES_PER_OBJECT},
    },
    typed_buffer::TypedBuffer,
    util::{PhaseStateCache, dispatch_dimensions},
};

pub struct BroadPhaseBVH {
    object_count: u32,
    candidate_bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
    candidate_count: TypedBuffer<u32>,
    phase_state_cache: PhaseStateCache<WgpuBindGroup1>,
}

impl BroadPhaseBVH {
    pub fn new(
        device: &Device,
        object_count: usize,
        object_count_buffer: TypedBuffer<u32>,
        candidates: TypedBuffer<CollisionCandidate>,
        candidate_count: TypedBuffer<u32>,
        nodes: TypedBuffer<BvhNode>,
    ) -> Self {
        let object_count: u32 = object_count.try_into().unwrap();
        let max_candidates_buffer: TypedBuffer<u32> = TypedBuffer::from_data(
            device,
            &[object_count * MAX_CANDIDATES_PER_OBJECT],
            "max candidates",
            BufferUsages::UNIFORM,
        );
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
        let phase_state_cache = PhaseStateCache::new();
        Self {
            object_count,
            candidate_bind_group,
            pipeline,
            candidate_count,
            phase_state_cache,
        }
    }

    pub fn prepare(&mut self, device: &Device, phase_state_index: usize, phase_state: &PhaseState) {
        self.phase_state_cache.update(phase_state_index, || {
            WgpuBindGroup1::from_bindings(
                device,
                WgpuBindGroup1Entries::new(WgpuBindGroup1EntriesParams {
                    aabbs: phase_state.aabbs().buffer().as_entire_buffer_binding(),
                    flags: phase_state.flags().buffer().as_entire_buffer_binding(),
                }),
            )
        });
    }

    pub fn compute(&self, queue: &Queue, compute_pass: &mut ComputePass) {
        let pipeline = self.pipeline.clone();
        let phase_state_bind_group = self.phase_state_cache.get_current();
        self.candidate_count.write(queue, &[0]);
        compute_pass.set_pipeline(&pipeline);
        self.candidate_bind_group.set(compute_pass);
        phase_state_bind_group.set(compute_pass);
        let (x, y, z) = dispatch_dimensions(self.object_count, WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(x, y, z);
    }
}
