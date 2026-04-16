use std::sync::Arc;

use wgpu::{BufferUsages, ComputePass, ComputePipeline, Device, Queue};

use crate::{
    gpu_buffer::TypedBuffer,
    shaders::{
        collision_broad_phase::{
            BATCH_SIZE, WORKGROUP_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            WgpuBindGroup1, WgpuBindGroup1Entries, WgpuBindGroup1EntriesParams,
            compute::create_broad_phase_pipeline_embed_source,
        },
        common::{AABB, BvhNode, CollisionCandidate, Flags},
    },
    util::dispatch_dimensions,
};

pub struct BroadPhase {
    object_count: u32,
    candidate_bind_group: Arc<WgpuBindGroup0>,
    phase_state_bind_group: Arc<WgpuBindGroup1>,
    pipeline: ComputePipeline,
    candidate_count: TypedBuffer<u32>,
}

impl BroadPhase {
    pub fn new(
        device: &Device,
        object_count: usize,
        candidates: TypedBuffer<CollisionCandidate>,
        candidate_count: TypedBuffer<u32>,
        nodes: TypedBuffer<BvhNode>,
        aabbs: TypedBuffer<AABB>,
        flags: TypedBuffer<Flags>,
    ) -> Self {
        let object_count_buffer =
            TypedBuffer::from_data(device, &[object_count], "object count", BufferUsages::UNIFORM);
        let max_candidates_buffer = TypedBuffer::from_data(device, &[0], "max candidates", BufferUsages::UNIFORM);
        let candidate_bind_group = Arc::new(WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                object_count: object_count_buffer.buffer().as_entire_buffer_binding(),
                max_candidates: max_candidates_buffer.buffer().as_entire_buffer_binding(),
                candidates: candidates.buffer().as_entire_buffer_binding(),
                candidate_count: candidate_count.buffer().as_entire_buffer_binding(),
            }),
        ));
        let phase_state_bind_group = Arc::new(WgpuBindGroup1::from_bindings(
            device,
            WgpuBindGroup1Entries::new(WgpuBindGroup1EntriesParams {
                nodes: nodes.buffer().as_entire_buffer_binding(),
                aabbs: aabbs.buffer().as_entire_buffer_binding(),
                flags: flags.buffer().as_entire_buffer_binding(),
            }),
        ));
        let pipeline = create_broad_phase_pipeline_embed_source(device);
        Self {
            object_count: object_count.try_into().unwrap(),
            candidate_bind_group,
            phase_state_bind_group,
            pipeline,
            candidate_count,
        }
    }

    pub fn compute(&self, queue: &Queue, compute_pass: &mut ComputePass) {
        let pipeline = self.pipeline.clone();
        let batch_count = self.object_count.div_ceil(BATCH_SIZE);
        let candidate_bind_group = self.candidate_bind_group.clone();
        let phase_state_bind_group = self.phase_state_bind_group.clone();
        self.candidate_count.write(queue, &[0]);
        compute_pass.set_pipeline(&pipeline);
        candidate_bind_group.set(compute_pass);
        phase_state_bind_group.set(compute_pass);
        let (x, y, z) = dispatch_dimensions(batch_count, WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(x, y, z);
    }
}
