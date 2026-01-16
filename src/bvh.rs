use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    gpu_buffer::GpuBuffer,
    shaders::{
        bvh::{
            CombineNodePass, WORKGROUP_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            compute::create_combine_nodes_pipeline_embed_source,
        },
        common::{AABB, BvhNode},
    },
};

pub struct BvhBuilder {
    pipeline: ComputePipeline,
    bind_group: WgpuBindGroup0,
    object_count: usize,
    passes: Vec<CombineNodePass>,
}

impl BvhBuilder {
    pub fn new(device: &Device, aabbs: GpuBuffer<AABB>, nodes: GpuBuffer<BvhNode>, object_count: usize) -> Self {
        let pipeline = create_combine_nodes_pipeline_embed_source(device);
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                aabbs: aabbs.buffer().as_entire_buffer_binding(),
                nodes: nodes.buffer().as_entire_buffer_binding(),
            }),
        );
        Self {
            pipeline,
            bind_group,
            object_count,
            passes: Vec::new(),
        }
    }

    pub fn compute(&mut self, compute_pass: &mut ComputePass) {
        self.passes.clear();
        calculate_passes(self.object_count, &mut self.passes);
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);

        for &pass in &self.passes {
            compute_pass.set_push_constants(0, bytemuck::cast_slice(&[pass]));
            let total_workgroups = pass.parent_count.div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(total_workgroups.min(65535), total_workgroups.div_ceil(65535), 1);
        }
    }

    pub fn node_count(&self) -> u32 {
        self.passes.last().unwrap().dst_start
    }
}

pub fn calculate_passes(n: usize, passes: &mut Vec<CombineNodePass>) {
    let mut src_range = 0..n;
    while src_range.len() > 1 {
        let parent_count = src_range.len() / 2;
        passes.push(CombineNodePass {
            src_start: u32::try_from(src_range.start).unwrap(),
            dst_start: u32::try_from(src_range.end).unwrap(),
            parent_count: u32::try_from(parent_count).unwrap(),
        });
        let next_start = src_range.start + parent_count * 2;
        let leftovers = src_range.end - next_start;
        let next_end = next_start + parent_count + leftovers;
        src_range = next_start..next_end;
    }
}
