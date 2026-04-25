use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    phase_state::PhaseState,
    shaders::{
        build_bvh::{
            CombineNodePass, WORKGROUP_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            WgpuBindGroup1, WgpuBindGroup1Entries, WgpuBindGroup1EntriesParams,
            compute::create_combine_nodes_pipeline_embed_source,
        },
        common::BvhNode,
    },
    typed_buffer::TypedBuffer,
    util::{PhaseStateCache, dispatch_dimensions},
};

pub struct BvhBuilder {
    passes: Vec<CombineNodePass>,
    pipeline: ComputePipeline,
    main_bind_group: WgpuBindGroup0,
    phase_state_cache: PhaseStateCache<WgpuBindGroup1>,
}

impl BvhBuilder {
    pub fn new(passes: Vec<CombineNodePass>, device: &Device, nodes: TypedBuffer<BvhNode>) -> Self {
        let pipeline = create_combine_nodes_pipeline_embed_source(device);
        let main_bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                nodes: nodes.buffer().as_entire_buffer_binding(),
            }),
        );
        let phase_state_cache = PhaseStateCache::new();
        Self {
            passes,
            pipeline,
            main_bind_group,
            phase_state_cache,
        }
    }

    pub fn prepare(&mut self, device: &Device, phase_state_index: usize, phase_state: &PhaseState) {
        self.phase_state_cache.update(phase_state_index, || {
            WgpuBindGroup1::from_bindings(
                device,
                WgpuBindGroup1Entries::new(WgpuBindGroup1EntriesParams {
                    aabbs: phase_state.aabbs().buffer().as_entire_buffer_binding(),
                }),
            )
        });
    }

    pub fn compute(&mut self, compute_pass: &mut ComputePass) {
        let phase_state_bind_group = self.phase_state_cache.get_current();
        compute_pass.set_pipeline(&self.pipeline);
        self.main_bind_group.set(compute_pass);
        phase_state_bind_group.set(compute_pass);
        for &pass in &self.passes {
            compute_pass.set_immediates(0, bytemuck::cast_slice(&[pass]));
            let (x, y, z) = dispatch_dimensions(pass.parent_count, WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(x, y, z);
        }
    }

    pub fn node_count(&self) -> usize {
        usize::try_from(self.passes.last().unwrap().dst_start + 1).unwrap()
    }
}

pub struct BvhBuildParameters {
    pub passes: Vec<CombineNodePass>,
    pub node_count: usize,
}

impl BvhBuildParameters {
    pub fn new(n: usize) -> Self {
        assert!(n > 0);
        let mut passes = Vec::new();
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
        let final_pass = &passes[passes.len() - 1];
        let node_count = usize::try_from(final_pass.dst_start + 1).unwrap();
        Self { passes, node_count }
    }
}
