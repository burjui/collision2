use std::array::from_fn;

use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    gpu_buffer::GpuBuffer,
    phase_state::{PhaseState, PhaseStateRing},
    shaders::{
        bvh::{
            CombineNodePass, WORKGROUP_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            WgpuBindGroup1, WgpuBindGroup1Entries, WgpuBindGroup1EntriesParams,
            compute::create_combine_nodes_pipeline_embed_source,
        },
        common::BvhNode,
    },
};

pub struct BvhBuilder {
    passes: Vec<CombineNodePass>,
    pipeline: ComputePipeline,
    fixed_bind_group: WgpuBindGroup0,
    phase_state_bind_groups: [Option<WgpuBindGroup1>; PhaseStateRing::CAPACITY],
    phase_state_index: Option<usize>,
}

impl BvhBuilder {
    pub fn new(params: BvhBuildParameters, device: &Device, nodes: GpuBuffer<BvhNode>) -> Self {
        let pipeline = create_combine_nodes_pipeline_embed_source(device);
        let fixed_bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                nodes: nodes.buffer().as_entire_buffer_binding(),
            }),
        );
        Self {
            passes: params.passes,
            pipeline,
            fixed_bind_group,
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
                }),
            )
        });
        self.phase_state_index = Some(phase_state_index);
    }

    pub fn compute(&mut self, compute_pass: &mut ComputePass) {
        let phase_state_index = self.phase_state_index.expect("prepare() must be called every frame");
        let phase_state_bind_group = self.phase_state_bind_groups[phase_state_index].as_ref().unwrap();
        compute_pass.set_pipeline(&self.pipeline);
        self.fixed_bind_group.set(compute_pass);
        phase_state_bind_group.set(compute_pass);
        for &pass in &self.passes {
            compute_pass.set_push_constants(0, bytemuck::cast_slice(&[pass]));
            let total_workgroups = pass.parent_count.div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(total_workgroups.min(65535), total_workgroups.div_ceil(65535), 1);
        }
    }

    pub fn node_count(&self) -> usize {
        usize::try_from(self.passes.last().unwrap().dst_start + 1).unwrap()
    }
}

pub struct BvhBuildParameters {
    passes: Vec<CombineNodePass>,
    node_count: usize,
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

    pub fn passes(&self) -> &[CombineNodePass] {
        &self.passes
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }
}
