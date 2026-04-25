use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    phase_state::PhaseState,
    shaders::calculate_grid_aabb::{
        WORKGROUP_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams, WgpuBindGroup1,
        WgpuBindGroup1Entries, WgpuBindGroup1EntriesParams, compute::create_calculate_grid_aabb_pipeline_embed_source,
    },
    typed_buffer::TypedBuffer,
    util::{PhaseStateCache, dispatch_dimensions},
};

pub struct CalculateGridAABB {
    object_count: u32,
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
    phase_state_cache: PhaseStateCache<WgpuBindGroup1>,
}

impl CalculateGridAABB {
    pub fn new(
        device: &Device,
        object_count: usize,
        object_count_buffer: TypedBuffer<u32>,
        grid_min_x: TypedBuffer<f32>,
        grid_min_y: TypedBuffer<f32>,
        grid_max_x: TypedBuffer<f32>,
        grid_max_y: TypedBuffer<f32>,
    ) -> Self {
        let object_count: u32 = object_count.try_into().unwrap();
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                object_count: object_count_buffer.buffer().as_entire_buffer_binding(),
                grid_min_x: grid_min_x.buffer().as_entire_buffer_binding(),
                grid_min_y: grid_min_y.buffer().as_entire_buffer_binding(),
                grid_max_x: grid_max_x.buffer().as_entire_buffer_binding(),
                grid_max_y: grid_max_y.buffer().as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_calculate_grid_aabb_pipeline_embed_source(device);
        let phase_state_cache = PhaseStateCache::new();
        Self {
            object_count,
            bind_group,
            pipeline,
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

    pub fn compute(&self, compute_pass: &mut ComputePass) {
        let phase_state_bind_group = self.phase_state_cache.get_current();
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);
        phase_state_bind_group.set(compute_pass);
        let (x, y, z) = dispatch_dimensions(self.object_count, WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(x, y, z);
    }
}
