use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    device_buffer::DeviceBuffer,
    phase_state::{PhaseState, PhaseStateRingConfig},
    shaders::calculate_grid_aabb::{
        WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams, WgpuBindGroup1, WgpuBindGroup1Entries,
        WgpuBindGroup1EntriesParams, compute::create_calculate_grid_aabb_pipeline_embed_source,
    },
    util::{ComputeStage, PhaseStateCache, dispatch_compute},
};

pub struct CalculateGridAABB {
    object_count: u32,
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
    phase_state_cache: PhaseStateCache<WgpuBindGroup1>,
}

pub struct CalculateGridAABBParams<'a> {
    pub phase_state_ring_config: PhaseStateRingConfig,
    pub object_count: u32,
    pub object_count_buffer: &'a DeviceBuffer<u32>,
    pub particle_radius: &'a DeviceBuffer<f32>,
    pub grid_min_x: &'a DeviceBuffer<f32>,
    pub grid_max_x: &'a DeviceBuffer<f32>,
    pub grid_min_y: &'a DeviceBuffer<f32>,
    pub grid_max_y: &'a DeviceBuffer<f32>,
}

impl CalculateGridAABB {
    pub fn new(
        device: &Device,
        CalculateGridAABBParams {
            object_count,
            phase_state_ring_config,
            object_count_buffer,
            particle_radius,
            grid_min_x,
            grid_max_x,
            grid_min_y,
            grid_max_y,
        }: CalculateGridAABBParams<'_>,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                object_count: object_count_buffer.as_entire_buffer_binding(),
                particle_radius: particle_radius.as_entire_buffer_binding(),
                grid_min_x: grid_min_x.as_entire_buffer_binding(),
                grid_max_x: grid_max_x.as_entire_buffer_binding(),
                grid_min_y: grid_min_y.as_entire_buffer_binding(),
                grid_max_y: grid_max_y.as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_calculate_grid_aabb_pipeline_embed_source(device);
        let phase_state_cache = PhaseStateCache::new(phase_state_ring_config);
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
                    positions: phase_state.positions().as_entire_buffer_binding(),
                }),
            )
        });
    }
}

impl ComputeStage for CalculateGridAABB {
    const LABEL: &'static str = "Calculate grid AABB";

    fn compute_impl(&self, compute_pass: &mut ComputePass) {
        let phase_state_bind_group = self.phase_state_cache.get_current();
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);
        phase_state_bind_group.set(compute_pass);
        dispatch_compute(compute_pass, self.object_count);
    }
}
