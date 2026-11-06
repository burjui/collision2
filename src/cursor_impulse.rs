use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    device_buffer::DeviceBuffer,
    phase_state::{PhaseState, PhaseStateRingConfig},
    shaders::cursor_impulse::{
        WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams, WgpuBindGroup1, WgpuBindGroup1Entries,
        WgpuBindGroup1EntriesParams, compute::create_apply_cursor_impulse_pipeline_embed_source,
    },
    util::{ComputeStage, PhaseStateCache, dispatch_compute},
};

pub struct CursorImpulse {
    object_count: u32,
    uniform_bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
    phase_state_cache: PhaseStateCache<WgpuBindGroup1>,
}

pub struct CursorImpulseParams<'a> {
    pub phase_state_ring_config: PhaseStateRingConfig,
    pub object_count: u32,
    pub object_count_buffer: &'a DeviceBuffer<u32>,
    pub cursor_position: &'a DeviceBuffer<[f32; 2]>,
    pub cursor_impulse_radius: &'a DeviceBuffer<f32>,
    pub cursor_impulse_speed: &'a DeviceBuffer<f32>,
}

impl CursorImpulse {
    pub fn new(
        device: &Device,
        CursorImpulseParams {
            phase_state_ring_config,
            object_count,
            object_count_buffer,
            cursor_position,
            cursor_impulse_radius,
            cursor_impulse_speed,
        }: CursorImpulseParams,
    ) -> Self {
        let uniform_bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                object_count: object_count_buffer.as_entire_buffer_binding(),
                cursor_position: cursor_position.as_entire_buffer_binding(),
                cursor_impulse_radius: cursor_impulse_radius.as_entire_buffer_binding(),
                cursor_impulse_speed: cursor_impulse_speed.as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_apply_cursor_impulse_pipeline_embed_source(device);
        let phase_state_cache = PhaseStateCache::new(phase_state_ring_config);
        Self {
            object_count,
            uniform_bind_group,
            pipeline,
            phase_state_cache,
        }
    }

    pub fn prepare(&mut self, device: &Device, phase_state_index: usize, phase_state: &PhaseState) {
        self.phase_state_cache.update(phase_state_index, || {
            WgpuBindGroup1::from_bindings(
                device,
                WgpuBindGroup1Entries::new(WgpuBindGroup1EntriesParams {
                    flags: phase_state.flags().as_entire_buffer_binding(),
                    positions: phase_state.positions().as_entire_buffer_binding(),
                    velocities: phase_state.velocities().as_entire_buffer_binding(),
                }),
            )
        });
    }
}

impl ComputeStage for CursorImpulse {
    const LABEL: &'static str = "Collision broad phase";

    fn compute(&self, compute_pass: &mut ComputePass) {
        let phase_state_bind_group = self.phase_state_cache.get_current();
        compute_pass.set_pipeline(&self.pipeline);
        self.uniform_bind_group.set(compute_pass);
        phase_state_bind_group.set(compute_pass);
        dispatch_compute(compute_pass, self.object_count);
    }
}
