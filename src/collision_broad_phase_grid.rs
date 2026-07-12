use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    buffer_sets::BroadPhaseBuffers,
    compute_stage::ComputeStage,
    device_buffer::DeviceBuffer,
    phase_state::{PhaseState, PhaseStateRingConfig},
    shaders::collision_broad_phase_grid::{
        WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams, WgpuBindGroup1, WgpuBindGroup1Entries,
        WgpuBindGroup1EntriesParams, compute::create_broad_phase_grid_pipeline_embed_source,
    },
    util::{PhaseStateCache, dispatch_compute},
};

pub struct CollisionBroadPhaseGrid {
    object_count: u32,
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
    phase_state_cache: PhaseStateCache<WgpuBindGroup1>,
}

impl CollisionBroadPhaseGrid {
    pub fn new(
        device: &Device,
        object_count: u32,
        object_count_buffer: DeviceBuffer<u32>,
        broad_phase_buffers: &BroadPhaseBuffers,
        phase_state_ring_config: PhaseStateRingConfig,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                object_count: object_count_buffer.as_entire_buffer_binding(),
                grid_min_x: broad_phase_buffers.grid_min_x.as_entire_buffer_binding(),
                grid_min_y: broad_phase_buffers.grid_min_y.as_entire_buffer_binding(),
                cell_size: broad_phase_buffers.cell_size.as_entire_buffer_binding(),
                grid_size_x: broad_phase_buffers.grid_size_x.as_entire_buffer_binding(),
                grid_size_y: broad_phase_buffers.grid_size_y.as_entire_buffer_binding(),
                object_cells: broad_phase_buffers.object_cells.as_entire_buffer_binding(),
                cell_object_count: broad_phase_buffers.cell_object_count.as_entire_buffer_binding(),
                cell_offsets: broad_phase_buffers.cell_offsets.as_entire_buffer_binding(),
                cells: broad_phase_buffers.cells.as_entire_buffer_binding(),
                candidates: broad_phase_buffers.candidates.as_entire_buffer_binding(),
                candidate_count: broad_phase_buffers.candidate_count.as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_broad_phase_grid_pipeline_embed_source(device);
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
                    aabbs: phase_state.aabbs().as_entire_buffer_binding(),
                    flags: phase_state.flags().as_entire_buffer_binding(),
                }),
            )
        });
    }
}

impl ComputeStage for CollisionBroadPhaseGrid {
    const LABEL: &'static str = "Collision broad phase";

    fn compute_impl(&self, compute_pass: &mut ComputePass) {
        let phase_state_bind_group = self.phase_state_cache.get_current();
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);
        phase_state_bind_group.set(compute_pass);
        dispatch_compute(compute_pass, self.object_count);
    }
}
