use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    buffer_sets::BroadPhaseBuffers,
    compute_stage::ComputeStage,
    device_buffer::DeviceBuffer,
    phase_state::{PhaseState, PhaseStateRingConfig},
    shaders::collision_broad_phase_grid::{
        WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams, WgpuBindGroup1, WgpuBindGroup1Entries,
        WgpuBindGroup1EntriesParams, WgpuBindGroup2, WgpuBindGroup2Entries, WgpuBindGroup2EntriesParams,
        WgpuBindGroup3, WgpuBindGroup3Entries, WgpuBindGroup3EntriesParams,
        compute::create_broad_phase_grid_pipeline_embed_source,
    },
    util::{PhaseStateCache, dispatch_compute},
};

pub struct CollisionBroadPhaseGrid {
    object_count: u32,
    uniform_bind_group: WgpuBindGroup0,
    main_bind_group: WgpuBindGroup1,
    mutable_bind_group: WgpuBindGroup3,
    pipeline: ComputePipeline,
    phase_state_cache: PhaseStateCache<WgpuBindGroup2>,
}

impl CollisionBroadPhaseGrid {
    pub fn new(
        device: &Device,
        object_count: u32,
        object_count_buffer: DeviceBuffer<u32>,
        particle_radius: DeviceBuffer<f32>,
        broad_phase_buffers: &BroadPhaseBuffers,
        phase_state_ring_config: PhaseStateRingConfig,
    ) -> Self {
        let uniform_bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                object_count: object_count_buffer.as_entire_buffer_binding(),
                particle_radius: particle_radius.as_entire_buffer_binding(),
                grid_min_x: broad_phase_buffers.grid_min_x.as_entire_buffer_binding(),
                grid_min_y: broad_phase_buffers.grid_min_y.as_entire_buffer_binding(),
                grid_size_x: broad_phase_buffers.grid_size_x.as_entire_buffer_binding(),
                grid_size_y: broad_phase_buffers.grid_size_y.as_entire_buffer_binding(),
                kick_center: broad_phase_buffers.kick_center.as_entire_buffer_binding(),
                kick_radius: broad_phase_buffers.kick_radius.as_entire_buffer_binding(),
            }),
        );

        let main_bind_group = WgpuBindGroup1::from_bindings(
            device,
            WgpuBindGroup1Entries::new(WgpuBindGroup1EntriesParams {
                object_cells: broad_phase_buffers.object_cells.as_entire_buffer_binding(),
                cell_object_count: broad_phase_buffers.cell_object_count.as_entire_buffer_binding(),
                cell_offsets: broad_phase_buffers.cell_offsets.as_entire_buffer_binding(),
                cells: broad_phase_buffers.cells.as_entire_buffer_binding(),
                masses: broad_phase_buffers.masses.as_entire_buffer_binding(),
            }),
        );

        let mutable_bind_group = WgpuBindGroup3::from_bindings(
            device,
            WgpuBindGroup3Entries::new(WgpuBindGroup3EntriesParams {
                forces: broad_phase_buffers.forces.as_entire_buffer_binding(),
                candidates: broad_phase_buffers.candidates.as_entire_buffer_binding(),
                candidate_count: broad_phase_buffers.candidate_count.as_entire_buffer_binding(),
                kick_magnitude: broad_phase_buffers.kick_magnitude.as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_broad_phase_grid_pipeline_embed_source(device);
        let phase_state_cache = PhaseStateCache::new(phase_state_ring_config);
        Self {
            object_count,
            uniform_bind_group,
            main_bind_group,
            mutable_bind_group,
            pipeline,
            phase_state_cache,
        }
    }

    pub fn prepare(&mut self, device: &Device, phase_state_index: usize, phase_state: &PhaseState) {
        self.phase_state_cache.update(phase_state_index, || {
            WgpuBindGroup2::from_bindings(
                device,
                WgpuBindGroup2Entries::new(WgpuBindGroup2EntriesParams {
                    positions: phase_state.positions().as_entire_buffer_binding(),
                    flags: phase_state.flags().as_entire_buffer_binding(),
                    velocities: phase_state.velocities().as_entire_buffer_binding(),
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
        self.uniform_bind_group.set(compute_pass);
        self.main_bind_group.set(compute_pass);
        self.mutable_bind_group.set(compute_pass);
        phase_state_bind_group.set(compute_pass);
        dispatch_compute(compute_pass, self.object_count);
    }
}
