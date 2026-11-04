use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    device_buffer::DeviceBuffer,
    phase_state::{PhaseState, PhaseStateRingConfig},
    shaders::{
        collisions::{
            WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams, WgpuBindGroup1, WgpuBindGroup1Entries,
            WgpuBindGroup1EntriesParams, WgpuBindGroup2, WgpuBindGroup2Entries, WgpuBindGroup2EntriesParams,
            WgpuBindGroup3, WgpuBindGroup3Entries, WgpuBindGroup3EntriesParams,
            compute::create_process_collisions_pipeline_embed_source,
        },
        common::CellPosition,
    },
    util::{ComputeStage, PhaseStateCache, dispatch_compute},
};

pub struct Collisions {
    object_count: u32,
    uniform_bind_group: WgpuBindGroup0,
    main_bind_group: WgpuBindGroup1,
    mutable_bind_group: WgpuBindGroup3,
    pipeline: ComputePipeline,
    phase_state_cache: PhaseStateCache<WgpuBindGroup2>,
}

pub struct CollisionsParams<'a> {
    pub phase_state_ring_config: PhaseStateRingConfig,
    pub object_count: u32,
    pub object_count_buffer: &'a DeviceBuffer<u32>,
    pub particle_radius: &'a DeviceBuffer<f32>,
    pub grid_min_x: &'a DeviceBuffer<f32>,
    pub grid_min_y: &'a DeviceBuffer<f32>,
    pub grid_size_x: &'a DeviceBuffer<u32>,
    pub grid_size_y: &'a DeviceBuffer<u32>,
    pub object_cells: &'a DeviceBuffer<CellPosition>,
    pub cell_object_count: &'a DeviceBuffer<u32>,
    pub cell_offsets: &'a DeviceBuffer<u32>,
    pub cells: &'a DeviceBuffer<u32>,
    pub forces: &'a DeviceBuffer<u32>,
    pub stiffness: &'a DeviceBuffer<f32>,
    pub gamma_coeff: &'a DeviceBuffer<f32>,
    pub sqrt_m_eff: &'a DeviceBuffer<f32>,
}

impl Collisions {
    pub fn new(
        device: &Device,
        CollisionsParams {
            phase_state_ring_config,
            object_count,
            object_count_buffer,
            particle_radius,
            grid_min_x,
            grid_min_y,
            grid_size_x,
            grid_size_y,
            object_cells,
            cell_object_count,
            cell_offsets,
            cells,
            forces,
            stiffness,
            gamma_coeff,
            sqrt_m_eff,
        }: CollisionsParams<'_>,
    ) -> Self {
        let uniform_bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                object_count: object_count_buffer.as_entire_buffer_binding(),
                particle_radius: particle_radius.as_entire_buffer_binding(),
                grid_min_x: grid_min_x.as_entire_buffer_binding(),
                grid_min_y: grid_min_y.as_entire_buffer_binding(),
                grid_size_x: grid_size_x.as_entire_buffer_binding(),
                grid_size_y: grid_size_y.as_entire_buffer_binding(),
                stiffness: stiffness.as_entire_buffer_binding(),
                gamma_coeff: gamma_coeff.as_entire_buffer_binding(),
                sqrt_m_eff: sqrt_m_eff.as_entire_buffer_binding(),
            }),
        );
        let main_bind_group = WgpuBindGroup1::from_bindings(
            device,
            WgpuBindGroup1Entries::new(WgpuBindGroup1EntriesParams {
                object_cells: object_cells.as_entire_buffer_binding(),
                cell_object_count: cell_object_count.as_entire_buffer_binding(),
                cell_offsets: cell_offsets.as_entire_buffer_binding(),
                cells: cells.as_entire_buffer_binding(),
            }),
        );
        let mutable_bind_group = WgpuBindGroup3::from_bindings(
            device,
            WgpuBindGroup3Entries::new(WgpuBindGroup3EntriesParams {
                forces: forces.as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_process_collisions_pipeline_embed_source(device);
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

impl ComputeStage for Collisions {
    const LABEL: &'static str = "Collision broad phase";

    fn compute(&self, compute_pass: &mut ComputePass) {
        let phase_state_bind_group = self.phase_state_cache.get_current();
        compute_pass.set_pipeline(&self.pipeline);
        self.uniform_bind_group.set(compute_pass);
        self.main_bind_group.set(compute_pass);
        self.mutable_bind_group.set(compute_pass);
        phase_state_bind_group.set(compute_pass);
        dispatch_compute(compute_pass, self.object_count);
    }
}
