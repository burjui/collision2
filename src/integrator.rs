use nalgebra::Vector2;
use wgpu::{BufferUsages, ComputePass, ComputePipeline, Device};

use crate::{
    compute_stage::ComputeStage,
    config::CONFIG,
    device_buffer::DeviceBuffer,
    phase_state::{PhaseState, PhaseStateRingConfig},
    shaders::{
        common::{AABB, Mass},
        integrate::{
            BlackHole, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams, WgpuBindGroup1,
            WgpuBindGroup1Entries, WgpuBindGroup1EntriesParams, WgpuBindGroup2, WgpuBindGroup2Entries,
            WgpuBindGroup2EntriesParams, WgpuBindGroup3, WgpuBindGroup3Entries, WgpuBindGroup3EntriesParams,
            compute::create_integrate_pipeline_embed_source,
        },
    },
    util::{PhaseStateCache, dispatch_compute},
};

pub struct Integrator {
    object_count: u32,
    main_bind_group: WgpuBindGroup0,
    blackhole_bind_group: WgpuBindGroup1,
    collision_bind_group: WgpuBindGroup2,
    pipeline: ComputePipeline,
    phase_state_cache: PhaseStateCache<WgpuBindGroup3>,
}

impl Integrator {
    const BLACKHOLE_DUMMY: BlackHole = BlackHole::new([0.0, 0.0], 0.0, 0.0, 0.0);
    const BLACKHOLE_MASS_SCALE: f32 = 1.0 * 1000.0;
    const BLACKHOLE_SIZE_SCALE: f32 = 10.0;
    const GRAVITATIONAL_CONSTANT: f32 = 1.0 * 100000.0;

    pub fn new(
        device: &Device,
        object_count: u32,
        object_count_buffer: DeviceBuffer<u32>,
        constraints: DeviceBuffer<AABB>,
        dt: DeviceBuffer<f32>,
        masses: DeviceBuffer<Mass>,
        collision_forces: DeviceBuffer<u32>,
        phase_state_ring_config: PhaseStateRingConfig,
    ) -> Self {
        let mut blackholes = Vec::new();
        if CONFIG.blackhole {
            let position = Vector2::from(CONFIG.world_size()) * 0.5;
            blackholes.push(BlackHole::new(
                position.into(),
                CONFIG.blackhole_radius,
                CONFIG.blackhole_mass,
                CONFIG.blackhole_spin,
            ));
        }
        blackholes.push(Self::BLACKHOLE_DUMMY); // because empty buffers are not allowed

        let blackholes_buffer = DeviceBuffer::from_data(device, &blackholes, "blackholes", BufferUsages::STORAGE);
        let pipeline = create_integrate_pipeline_embed_source(device);
        let blackhole_count = u32::try_from(blackholes.len() - 1).unwrap();
        let blackhole_count =
            DeviceBuffer::from_data(device, &[blackhole_count], "blackhole count", BufferUsages::UNIFORM);
        let blackhole_mass_scale = DeviceBuffer::from_data(
            device,
            &[Self::BLACKHOLE_MASS_SCALE],
            "blackhole mass scale",
            BufferUsages::UNIFORM,
        );
        let blackhole_size_scale = DeviceBuffer::from_data(
            device,
            &[Self::BLACKHOLE_SIZE_SCALE],
            "blackhole size scale",
            BufferUsages::UNIFORM,
        );
        let gravitational_constant = DeviceBuffer::from_data(
            device,
            &[Self::GRAVITATIONAL_CONSTANT],
            "gravitational constant",
            BufferUsages::UNIFORM,
        );
        let global_acceleration: [f32; 2] = CONFIG.accel();
        let global_acceleration_buffer =
            DeviceBuffer::from_data(device, &[global_acceleration], "global force", BufferUsages::UNIFORM);

        let main_bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                dt: dt.as_entire_buffer_binding(),
                gravitational_constant: gravitational_constant.as_entire_buffer_binding(),
                global_acceleration: global_acceleration_buffer.as_entire_buffer_binding(),
                object_count: object_count_buffer.as_entire_buffer_binding(),
                constraints: constraints.as_entire_buffer_binding(),
                masses: masses.as_entire_buffer_binding(),
            }),
        );
        let blackhole_bind_group = WgpuBindGroup1::from_bindings(
            device,
            WgpuBindGroup1Entries::new(WgpuBindGroup1EntriesParams {
                blackhole_count: blackhole_count.as_entire_buffer_binding(),
                blackhole_mass_scale: blackhole_mass_scale.as_entire_buffer_binding(),
                blackhole_size_scale: blackhole_size_scale.as_entire_buffer_binding(),
                blackholes: blackholes_buffer.as_entire_buffer_binding(),
            }),
        );
        let collision_bind_group = WgpuBindGroup2::from_bindings(
            device,
            WgpuBindGroup2Entries::new(WgpuBindGroup2EntriesParams {
                collision_forces: collision_forces.as_entire_buffer_binding(),
            }),
        );
        let phase_state_cache = PhaseStateCache::new(phase_state_ring_config);
        Self {
            object_count,
            main_bind_group,
            blackhole_bind_group,
            collision_bind_group,
            pipeline,
            phase_state_cache,
        }
    }

    pub fn prepare(&mut self, device: &Device, phase_state_index: usize, src: &PhaseState, dst: &PhaseState) {
        self.phase_state_cache.update(phase_state_index, || {
            WgpuBindGroup3::from_bindings(
                device,
                WgpuBindGroup3Entries::new(WgpuBindGroup3EntriesParams {
                    flags: src.flags().as_entire_buffer_binding(),
                    aabbs: src.aabbs().as_entire_buffer_binding(),
                    velocities: src.velocities().as_entire_buffer_binding(),
                    integrated_flags: dst.flags().as_entire_buffer_binding(),
                    integrated_velocities: dst.velocities().as_entire_buffer_binding(),
                    integrated_aabbs: dst.aabbs().as_entire_buffer_binding(),
                }),
            )
        });
    }
}

impl ComputeStage for Integrator {
    const LABEL: &'static str = "Integrate";

    fn compute_impl(&self, compute_pass: &mut ComputePass) {
        let phase_state_bind_group = self.phase_state_cache.get_current();
        compute_pass.set_pipeline(&self.pipeline);
        self.main_bind_group.set(compute_pass);
        self.blackhole_bind_group.set(compute_pass);
        self.collision_bind_group.set(compute_pass);
        phase_state_bind_group.set(compute_pass);
        dispatch_compute(compute_pass, self.object_count);
    }
}
