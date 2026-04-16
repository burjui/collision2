use std::array::from_fn;

use wgpu::{BufferUsages, ComputePass, ComputePipeline, Device};

use crate::{
    gpu_buffer::TypedBuffer,
    phase_state::{PhaseState, PhaseStateRing},
    shaders::{
        common::{BvhNode, Mass},
        integrate::{
            BlackHole, WORKGROUP_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
            WgpuBindGroup1, WgpuBindGroup1Entries, WgpuBindGroup1EntriesParams, WgpuBindGroup2, WgpuBindGroup2Entries,
            WgpuBindGroup2EntriesParams, compute::create_integrate_pipeline_embed_source,
        },
    },
    util::dispatch_dimensions,
};

pub struct GpuIntegrator {
    object_count: u32,
    main_bind_group: WgpuBindGroup0,
    blackhole_bind_group: WgpuBindGroup1,
    pipeline: ComputePipeline,
    phase_state_bind_groups: [Option<WgpuBindGroup2>; PhaseStateRing::CAPACITY],
    phase_state_index: Option<usize>,
}

impl GpuIntegrator {
    const BLACKHOLE_DUMMY: BlackHole = BlackHole::new([0.0, 0.0], 0.0, 0.0, 0.0);
    const BLACKHOLES: &[BlackHole] = &[
        // BlackHole::new([-200.0, 500.0], 2.0, 10.0, 0.0 * -50.0),
        // BlackHole::new([500.0, 200.0], 1.0, 10.0, 0.0 * -50.0),
        // BlackHole::new([0.0, 0.0], 7.0, 20.0, 50.0 * 100.0),
        // BlackHole::new([-600.0, -300.0], 1.0, 20.0, 0.0 * -50.0),
        // BlackHole::new([600.0, -700.0], 1.0, 10.0, 0.0 * -50.0),
        //-------------
        Self::BLACKHOLE_DUMMY,
    ];
    const BLACKHOLE_MASS_SCALE: f32 = 1.0 * 1000.0;
    const BLACKHOLE_SIZE_SCALE: f32 = 10.0;
    const GRAVITATIONAL_CONSTANT: f32 = 1.0 * 100000.0;
    const GLOBAL_FORCE: [f32; 2] = [0.0, -1000.0];

    pub fn new(
        device: &Device,
        object_count: usize,
        dt: TypedBuffer<f32>,
        masses: TypedBuffer<Mass>,
        nodes: TypedBuffer<BvhNode>,
    ) -> Self {
        let blackholes = TypedBuffer::from_data(device, Self::BLACKHOLES, "blackholes", BufferUsages::STORAGE);
        let pipeline = create_integrate_pipeline_embed_source(device);
        let blackhole_count = u32::try_from(Self::BLACKHOLES.len() - 1).unwrap();
        let blackhole_count =
            TypedBuffer::from_data(device, &[blackhole_count], "blackhole count", BufferUsages::UNIFORM);
        let blackhole_mass_scale = TypedBuffer::from_data(
            device,
            &[Self::BLACKHOLE_MASS_SCALE],
            "blackhole mass scale",
            BufferUsages::UNIFORM,
        );
        let blackhole_size_scale = TypedBuffer::from_data(
            device,
            &[Self::BLACKHOLE_SIZE_SCALE],
            "blackhole size scale",
            BufferUsages::UNIFORM,
        );
        let gravitational_constant = TypedBuffer::from_data(
            device,
            &[Self::GRAVITATIONAL_CONSTANT],
            "gravitational constant",
            BufferUsages::UNIFORM,
        );
        let global_force = TypedBuffer::from_data(device, &[Self::GLOBAL_FORCE], "global force", BufferUsages::UNIFORM);
        let main_bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                dt: dt.buffer().as_entire_buffer_binding(),
                gravitational_constant: gravitational_constant.buffer().as_entire_buffer_binding(),
                global_force: global_force.buffer().as_entire_buffer_binding(),
                masses: masses.buffer().as_entire_buffer_binding(),
                nodes: nodes.buffer().as_entire_buffer_binding(),
            }),
        );
        let blackhole_bind_group = WgpuBindGroup1::from_bindings(
            device,
            WgpuBindGroup1Entries::new(WgpuBindGroup1EntriesParams {
                blackhole_count: blackhole_count.buffer().as_entire_buffer_binding(),
                blackhole_mass_scale: blackhole_mass_scale.buffer().as_entire_buffer_binding(),
                blackhole_size_scale: blackhole_size_scale.buffer().as_entire_buffer_binding(),
                blackholes: blackholes.buffer().as_entire_buffer_binding(),
            }),
        );

        Self {
            object_count: object_count.try_into().unwrap(),
            main_bind_group,
            blackhole_bind_group,
            pipeline,
            phase_state_bind_groups: from_fn(|_| None),
            phase_state_index: None,
        }
    }

    pub fn prepare(&mut self, phase_state_index: usize, device: &Device, src: &PhaseState, dst: &PhaseState) {
        self.phase_state_bind_groups[phase_state_index].get_or_insert_with(|| {
            WgpuBindGroup2::from_bindings(
                device,
                WgpuBindGroup2Entries::new(WgpuBindGroup2EntriesParams {
                    flags: src.flags().buffer().as_entire_buffer_binding(),
                    aabbs: src.aabbs().buffer().as_entire_buffer_binding(),
                    velocities: src.velocities().buffer().as_entire_buffer_binding(),
                    integrated_flags: dst.flags().buffer().as_entire_buffer_binding(),
                    integrated_velocities: dst.velocities().buffer().as_entire_buffer_binding(),
                    integrated_aabbs: dst.aabbs().buffer().as_entire_buffer_binding(),
                }),
            )
        });
        self.phase_state_index = Some(phase_state_index);
    }

    pub fn compute(&self, compute_pass: &mut ComputePass) {
        let phase_state_index = self.phase_state_index.expect("prepare() must be called every frame");
        let phase_state_bind_group = self.phase_state_bind_groups[phase_state_index].as_ref().unwrap();
        compute_pass.set_pipeline(&self.pipeline);
        self.main_bind_group.set(compute_pass);
        self.blackhole_bind_group.set(compute_pass);
        phase_state_bind_group.set(compute_pass);
        let (x, y, z) = dispatch_dimensions(self.object_count, WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(x, y, z);
    }
}
