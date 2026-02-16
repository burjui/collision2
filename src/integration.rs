use std::sync::{Arc, Mutex};

use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    gpu_buffer::GpuBuffer,
    phase_state::PhaseStateBuffers,
    shaders::{
        common::{BvhNode, Mass},
        integration::{
            WORKGROUP_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams, WgpuBindGroup1,
            WgpuBindGroup1Entries, WgpuBindGroup1EntriesParams, WgpuBindGroup2, WgpuBindGroup2Entries,
            WgpuBindGroup2EntriesParams, compute::create_cs_main_pipeline_embed_source,
        },
    },
};

pub struct GpuIntegrator {
    device: Device,
    phase_states_buffers: Arc<Mutex<PhaseStateBuffers>>,
    masses: GpuBuffer<Mass>,
    nodes: GpuBuffer<BvhNode>,
    bind_group_uniform: WgpuBindGroup0,
    object_count: usize,
    pipeline: ComputePipeline,
}

impl GpuIntegrator {
    pub fn new(
        device: Device,
        phase_states_buffers: Arc<Mutex<PhaseStateBuffers>>,
        dt: GpuBuffer<f32>,
        masses: GpuBuffer<Mass>,
        nodes: GpuBuffer<BvhNode>,
        node_count: GpuBuffer<u32>,
        object_count: usize,
    ) -> Self {
        let pipeline = create_cs_main_pipeline_embed_source(&device);
        let bind_group_uniform = WgpuBindGroup0::from_bindings(
            &device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                dt: dt.buffer().as_entire_buffer_binding(),
                node_count: node_count.buffer().as_entire_buffer_binding(),
            }),
        );

        Self {
            device,
            phase_states_buffers,
            masses,
            nodes,
            bind_group_uniform,
            object_count,
            pipeline,
        }
    }

    pub fn compute(&self, compute_pass: &mut ComputePass) {
        let mut guard = self.phase_states_buffers.lock().unwrap();
        let (src, dst) = guard.next_pair();
        drop(guard);

        let bind_group_src = WgpuBindGroup1::from_bindings(
            &self.device,
            WgpuBindGroup1Entries::new(WgpuBindGroup1EntriesParams {
                flags: src.flags().buffer().as_entire_buffer_binding(),
                masses: self.masses.buffer().as_entire_buffer_binding(),
                velocities: src.velocities().buffer().as_entire_buffer_binding(),
                aabbs: src.aabbs().buffer().as_entire_buffer_binding(),
                nodes: self.nodes.buffer().as_entire_buffer_binding(),
            }),
        );
        let bind_group_dst = WgpuBindGroup2::from_bindings(
            &self.device,
            WgpuBindGroup2Entries::new(WgpuBindGroup2EntriesParams {
                integrated_flags: dst.flags().buffer().as_entire_buffer_binding(),
                integrated_velocities: dst.velocities().buffer().as_entire_buffer_binding(),
                integrated_aabbs: dst.aabbs().buffer().as_entire_buffer_binding(),
            }),
        );

        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group_uniform.set(compute_pass);
        bind_group_src.set(compute_pass);
        bind_group_dst.set(compute_pass);
        let total_workgroups = u32::try_from(self.object_count).unwrap().div_ceil(WORKGROUP_SIZE);
        let x = total_workgroups.min(65535);
        let y = total_workgroups.div_ceil(65535);
        compute_pass.dispatch_workgroups(x, y, 1);
    }
}
