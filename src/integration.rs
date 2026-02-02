use wgpu::{ComputePass, ComputePipeline, Device};

use crate::{
    gpu_buffer::GpuBuffer,
    shaders::{
        common::{AABB, BvhNode, Flags, Mass, Velocity},
        integration::{
            WORKGROUP_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams, WgpuBindGroup1,
            WgpuBindGroup1Entries, WgpuBindGroup1EntriesParams, compute::create_cs_main_pipeline_embed_source,
        },
    },
};

pub struct GpuIntegrator {
    pipeline: ComputePipeline,
    bind_group_src: WgpuBindGroup0,
    bind_group_dst: WgpuBindGroup1,
    object_count: usize,
}

impl GpuIntegrator {
    pub fn new(
        device: &Device,
        dt: GpuBuffer<f32>,
        flags: GpuBuffer<Flags>,
        masses: GpuBuffer<Mass>,
        velocities: GpuBuffer<Velocity>,
        aabbs: GpuBuffer<AABB>,
        nodes: GpuBuffer<BvhNode>,
        node_count: GpuBuffer<u32>,
        integrated_flags: GpuBuffer<Flags>,
        integrated_velocities: GpuBuffer<Velocity>,
        integrated_aabbs: GpuBuffer<AABB>,
        errors: GpuBuffer<u32>,
    ) -> Self {
        let pipeline = create_cs_main_pipeline_embed_source(device);
        let bind_group_src = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                dt: dt.buffer().as_entire_buffer_binding(),
                flags: flags.buffer().as_entire_buffer_binding(),
                masses: masses.buffer().as_entire_buffer_binding(),
                velocities: velocities.buffer().as_entire_buffer_binding(),
                aabbs: aabbs.buffer().as_entire_buffer_binding(),
                nodes: nodes.buffer().as_entire_buffer_binding(),
                node_count: node_count.buffer().as_entire_buffer_binding(),
            }),
        );
        let bind_group_dst = WgpuBindGroup1::from_bindings(
            device,
            WgpuBindGroup1Entries::new(WgpuBindGroup1EntriesParams {
                integrated_flags: integrated_flags.buffer().as_entire_buffer_binding(),
                integrated_velocities: integrated_velocities.buffer().as_entire_buffer_binding(),
                integrated_aabbs: integrated_aabbs.buffer().as_entire_buffer_binding(),
                errors: errors.buffer().as_entire_buffer_binding(),
            }),
        );
        Self {
            pipeline,
            bind_group_src,
            bind_group_dst,
            object_count: flags.len(),
        }
    }

    pub fn compute(&self, compute_pass: &mut ComputePass) {
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group_src.set(compute_pass);
        self.bind_group_dst.set(compute_pass);
        let total_workgroups = u32::try_from(self.object_count).unwrap().div_ceil(WORKGROUP_SIZE);
        let x = total_workgroups.min(65535);
        let y = total_workgroups.div_ceil(65535);
        compute_pass.dispatch_workgroups(x, y, 1);
    }
}
