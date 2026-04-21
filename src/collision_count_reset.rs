use wgpu::{BufferUsages, ComputePass, ComputePipeline, Device};

use crate::{
    gpu_buffer::TypedBuffer,
    shaders::collision_count_reset::{
        WORKGROUP_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
        compute::create_reset_collision_count_pipeline_embed_source,
    },
    util::dispatch_dimensions,
};

pub struct CollisionCountReset {
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
    dispatch_dimensions: (u32, u32, u32),
}

impl CollisionCountReset {
    pub fn new(device: &Device, max_candidates: u32, collision_count: TypedBuffer<u32>) -> Self {
        let dispatch_dimensions = dispatch_dimensions(max_candidates, WORKGROUP_SIZE);
        let max_candidates = TypedBuffer::from_data(device, &[max_candidates], "max_candidates", BufferUsages::UNIFORM);
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                max_candidates: max_candidates.buffer().as_entire_buffer_binding(),
                collision_count: collision_count.buffer().as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_reset_collision_count_pipeline_embed_source(device);
        Self {
            bind_group,
            pipeline,
            dispatch_dimensions,
        }
    }

    pub fn compute(&self, compute_pass: &mut ComputePass) {
        let pipeline = self.pipeline.clone();
        compute_pass.set_pipeline(&pipeline);
        self.bind_group.set(compute_pass);
        let (x, y, z) = self.dispatch_dimensions;
        compute_pass.dispatch_workgroups(x, y, z);
    }
}
