use wgpu::{BufferUsages, ComputePass, ComputePipeline, Device};

use crate::{
    gpu_buffer::TypedBuffer,
    shaders::collision_forces_reset::{
        WORKGROUP_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
        compute::create_reset_collision_forces_pipeline_embed_source,
    },
    util::dispatch_dimensions,
};

pub struct CollisionReset {
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
    dispatch_dimensions: (u32, u32, u32),
}

impl CollisionReset {
    pub fn new(
        device: &Device,
        object_count: u32,
        collision_forces_x: TypedBuffer<u32>,
        collision_forces_y: TypedBuffer<u32>,
    ) -> Self {
        let dispatch_dimensions = dispatch_dimensions(object_count, WORKGROUP_SIZE);
        let object_count = TypedBuffer::from_data(device, &[object_count], "object count", BufferUsages::UNIFORM);
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                object_count: object_count.buffer().as_entire_buffer_binding(),
                collision_forces_x: collision_forces_x.buffer().as_entire_buffer_binding(),
                collision_forces_y: collision_forces_y.buffer().as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_reset_collision_forces_pipeline_embed_source(device);
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
