use wgpu::{BufferUsages, ComputePass, ComputePipeline, Device};

use crate::{
    shaders::collision_forces_reset::{
        WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
        compute::create_reset_collision_forces_pipeline_embed_source,
    },
    typed_buffer::TypedBuffer,
    util::dispatch_compute,
};

pub struct CollisionReset {
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
    object_count: u32,
}

impl CollisionReset {
    pub fn new(device: &Device, object_count: u32, collision_forces: TypedBuffer<u32>) -> Self {
        let object_count_buffer =
            TypedBuffer::from_data(device, &[object_count], "object count", BufferUsages::UNIFORM);
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                object_count: object_count_buffer.buffer().as_entire_buffer_binding(),
                collision_forces: collision_forces.buffer().as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_reset_collision_forces_pipeline_embed_source(device);
        Self {
            bind_group,
            pipeline,
            object_count,
        }
    }

    pub fn compute(&self, compute_pass: &mut ComputePass) {
        let pipeline = self.pipeline.clone();
        compute_pass.set_pipeline(&pipeline);
        self.bind_group.set(compute_pass);
        dispatch_compute(compute_pass, self.object_count);
    }
}
