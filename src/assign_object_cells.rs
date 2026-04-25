use wgpu::{ComputePipeline, Device};

// use crate::shaders::common::CellPosition;
use crate::{
    shaders::assign_object_cells::{
        WORKGROUP_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
        compute::create_assign_object_cells_pipeline_embed_source,
    },
    typed_buffer::TypedBuffer,
    util::dispatch_dimensions,
};

pub struct AssignObjectCells {
    object_count: u32,
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
}

impl AssignObjectCells {
    pub fn new(
        device: &Device,
        object_count: usize,
        object_count_buffer: TypedBuffer<u32>,
        cell_object_count: TypedBuffer<u32>,
    ) -> Self {
        let object_count: u32 = object_count.try_into().unwrap();
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                object_count: object_count_buffer.buffer().as_entire_buffer_binding(),
                cell_object_count: cell_object_count.buffer().as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_assign_object_cells_pipeline_embed_source(device);
        Self {
            object_count,
            bind_group,
            pipeline,
        }
    }

    pub fn compute(&self, compute_pass: &mut wgpu::ComputePass) {
        compute_pass.set_pipeline(&self.pipeline);
        self.bind_group.set(compute_pass);
        let (x, y, z) = dispatch_dimensions(self.object_count, WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(x, y, z);
    }
}
