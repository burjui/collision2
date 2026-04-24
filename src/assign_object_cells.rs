use wgpu::{BufferUsages, ComputePipeline, Device};

// use crate::shaders::common::CellPosition;
use crate::{
    shaders::assign_object_cells::{
        WgpuBindGroup0, WgpuBindGroup0Entries, WgpuBindGroup0EntriesParams,
        compute::create_assign_object_cells_pipeline_embed_source,
    },
    typed_buffer::TypedBuffer,
};

pub struct AssignObjectCells {
    bind_group: WgpuBindGroup0,
    pipeline: ComputePipeline,
}

impl AssignObjectCells {
    pub fn new(
        device: &Device,
        object_count: usize,
        cell_object_count: TypedBuffer<u32>,
        // object_cells: TypedBuffer<CellPosition>,
    ) -> Self {
        let object_count: u32 = object_count.try_into().unwrap();
        let object_count_buffer =
            TypedBuffer::from_data(device, &[object_count], "object_count", BufferUsages::UNIFORM);
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                object_count: object_count_buffer.buffer().as_entire_buffer_binding(),
                cell_object_count: cell_object_count.buffer().as_entire_buffer_binding(),
            }),
        );
        let pipeline = create_assign_object_cells_pipeline_embed_source(device);
        Self { bind_group, pipeline }
    }
}
