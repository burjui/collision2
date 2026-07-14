use crate::{
    device_buffer::DeviceBuffer,
    shaders::common::{CellPosition, CollisionCandidate, Mass},
};

pub struct BroadPhaseBuffers {
    pub particle_radius: DeviceBuffer<f32>,
    pub grid_min_x: DeviceBuffer<f32>,
    pub grid_max_x: DeviceBuffer<f32>,
    pub grid_min_y: DeviceBuffer<f32>,
    pub grid_max_y: DeviceBuffer<f32>,
    pub grid_size_x: DeviceBuffer<u32>,
    pub grid_size_y: DeviceBuffer<u32>,
    pub object_cells: DeviceBuffer<CellPosition>,
    pub cell_object_count: DeviceBuffer<u32>,
    pub current_cell_offset: DeviceBuffer<u32>,
    pub cell_offsets: DeviceBuffer<u32>,
    pub cells: DeviceBuffer<u32>,
    pub candidates: DeviceBuffer<CollisionCandidate>,
    pub candidate_count: DeviceBuffer<u32>,
    pub masses: DeviceBuffer<Mass>,
    pub forces: DeviceBuffer<u32>,
}
