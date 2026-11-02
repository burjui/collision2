use crate::{
    device_buffer::DeviceBuffer,
    shaders::common::{CellPosition, Mass},
};

pub struct CollisionBuffers {
    pub object_count: DeviceBuffer<u32>,
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
    pub masses: DeviceBuffer<Mass>,
    pub forces: DeviceBuffer<u32>,
    pub kick_center: DeviceBuffer<[f32; 2]>,
    pub kick_radius: DeviceBuffer<f32>,
    pub kick_magnitude: DeviceBuffer<f32>,
    pub stiffness: DeviceBuffer<f32>,
    pub gamma_coeff: DeviceBuffer<f32>,
    pub sqrt_m_eff: DeviceBuffer<f32>,
}
