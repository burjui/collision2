#import common::{ AABB, GridSize, flat_invocation_index }

const WORKGROUP_SIZE: u32 = 64;

@group(0) @binding(1) var<uniform> grid_min_x: f32;
@group(0) @binding(2) var<uniform> grid_min_y: f32;
@group(0) @binding(3) var<uniform> grid_max_x: f32;
@group(0) @binding(4) var<uniform> grid_max_y: f32;
@group(0) @binding(5) var<uniform> cell_size: f32;
@group(0) @binding(6) var<storage, read_write> grid_size: GridSize;

@compute @workgroup_size(WORKGROUP_SIZE)
fn calculate_grid_size() {
    let size_x = grid_max_x - grid_min_x;
    let size_y = grid_max_y - grid_min_y;
    grid_size.inner = vec2u(u32(size_x / cell_size), u32(size_y / cell_size));
}
