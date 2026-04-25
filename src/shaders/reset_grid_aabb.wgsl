
#import common::{ AABB, flat_invocation_index }

const WORKGROUP_SIZE: u32 = 64;

@group(0) @binding(0) var<uniform> first_aabb: AABB;
@group(0) @binding(1) var<storage, read_write> grid_min_x: f32;
@group(0) @binding(2) var<storage, read_write> grid_min_y: f32;
@group(0) @binding(3) var<storage, read_write> grid_max_x: f32;
@group(0) @binding(4) var<storage, read_write> grid_max_y: f32;

@compute @workgroup_size(WORKGROUP_SIZE)
fn reset_grid_aabb() {
    grid_min_x = first_aabb.min.x;
    grid_min_y = first_aabb.min.y;
    grid_max_x = first_aabb.max.x;
    grid_max_y = first_aabb.max.y;
}
