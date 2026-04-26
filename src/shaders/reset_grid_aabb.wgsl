
#import common::{ AABB, flat_invocation_index, f32_to_i32 }

@group(0) @binding(0) var<uniform> first_aabb: AABB;
@group(0) @binding(1) var<storage, read_write> grid_min_x: i32;
@group(0) @binding(2) var<storage, read_write> grid_min_y: i32;
@group(0) @binding(3) var<storage, read_write> grid_max_x: i32;
@group(0) @binding(4) var<storage, read_write> grid_max_y: i32;

@compute @workgroup_size(1)
fn reset_grid_aabb() {
    grid_min_x = f32_to_i32(first_aabb.min.x);
    grid_min_y = f32_to_i32(first_aabb.min.y);
    grid_max_x = f32_to_i32(first_aabb.max.x);
    grid_max_y = f32_to_i32(first_aabb.max.y);
}
