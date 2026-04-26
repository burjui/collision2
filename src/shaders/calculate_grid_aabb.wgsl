#import common::{ AABB, flat_invocation_index, f32_to_i32, i32_to_f32, INFINITY };

const WORKGROUP_SIZE: u32 = 64;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<storage, read_write> grid_min_x: atomic<i32>;
@group(0) @binding(2) var<storage, read_write> grid_min_y: atomic<i32>;
@group(0) @binding(3) var<storage, read_write> grid_max_x: atomic<i32>;
@group(0) @binding(4) var<storage, read_write> grid_max_y: atomic<i32>;

@group(1) @binding(0) var<storage, read_write> aabbs: array<AABB>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn calculate_grid_aabb(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
) {
    let original_i = flat_invocation_index(gid, nwg, WORKGROUP_SIZE);
    let i_is_valid = original_i < object_count;
    let i = select(0, original_i, i_is_valid);
    let aabb = aabbs[i];
    let masked_min_x = select(INFINITY, aabb.min.x, i_is_valid);
    let masked_min_y = select(INFINITY, aabb.min.y, i_is_valid);
    let masked_max_x = select(-INFINITY, aabb.max.x, i_is_valid);
    let masked_max_y = select(-INFINITY, aabb.max.y, i_is_valid);
    atomicMin(&grid_min_x, f32_to_i32(masked_min_x));
    atomicMin(&grid_min_y, f32_to_i32(masked_min_y));
    atomicMax(&grid_max_x, f32_to_i32(masked_max_x));
    atomicMax(&grid_max_y, f32_to_i32(masked_max_y));
}
