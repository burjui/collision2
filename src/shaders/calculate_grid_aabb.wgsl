#import common::{ INFINITY, Position, WORKGROUP_SIZE }

var<immediate> thread_offset: u32;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<uniform> particle_radius: f32;
@group(0) @binding(2) var<storage, read_write> grid_min_x: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> grid_max_x: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> grid_min_y: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> grid_max_y: atomic<u32>;

@group(1) @binding(0) var<storage, read> positions: array<Position>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn calculate_grid_aabb(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(subgroup_invocation_id) sid: u32,
) {
    let i = gid.x + thread_offset;
    let is_valid = (i < object_count);
    let object_index = select(0, i, is_valid);
    let position = positions[object_index].inner;
    let object_min = position - particle_radius;
    let object_max = position + particle_radius;
    let min_x = subgroupMin(object_min.x);
    let max_x = subgroupMax(object_max.x);
    let min_y = subgroupMin(object_min.y);
    let max_y = subgroupMax(object_max.y);
    if sid == 0 {
        // Assuming positive grid AABB dimensions to be able to bitcast like that
        atomicMin(&grid_min_x, bitcast<u32>(min_x));
        atomicMax(&grid_max_x, bitcast<u32>(max_x));
        atomicMin(&grid_min_y, bitcast<u32>(min_y));
        atomicMax(&grid_max_y, bitcast<u32>(max_y));
    }
}
