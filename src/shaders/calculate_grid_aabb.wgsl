#import common::{ AABB, WORKGROUP_SIZE }

var<immediate> thread_offset: u32;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<storage, read_write> grid_min_x: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> grid_max_x: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> grid_min_y: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> grid_max_y: atomic<u32>;

@group(1) @binding(0) var<storage, read_write> aabbs: array<AABB>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn calculate_grid_aabb(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(num_subgroups) num_subgroups: u32
) {
    let i = gid.x + thread_offset;
    let object_index = select(0, i, i < object_count);
    let aabb = aabbs[object_index];
    let min_x = subgroupMin(aabb.min.x);
    let max_x = subgroupMax(aabb.max.x);
    let min_y = subgroupMin(aabb.min.y);
    let max_y = subgroupMax(aabb.max.y);
    if sid == 0 {
        // Assuming positive grid AABB dimensions to be able to bitcast like that
        atomicMin(&grid_min_x, bitcast<u32>(min_x));
        atomicMax(&grid_max_x, bitcast<u32>(max_x));
        atomicMin(&grid_min_y, bitcast<u32>(min_y));
        atomicMax(&grid_max_y, bitcast<u32>(max_y));
    }
}
