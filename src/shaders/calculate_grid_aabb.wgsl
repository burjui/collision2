#import common::{ AABB, GridPosition, flat_invocation_index };

const WORKGROUP_SIZE: u32 = 64;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<storage, read_write> grid_min_x: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> grid_min_y: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> grid_max_x: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> grid_max_y: atomic<u32>;

@group(1) @binding(0) var<storage, read_write> aabbs: array<AABB>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn calculate_grid_aabb(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(subgroup_invocation_id) sid: u32
) {
    let i = flat_invocation_index(gid, nwg, WORKGROUP_SIZE);
    if i >= object_count {
        return;
    }
    let aabb = aabbs[i];
    if sid == 0 {
        atomicGridMinX(subgroupMin(aabb.min.x));
        atomicGridMinY(subgroupMin(aabb.min.y));
        atomicGridMaxX(subgroupMax(aabb.max.x));
        atomicGridMaxY(subgroupMax(aabb.max.y));
    }
}

fn atomicGridMinX(value: f32) {
    var old = atomicLoad(&grid_min_x);
    loop {
        let old_f32 = bitcast<f32>(old);
        let new_f32 = min(old_f32, value);
        let new_u32 = bitcast<u32>(new_f32);
        let res = atomicCompareExchangeWeak(&grid_min_x, old, new_u32);
        if res.exchanged {
            break;
        }
        old = res.old_value;
    }
}

fn atomicGridMinY(value: f32) {
    var old = atomicLoad(&grid_min_y);
    loop {
        let old_f32 = bitcast<f32>(old);
        let new_f32 = min(old_f32, value);
        let new_u32 = bitcast<u32>(new_f32);
        let res = atomicCompareExchangeWeak(&grid_min_y, old, new_u32);
        if res.exchanged {
            break;
        }
        old = res.old_value;
    }
}

fn atomicGridMaxX(value: f32) {
    var old = atomicLoad(&grid_max_x);
    loop {
        let old_f32 = bitcast<f32>(old);
        let new_f32 = max(old_f32, value);
        let new_u32 = bitcast<u32>(new_f32);
        let res = atomicCompareExchangeWeak(&grid_max_x, old, new_u32);
        if res.exchanged {
            break;
        }
        old = res.old_value;
    }
}

fn atomicGridMaxY(value: f32) {
    var old = atomicLoad(&grid_max_y);
    loop {
        let old_f32 = bitcast<f32>(old);
        let new_f32 = max(old_f32, value);
        let new_u32 = bitcast<u32>(new_f32);
        let res = atomicCompareExchangeWeak(&grid_max_y, old, new_u32);
        if res.exchanged {
            break;
        }
        old = res.old_value;
    }
}
