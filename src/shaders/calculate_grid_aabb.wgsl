#import common::{ AABB, WORKGROUP_SIZE }

var<immediate> thread_offset: u32;

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
    let i = gid.x + thread_offset;
    let object_index = select(0, i, i < object_count);
    let aabb = aabbs[object_index];
    let min_x = subgroupMin(aabb.min.x);
    let min_y = subgroupMin(aabb.min.y);
    let max_x = subgroupMax(aabb.max.x);
    let max_y = subgroupMax(aabb.max.y);
    if sid == 0 {
        atomicGridMinX(min_x);
        atomicGridMinY(min_y);
        atomicGridMaxX(max_x);
        atomicGridMaxY(max_y);
    }
}

fn atomicGridMinX(value: f32) {
    var old = atomicLoad(&grid_min_x);
    loop {
        let old_f32 = bitcast<f32>(old);
        let new_f32 = min(old_f32, value);
        let new_value = bitcast<u32>(new_f32);
        let res = atomicCompareExchangeWeak(&grid_min_x, old, new_value);
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
        let new_value = bitcast<u32>(new_f32);
        let res = atomicCompareExchangeWeak(&grid_min_y, old, new_value);
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
        let new_value = bitcast<u32>(new_f32);
        let res = atomicCompareExchangeWeak(&grid_max_x, old, new_value);
        if res.exchanged {
            break;
        }
        old = res.old_value;
    }
}

fn atomicGridMaxY( value: f32) {
    var old = atomicLoad(&grid_max_y);
    loop {
        let old_f32 = bitcast<f32>(old);
        let new_f32 = max(old_f32, value);
        let new_value = bitcast<u32>(new_f32);
        let res = atomicCompareExchangeWeak(&grid_max_y, old, new_value);
        if res.exchanged {
            break;
        }
        old = res.old_value;
    }
}
