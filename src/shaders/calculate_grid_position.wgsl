#import common::{ AABB, GridPosition, flat_invocation_index };

const WORKGROUP_SIZE: u32 = 64;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<storage, read_write> grid_position_x: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> grid_position_y: atomic<u32>;
@group(1) @binding(0) var<storage, read_write> aabbs: array<AABB>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn calculate_grid_position(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
) {
    let i = flat_invocation_index(gid, nwg, WORKGROUP_SIZE);
    if i >= object_count {
        return;
    }
    let aabb = aabbs[i];
    atomicMinF32x(aabb.min.x);
    atomicMinF32y(aabb.min.y);
}

fn atomicMinF32x(value: f32) {
    var old = atomicLoad(&grid_position_x);
    loop {
        let old_f32 = bitcast<f32>(old);
        let new_f32 = min(old_f32, value);
        let new_u32 = bitcast<u32>(new_f32);
        let res = atomicCompareExchangeWeak(&grid_position_x, old, new_u32);
        if res.exchanged {
            break;
        }
        old = res.old_value;
    }
}

fn atomicMinF32y(value: f32) {
    var old = atomicLoad(&grid_position_y);
    loop {
        let old_f32 = bitcast<f32>(old);
        let new_f32 = min(old_f32, value);
        let new_u32 = bitcast<u32>(new_f32);
        let res = atomicCompareExchangeWeak(&grid_position_y, old, new_u32);
        if res.exchanged {
            break;
        }
        old = res.old_value;
    }
}