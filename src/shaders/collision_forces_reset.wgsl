#import common::{ Force, flat_invocation_index }

const WORKGROUP_SIZE: u32 = 64;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<storage, read_write> collision_forces: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn reset_collision_forces(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) nwg: vec3u
) {
    let object_index = flat_invocation_index(gid, nwg, WORKGROUP_SIZE);
    if object_index >= object_count {
        return;
    }
    collision_forces[object_index * 2] = 0;
    collision_forces[object_index * 2 + 1] = 0;
}
