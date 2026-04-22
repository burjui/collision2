#import common::{ Force, flat_invocation_index }

const WORKGROUP_SIZE: u32 = 64;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(2) var<storage, read_write> collision_forces_x: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> collision_forces_y: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn reset_collision_forces(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) nwg: vec3u
) {
    let i = flat_invocation_index(gid, nwg, WORKGROUP_SIZE);
    if i >= object_count {
        return;
    }
    atomicStore(&collision_forces_x[i], 0);
    atomicStore(&collision_forces_y[i], 0);
}
