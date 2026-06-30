#import common::Force

const WORKGROUP_SIZE: u32 = 64;

var<immediate> thread_offset: u32;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<storage, read_write> collision_forces: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn reset_collision_forces(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) nwg: vec3u
) {
    let object_index = gid.x + thread_offset;
    if object_index >= object_count {
        return;
    }
    collision_forces[object_index * 2] = 0;
    collision_forces[object_index * 2 + 1] = 0;
}
