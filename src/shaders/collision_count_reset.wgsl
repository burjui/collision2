#import common::flat_invocation_index

const WORKGROUP_SIZE: u32 = 64;

@group(0) @binding(0) var<uniform> max_candidates: u32;
@group(0) @binding(1) var<storage, read_write> collision_count: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn reset_collision_count(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) nwg: vec3u
) {
    let i = flat_invocation_index(gid, nwg, WORKGROUP_SIZE);
    if i >= max_candidates {
        return;
    }
    collision_count[i] = 0;
}
