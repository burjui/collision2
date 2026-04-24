#import common::{ Position, flat_invocation_index };

const WORKGROUP_SIZE: u32 = 64;

@group(0) @binding(0) var<storage, read_write> grid_position: Position;

@compute @workgroup_size(WORKGROUP_SIZE)
fn calculate_grid_position(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
) {

}