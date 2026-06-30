#import common::{ DispatchIndirectArgs, MAX_DISPATCH_DIMENSION }
#import collision_narrow_phase::{ WORKGROUP_SIZE, BATCH_SIZE }

@group(0) @binding(0) var<uniform> candidate_count: u32;
@group(0) @binding(1) var<storage, read_write> narrow_phase_dispatch_dimensions: DispatchIndirectArgs;

const CHUNK_SIZE: u32 = WORKGROUP_SIZE * BATCH_SIZE;

// TODO: linearize dispatch
@compute @workgroup_size(1)
fn calculate_narrow_phase_dispatch_dimensions() {
    let total_workgroups = (candidate_count + CHUNK_SIZE - 1) / CHUNK_SIZE;
    let x = min(total_workgroups, MAX_DISPATCH_DIMENSION);
    let y = min((total_workgroups + x - 1) / x, MAX_DISPATCH_DIMENSION);
    let z = min((total_workgroups + x * y - 1) / (x * y), MAX_DISPATCH_DIMENSION);
    narrow_phase_dispatch_dimensions = DispatchIndirectArgs(x, y, z);
}
