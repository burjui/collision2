#import common::{ MAX_DISPATCH_DIMENSION, GridSize, DispatchIndirectArgs }
#import calculate_cell_offsets::WORKGROUP_SIZE

@group(0) @binding(0) var<uniform> grid_size: GridSize;
@group(0) @binding(1) var<storage, read_write> cell_offsets_dispatch_dimensions: DispatchIndirectArgs;

@compute @workgroup_size(1)
fn calculate_cell_offsets_dispatch_dimensions() {
    let grid_size = grid_size.inner;
    let total_workgroups = (grid_size.x * grid_size.y + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let x = min(total_workgroups, MAX_DISPATCH_DIMENSION);
    let y = min((total_workgroups + x - 1) / x, MAX_DISPATCH_DIMENSION);
    let z = min((total_workgroups + x * y - 1) / (x * y), MAX_DISPATCH_DIMENSION);
    cell_offsets_dispatch_dimensions = DispatchIndirectArgs(x, y, z);
}