#import common::{ GridSize, DispatchIndirectArgs }

@group(0) @binding(0) var<uniform> grid_size: GridSize;
@group(0) @binding(1) var<storage, read_write> cell_offsets_dispatch_dimensions: DispatchIndirectArgs;

@compute @workgroup_size(1)
fn calculate_cell_iteration_dispatch_dimensions() {
    cell_offsets_dispatch_dimensions = DispatchIndirectArgs(grid_size.x, grid_size.y, 1);
}
