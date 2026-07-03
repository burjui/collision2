#import common::DispatchIndirectArgs

@group(0) @binding(0) var<uniform> grid_min_x: f32;
@group(0) @binding(1) var<uniform> grid_max_x: f32;
@group(0) @binding(2) var<uniform> grid_min_y: f32;
@group(0) @binding(3) var<uniform> grid_max_y: f32;
@group(0) @binding(6) var<uniform> cell_size: f32;
@group(0) @binding(4) var<storage, read_write> grid_size_x: u32;
@group(0) @binding(5) var<storage, read_write> grid_size_y: u32;
@group(0) @binding(7) var<storage, read_write> cell_offsets_dispatch_dimensions: DispatchIndirectArgs;

@compute @workgroup_size(1)
fn calculate_cell_iteration_dispatch_dimensions() {
    grid_size_x = u32(ceil((grid_max_x - grid_min_x) / cell_size));
    grid_size_y = u32(ceil((grid_max_y - grid_min_y) / cell_size));
    cell_offsets_dispatch_dimensions = DispatchIndirectArgs(grid_size_x, grid_size_y, 1);
}
