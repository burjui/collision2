#import common::{ DispatchIndirectArgs, MAX_DISPATCH_DIMENSION, WORKGROUP_SIZE, div_ceil }

const N_CELL_INDIRECT_DISPATCHES: u32 = 3;

@group(0) @binding(0) var<uniform> particle_radius: f32;
@group(0) @binding(1) var<uniform> grid_min_x: f32;
@group(0) @binding(2) var<uniform> grid_max_x: f32;
@group(0) @binding(3) var<uniform> grid_min_y: f32;
@group(0) @binding(4) var<uniform> grid_max_y: f32;
@group(0) @binding(5) var<storage, read_write> grid_size_x: u32;
@group(0) @binding(6) var<storage, read_write> grid_size_y: u32;
@group(0) @binding(7) var<storage, read_write> cell_offsets_dispatch_dimensions: array<DispatchIndirectArgs, N_CELL_INDIRECT_DISPATCHES>;

@compute @workgroup_size(1)
fn calculate_cell_offsets_dispatch_dimensions() {
    let cell_size = particle_radius * 2.0;
    grid_size_x = u32(floor((grid_max_x - grid_min_x) / cell_size) + 1);
    grid_size_y = u32(floor((grid_max_y - grid_min_y) / cell_size) + 1);
    let cell_count = grid_size_x * grid_size_y;
    let nwg = div_ceil(cell_count, WORKGROUP_SIZE);
    let n_indirect_dispatches = div_ceil(nwg, MAX_DISPATCH_DIMENSION);
    if (n_indirect_dispatches > N_CELL_INDIRECT_DISPATCHES) {
        return;
    }
    for (var i = 0u; i < n_indirect_dispatches; i++) {
        let current_nwg = min(nwg - MAX_DISPATCH_DIMENSION * i, MAX_DISPATCH_DIMENSION);
        cell_offsets_dispatch_dimensions[i] = DispatchIndirectArgs(current_nwg, 1, 1);
    }
}
