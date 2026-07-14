#import common::{ Position, CellPosition, WORKGROUP_SIZE }

var<immediate> thread_offset: u32;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<uniform> particle_radius: f32;
@group(0) @binding(2) var<uniform> grid_min_x: f32;
@group(0) @binding(3) var<uniform> grid_min_y: f32;
@group(0) @binding(4) var<uniform> grid_size_x: u32;
@group(0) @binding(5) var<uniform> grid_size_y: u32;
@group(0) @binding(6) var<storage, read_write> cell_object_count: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> object_cells: array<CellPosition>;

@group(1) @binding(0) var<storage, read> positions: array<Position>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn assign_object_cells(@builtin(global_invocation_id) gid: vec3u) {
    let object_index = gid.x + thread_offset;
    if object_index >= object_count {
        return;
    }

    let position = positions[object_index].inner;
    let cell_size = particle_radius * 2;
    let cell = clamp(
        vec2u((position - vec2f(grid_min_x, grid_min_y)) / cell_size),
        vec2u(0),
        vec2u(grid_size_x - 1u, grid_size_y - 1u),
    );
    let cell_index = cell.x + cell.y * grid_size_x;
    let offset = atomicAdd(&cell_object_count[cell_index], 1);
    object_cells[object_index] = CellPosition(cell, offset);
}
