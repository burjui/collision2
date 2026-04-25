#import common::{ MAX_OBJECTS_PER_CELL, GridSize, CellPosition, flat_invocation_index }

const WORKGROUP_SIZE: u32 = 64;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<uniform> grid_size: GridSize;
@group(0) @binding(2) var<storage, read> object_cells: array<CellPosition>;
@group(0) @binding(3) var<storage, read_write> cell_object_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cells: array<array<u32, MAX_OBJECTS_PER_CELL>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn populate_object_cells(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) nwg: vec3u
) {
    let i = flat_invocation_index(gid, nwg, WORKGROUP_SIZE);
    if i >= object_count {
        return;
    }
    let cell_position = object_cells[i].inner;
    let grid_size = grid_size.inner;
    let cell_offset = cell_position.x + cell_position.y * grid_size.x;
    let object_offset = atomicAdd(&cell_object_count[0], 1);
    cells[cell_offset][object_offset] = i;
}