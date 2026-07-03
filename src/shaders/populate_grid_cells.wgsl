#import common::CellPosition

const WORKGROUP_SIZE: u32 = 64;

var<immediate> thread_offset: u32;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<uniform> grid_min_x: f32;
@group(0) @binding(2) var<uniform> grid_max_x: f32;
@group(0) @binding(3) var<uniform> grid_min_y: f32;
@group(0) @binding(4) var<uniform> grid_max_y: f32;
@group(0) @binding(5) var<uniform> cell_size: f32;
@group(0) @binding(6) var<storage, read> object_cells: array<CellPosition>;
@group(0) @binding(7) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(8) var<storage, read_write> cells: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn populate_object_cells(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) nwg: vec3u
) {
    let object_index = gid.x + thread_offset;
    if object_index >= object_count {
        return;
    }
    let cell_position = object_cells[object_index];
    let grid_size_x = u32(ceil((grid_max_x - grid_min_x) / cell_size));
    let grid_size_y = u32(ceil((grid_max_y - grid_min_y) / cell_size));
    let cell_index = cell_position.cell.x + cell_position.cell.y * grid_size_x;
    let cell_offset = cell_offsets[cell_index];
    cells[cell_offset + cell_position.offset] = object_index;
}
