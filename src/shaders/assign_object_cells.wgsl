#import common::{ AABB, GridSize, CellPosition, flat_invocation_index }

const WORKGROUP_SIZE: u32 = 64;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<uniform> grid_position_x: f32;
@group(0) @binding(2) var<uniform> grid_position_y: f32;
@group(0) @binding(3) var<uniform> grid_size: GridSize;
@group(0) @binding(4) var<uniform> cell_size: f32;
@group(0) @binding(5) var<storage, read_write> cell_object_count: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> object_cells: array<CellPosition>;

@group(1) @binding(0) var<storage, read> aabbs: array<AABB>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn assign_object_cells(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) nwg: vec3u
) {
    let i = flat_invocation_index(gid, nwg, WORKGROUP_SIZE);
    if i >= object_count {
        return;
    }

    let aabb = aabbs[i];
    let cell_x = u32((aabb.min.x - grid_position_x) / cell_size);
    let cell_y = u32((aabb.min.y - grid_position_y) / cell_size);
    let cell_offset = cell_x + cell_y * grid_size.inner.x;
    atomicAdd(&cell_object_count[cell_offset], 1);
    object_cells[i] = CellPosition(vec2u(cell_x, cell_y));
}
