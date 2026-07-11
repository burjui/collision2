#import common::{ AABB, CellPosition, WORKGROUP_SIZE }

var<immediate> thread_offset: u32;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<uniform> grid_min_x: f32;
@group(0) @binding(2) var<uniform> grid_min_y: f32;
@group(0) @binding(3) var<uniform> cell_size: f32;
@group(0) @binding(4) var<uniform> grid_size_x: u32;
@group(0) @binding(5) var<storage, read_write> cell_object_count: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> object_cells: array<CellPosition>;

@group(1) @binding(0) var<storage, read> aabbs: array<AABB>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn assign_object_cells(@builtin(global_invocation_id) gid: vec3u) {
    let object_index = gid.x + thread_offset;
    if object_index >= object_count {
        return;
    }

    let aabb = aabbs[object_index];
    let center_x = (aabb.min.x + aabb.max.x) / 2;
    let center_y = (aabb.min.y + aabb.max.y) / 2;
    let cell_x = u32(max(0, (center_x - grid_min_x) / cell_size));
    let cell_y = u32(max(0, (center_y - grid_min_y) / cell_size));
    let cell_index = cell_x + cell_y * grid_size_x;
    let offset = atomicAdd(&cell_object_count[cell_index], 1);
    object_cells[object_index] = CellPosition(vec2u(cell_x, cell_y), offset);
}
