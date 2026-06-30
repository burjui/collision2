#import common::GridSize

@group(0) @binding(0) var<uniform> grid_size: GridSize;
@group(0) @binding(1) var<storage, read_write> cell_object_count: array<u32>;

@compute @workgroup_size(1)
fn reset_cell_object_count(@builtin(global_invocation_id) gid: vec3u) {
    let original_i = gid.x + gid.y * grid_size.x;
    let cell_count = grid_size.x * grid_size.y;
    let i = select(0, original_i, original_i < cell_count);
    cell_object_count[i] = 0;
}
