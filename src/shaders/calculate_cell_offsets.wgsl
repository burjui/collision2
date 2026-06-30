#define_import_path calculate_cell_offsets

#import common::GridSize

@group(0) @binding(0) var<uniform> grid_size: GridSize;
@group(0) @binding(1) var<storage, read> cell_object_count: array<u32>;
@group(0) @binding(2) var<storage, read_write> current_cell_offset: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> cell_offsets: array<u32>;

@compute @workgroup_size(1)
fn calculate_cell_offsets(@builtin(global_invocation_id) gid: vec3u) {
    let cell_index = gid.x + gid.y * grid_size.x;
    let cell_count = grid_size.x * grid_size.y;
    if cell_index >= cell_count {
        return;
    }
    let count = cell_object_count[cell_index];
    cell_offsets[cell_index] = atomicAdd(&current_cell_offset, count);
}