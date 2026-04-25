#define_import_path calculate_cell_offsets

#import common::{ GridSize, flat_invocation_index }

const WORKGROUP_SIZE: u32 = 64;

@group(0) @binding(0) var<uniform> grid_size: GridSize;
@group(0) @binding(1) var<storage, read> cell_object_count: array<u32>;
@group(0) @binding(2) var<storage, read_write> current_cell_offset: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> cell_offsets: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn calculate_cell_offsets(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) nwg: vec3u
) {
    let i = flat_invocation_index(gid, nwg, WORKGROUP_SIZE);
    let grid_size = grid_size.inner;
    let cell_count = grid_size.x * grid_size.y;
    if i >= cell_count {
        return;
    }
    let cell_object_count = cell_object_count[i];
    if cell_object_count == 0 {
        return;
    }
    cell_offsets[i] = atomicAdd(&current_cell_offset, cell_object_count);
}