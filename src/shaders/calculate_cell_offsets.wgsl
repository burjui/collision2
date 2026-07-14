#define_import_path calculate_cell_offsets

#import common::WORKGROUP_SIZE

var<immediate> thread_offset: u32;

@group(0) @binding(1) var<uniform> grid_min_x: f32;
@group(0) @binding(2) var<uniform> grid_min_y: f32;
@group(0) @binding(4) var<uniform> grid_size_x: u32;
@group(0) @binding(5) var<uniform> grid_size_y: u32;
@group(0) @binding(6) var<storage, read> cell_object_count: array<u32>;
@group(0) @binding(7) var<storage, read_write> current_cell_offset: atomic<u32>;
@group(0) @binding(8) var<storage, read_write> cell_offsets: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn calculate_cell_offsets(@builtin(global_invocation_id) gid: vec3u) {
    let cell_index = gid.x + thread_offset;
    let cell_count = grid_size_x * grid_size_y;
    if cell_index >= cell_count {
        return;
    }
    let count = cell_object_count[cell_index];
    cell_offsets[cell_index] = atomicAdd(&current_cell_offset, count);
}