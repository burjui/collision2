
#import common::INFINITY

@group(0) @binding(0) var<storage, read_write> grid_min_x: f32;
@group(0) @binding(1) var<storage, read_write> grid_max_x: f32;
@group(0) @binding(2) var<storage, read_write> grid_min_y: f32;
@group(0) @binding(3) var<storage, read_write> grid_max_y: f32;

@compute @workgroup_size(1)
fn reset_grid_aabb() {
    grid_min_x = INFINITY;
    grid_max_x = 0;
    grid_min_y = INFINITY;
    grid_max_y = 0;
}
