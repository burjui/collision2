#import common::GridSize

@group(0) @binding(1) var<uniform> grid_min_x: u32;
@group(0) @binding(2) var<uniform> grid_min_y: u32;
@group(0) @binding(3) var<uniform> grid_max_x: u32;
@group(0) @binding(4) var<uniform> grid_max_y: u32;
@group(0) @binding(5) var<uniform> cell_size: f32;
@group(0) @binding(6) var<storage, read_write> grid_size: GridSize;

@compute @workgroup_size(1)
fn calculate_grid_size() {
    let width = bitcast<f32>(grid_max_x) - bitcast<f32>(grid_min_x);
    let height = bitcast<f32>(grid_max_y) - bitcast<f32>(grid_min_y);
    let size_x = max(1, u32(ceil(width / cell_size)));
    let size_y = max(1, u32(ceil(height / cell_size)));
    grid_size = GridSize(size_x, size_y);
}
