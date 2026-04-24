#import common::{ CellPosition, flat_invocation_index }

const WORKGROUP_SIZE: u32 = 64;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<storage, read_write> cell_object_count: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn assign_object_cells() {

}