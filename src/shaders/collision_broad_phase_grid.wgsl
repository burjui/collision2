#import common::{
    AABB, Flags, CollisionCandidate, CellPosition,
    FLAG_PHYSICAL, FLAG_COLLISION, MAX_CANDIDATES_PER_OBJECT, WORKGROUP_SIZE
}

var<immediate> thread_offset: u32;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<uniform> grid_min_x: f32;
@group(0) @binding(3) var<uniform> grid_min_y: f32;
@group(0) @binding(5) var<uniform> cell_size: f32;
@group(0) @binding(4) var<uniform> grid_size_x: u32;
@group(0) @binding(2) var<uniform> grid_size_y: u32;
@group(0) @binding(6) var<storage, read> object_cells: array<CellPosition>;
@group(0) @binding(7) var<storage, read> cell_object_count: array<u32>;
@group(0) @binding(8) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(9) var<storage, read> cells: array<u32>;
@group(0) @binding(10) var<storage, read_write> candidates: array<CollisionCandidate>;
@group(0) @binding(11) var<storage, read_write> candidate_count: atomic<u32>;

@group(1) @binding(1) var<storage, read> aabbs: array<AABB>;
@group(1) @binding(2) var<storage, read> flags: array<Flags>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn broad_phase_grid(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_index) local_invocation_index: u32
) {
    let object_index = gid.x + thread_offset;
    if object_index >= object_count || (flags[object_index].inner & FLAG_PHYSICAL) == 0 {
        return;
    }

    let aabb = aabbs[object_index];
    let max_candidates = object_count * MAX_CANDIDATES_PER_OBJECT;
    let cell = vec2i(object_cells[object_index].cell);
    let min_cell = vec2u(max(vec2i(), cell - vec2i(1, 1)));
    let max_cell = vec2u(min(cell + vec2i(1, 1), vec2i(vec2u(grid_size_x - 1, grid_size_y - 1))));
    for (var i = min_cell.x; i <= max_cell.x; i++) {
        for (var j = min_cell.y; j <= max_cell.y; j++) {
            let cell_index = i + j * grid_size_x;
            let object_count = cell_object_count[cell_index];
            if object_count == 0 {
                continue;
            }
            let cell_offset = cell_offsets[cell_index];
            for (var k = 0u; k < object_count; k++) {
                let other_object_index = cells[cell_offset + k];
                if other_object_index >= object_index {
                    continue;
                }
                if (flags[other_object_index].inner & FLAG_COLLISION) == 0 {
                    continue;
                }
                let object_aabb = aabbs[other_object_index];
                if !aabb_overlaps(aabb, object_aabb) {
                    continue;
                }
                let candidates_index = atomicAdd(&candidate_count, 1);
                if candidates_index >= max_candidates {
                    return;
                }
                candidates[candidates_index] = CollisionCandidate(object_index, other_object_index);
            }
        }
    }
}

fn aabb_overlaps(a: AABB, b: AABB) -> bool {
    return a.min.x < b.max.x &&
           a.max.x > b.min.x &&
           a.min.y < b.max.y &&
           a.max.y > b.min.y;
}
