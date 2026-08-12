#import common::{
    Position, Flags, Mass, CollisionCandidate, CellPosition,
    FLAG_PHYSICAL, FLAG_COLLISION, MAX_CANDIDATES_PER_OBJECT, WORKGROUP_SIZE
}

var<immediate> thread_offset: u32;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<uniform> particle_radius: f32;
@group(0) @binding(2) var<uniform> grid_min_x: f32;
@group(0) @binding(3) var<uniform> grid_min_y: f32;
@group(0) @binding(5) var<uniform> grid_size_x: u32;
@group(0) @binding(6) var<uniform> grid_size_y: u32;
@group(0) @binding(7) var<storage, read> object_cells: array<CellPosition>;
@group(0) @binding(8) var<storage, read> cell_object_count: array<u32>;
@group(0) @binding(9) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(10) var<storage, read> cells: array<u32>;
@group(0) @binding(11) var<storage, read_write> candidates: array<CollisionCandidate>;
@group(0) @binding(12) var<storage, read_write> candidate_count: atomic<u32>;
@group(0) @binding(13) var<storage, read> masses: array<Mass>;

@group(1) @binding(1) var<storage, read> positions: array<Position>;
@group(1) @binding(2) var<storage, read> flags: array<Flags>;

@group(2) @binding(0) var<storage, read_write> forces: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn broad_phase_grid(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_index) local_invocation_index: u32
) {
    let object_index = gid.x + thread_offset;
    if object_index >= object_count ||
       (flags[object_index].inner & FLAG_PHYSICAL) == 0 ||
       (flags[object_index].inner & FLAG_COLLISION) == 0
    {
        return;
    }

    const FORCE_AREA_SIZE: i32 = 1;

    // let m1 = masses[object_index].inner;
    let c1 = positions[object_index].inner;
    let max_candidates = object_count * MAX_CANDIDATES_PER_OBJECT;
    let cell = vec2i(object_cells[object_index].cell);
    let min_cell = vec2u(max(vec2i(), cell - vec2i(FORCE_AREA_SIZE)));
    let max_cell = vec2u(min(cell + vec2i(FORCE_AREA_SIZE), vec2i(vec2u(grid_size_x - 1, grid_size_y - 1))));
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

                let c2 = positions[other_object_index].inner;
                if (flags[other_object_index].inner & FLAG_COLLISION) == 0 {
                    continue;
                }

                let delta = c1 - c2;
                let distance_squared = dot(delta, delta);
                let particle_size = particle_radius * 2;
                let particle_size_squared = particle_size * particle_size;
                if distance_squared > particle_size_squared || distance_squared < 1e-10 {
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

fn cas_add_force(i: u32, value: vec2f) {
    cas_add_force_component(i * 2, value.x);
    cas_add_force_component(i * 2 + 1, value.y);
}

fn cas_add_force_component(i: u32, value: f32) {
    var old = atomicLoad(&forces[i]);
    loop {
        let old_f = bitcast<f32>(old);
        let new_f = old_f + value;
        let new_value = bitcast<u32>(new_f);
        let res = atomicCompareExchangeWeak(&forces[i], old, new_value);
        if res.exchanged {
            break;
        }
        old = res.old_value;
    }
}
