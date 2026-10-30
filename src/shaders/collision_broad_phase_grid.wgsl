#import common::{
    Position, Flags, Mass, Velocity, CollisionCandidate, CellPosition,
    FLAG_PHYSICAL, FLAG_COLLISION, MAX_CANDIDATES_PER_OBJECT, WORKGROUP_SIZE
}

var<immediate> thread_offset: u32;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<uniform> particle_radius: f32;
@group(0) @binding(2) var<uniform> grid_min_x: f32;
@group(0) @binding(3) var<uniform> grid_min_y: f32;
@group(0) @binding(5) var<uniform> grid_size_x: u32;
@group(0) @binding(6) var<uniform> grid_size_y: u32;
@group(0) @binding(7) var<uniform> kick_center: vec2f;
@group(0) @binding(8) var<uniform> kick_radius: f32;
@group(0) @binding(9) var<uniform> kick_magnitude: f32;

@group(1) @binding(0) var<storage, read> object_cells: array<CellPosition>;
@group(1) @binding(1) var<storage, read> cell_object_count: array<u32>;
@group(1) @binding(2) var<storage, read> cell_offsets: array<u32>;
@group(1) @binding(3) var<storage, read> cells: array<u32>;

@group(2) @binding(0) var<storage, read> positions: array<Position>;
@group(2) @binding(1) var<storage, read> flags: array<Flags>;
@group(2) @binding(2) var<storage, read_write> velocities: array<Velocity>;

@group(3) @binding(0) var<storage, read_write> candidates: array<CollisionCandidate>;
@group(3) @binding(1) var<storage, read_write> candidate_count: atomic<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn broad_phase_grid(@builtin(global_invocation_id) gid: vec3u) {
    let object_index = gid.x + thread_offset;
    if object_index >= object_count ||
       (flags[object_index].inner & FLAG_PHYSICAL) == 0 ||
       (flags[object_index].inner & FLAG_COLLISION) == 0
    {
        return;
    }

    var c1 = positions[object_index].inner;
    let kick_vector = c1 - kick_center;
    let kick_vector_length_squared = dot(kick_vector, kick_vector);
    if kick_vector_length_squared < kick_radius * kick_radius && kick_magnitude > 0 {
        velocities[object_index].inner += normalize(kick_vector) * kick_magnitude;
    }

    let max_candidates = object_count * MAX_CANDIDATES_PER_OBJECT;
    let cell = vec2i(object_cells[object_index].cell);
    let min_cell = vec2u(max(vec2i(), cell - vec2i(1)));
    let max_cell = vec2u(min(cell + vec2i(1), vec2i(vec2u(grid_size_x - 1, grid_size_y - 1))));
    let particle_size = particle_radius * 2;
    let particle_size_squared = particle_size * particle_size;
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
