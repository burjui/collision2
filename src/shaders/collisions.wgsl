#import common::{
    Position, Flags, Velocity, CellPosition,
    FLAG_INTEGRATE, FLAG_COLLISION, MAX_CANDIDATES_PER_OBJECT, WORKGROUP_SIZE
}

var<immediate> thread_offset: u32;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<uniform> particle_radius: f32;
@group(0) @binding(2) var<uniform> grid_min_x: f32;
@group(0) @binding(3) var<uniform> grid_min_y: f32;
@group(0) @binding(5) var<uniform> grid_size_x: u32;
@group(0) @binding(6) var<uniform> grid_size_y: u32;
@group(0) @binding(10) var<uniform> stiffness: f32;
@group(0) @binding(11) var<uniform> gamma_coeff: f32;
@group(0) @binding(12) var<uniform> sqrt_m_eff: f32;

@group(1) @binding(0) var<storage, read> object_cells: array<CellPosition>;
@group(1) @binding(1) var<storage, read> cell_object_count: array<u32>;
@group(1) @binding(2) var<storage, read> cell_offsets: array<u32>;
@group(1) @binding(3) var<storage, read> cells: array<u32>;

@group(2) @binding(0) var<storage, read> positions: array<Position>;
@group(2) @binding(1) var<storage, read> flags: array<Flags>;
@group(2) @binding(2) var<storage, read_write> velocities: array<Velocity>;

@group(3) @binding(0) var<storage, read_write> forces: array<vec2f>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn process_collisions(@builtin(global_invocation_id) gid: vec3u) {
    let object_index = gid.x + thread_offset;
    if object_index >= object_count ||
       (flags[object_index].inner & FLAG_INTEGRATE) == 0 ||
       (flags[object_index].inner & FLAG_COLLISION) == 0
    {
        return;
    }
    let cell = vec2i(object_cells[object_index].cell);
    let min_cell = vec2u(max(vec2i(), cell - vec2i(1)));
    let max_cell = vec2u(min(cell + vec2i(1), vec2i(vec2u(grid_size_x - 1, grid_size_y - 1))));
    let particle_size = particle_radius * 2;
    let particle_size_squared = particle_size * particle_size;
    var force = vec2f();
    var c1 = positions[object_index].inner;
    let v1 = velocities[object_index].inner;
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
                if other_object_index == object_index || (flags[other_object_index].inner & FLAG_COLLISION) == 0 {
                    continue;
                }

                let c2 = positions[other_object_index].inner;
                let separation_vector = c1 - c2;
                let distance_squared = dot(separation_vector, separation_vector);
                if distance_squared > particle_size_squared || distance_squared < 1e-10 {
                    continue;
                }

                let a = object_index;
                let b = other_object_index;
                let v2 = velocities[b].inner;
                force += collision_force(v1, v2, separation_vector, sqrt(distance_squared));
            }
        }
    }

    forces[object_index] += force;
}

fn collision_force(
    v1: vec2f,
    v2: vec2f,
    separation_vector: vec2f,
    distance: f32
) -> vec2f {
    let n = separation_vector / distance;
    let interaction_distance = particle_radius * 2;
    let penetration = interaction_distance - distance;
    let f_elastic = stiffness * max(0.0, penetration) * n;

    let v_rel = v1 - v2;
    let v_n = dot(v_rel, n);
    let f_damping = -gamma_coeff * sqrt_m_eff * min(0.0, v_n) * n;
    return f_elastic + f_damping;
}
