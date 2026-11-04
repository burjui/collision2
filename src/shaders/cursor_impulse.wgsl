#import common::{
    Flags, Position, Velocity,
    FLAG_INTEGRATE, FLAG_COLLIDE, WORKGROUP_SIZE
}

var<immediate> thread_offset: u32;

@group(0) @binding(0) var<uniform> object_count: u32;
@group(0) @binding(1) var<uniform> cursor_position: vec2f;
@group(0) @binding(2) var<uniform> cursor_impulse_radius: f32;
@group(0) @binding(3) var<uniform> cursor_impulse_speed: f32;

@group(1) @binding(0) var<storage, read> flags: array<Flags>;
@group(1) @binding(1) var<storage, read> positions: array<Position>;
@group(1) @binding(2) var<storage, read_write> velocities: array<Velocity>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn apply_cursor_impulse(@builtin(global_invocation_id) gid: vec3u) {
    let object_index = gid.x + thread_offset;
    if object_index >= object_count || (flags[object_index].inner & FLAG_INTEGRATE) == 0
    {
        return;
    }

    let impulse_vector = positions[object_index].inner - cursor_position;
    let impulse_vector_length_squared = dot(impulse_vector, impulse_vector);
    if impulse_vector_length_squared < cursor_impulse_radius * cursor_impulse_radius && cursor_impulse_speed > 0 {
        velocities[object_index].inner += normalize(impulse_vector) * cursor_impulse_speed;
    }
}