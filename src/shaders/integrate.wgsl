#import common::{
    FLAG_DRAW_OBJECT, FLAG_PHYSICAL, FLAG_DRAW_AABB,
    AABB, Mass, Velocity, Flags,
}

var <immediate> thread_offset: u32;

@group(0) @binding(0) var<uniform> dt: f32;
@group(0) @binding(1) var<uniform> gravitational_constant: f32;
@group(0) @binding(2) var<uniform> global_acceleration: vec2f;
@group(0) @binding(3) var<uniform> object_count: u32;
@group(0) @binding(4) var<storage, read> masses: array<Mass>;

@group(1) @binding(0) var<uniform> blackhole_count: u32;
@group(1) @binding(1) var<uniform> blackhole_mass_scale: f32;
@group(1) @binding(2) var<uniform> blackhole_size_scale: f32;
@group(1) @binding(3) var<storage, read> blackholes: array<BlackHole>;

@group(2) @binding(0) var<storage, read> collision_forces: array<vec2f>;

@group(3) @binding(0) var<storage, read> flags: array<Flags>;
@group(3) @binding(1) var<storage, read> aabbs: array<AABB>;
@group(3) @binding(3) var<storage, read> velocities: array<Velocity>;
@group(3) @binding(4) var<storage, read_write> integrated_flags: array<Flags>;
@group(3) @binding(5) var<storage, read_write> integrated_aabbs: array<AABB>;
@group(3) @binding(6) var<storage, read_write> integrated_velocities: array<Velocity>;

const WORKGROUP_SIZE: u32 = 64;

struct BlackHole {
    position: vec2f,
    radius: f32,
    mass: f32,
    spin: f32,
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn integrate(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
) {
    let object_index = gid.x + thread_offset;
    if object_index >= object_count {
        return;
    }

    let aabb = aabbs[object_index];
    let initial_position = (aabb.min + aabb.max) / 2;
    let initial_velocity = velocities[object_index].inner;
    let mass = masses[object_index].inner;
    var f = flags[object_index].inner;
    var state = ObjectPhaseState(initial_position, velocities[object_index].inner);
    if (f & FLAG_PHYSICAL) != 0 {
        state = integrate_euler_symplectic(state, object_index, aabb, mass);
    }

    let size = aabb.max - aabb.min;
    for (var bh_index: u32 = 0; bh_index < blackhole_count && (f & FLAG_PHYSICAL) != 0; bh_index++) {
        let blackhole = blackholes[bh_index];
        let distance = length(blackhole.position - state.position) - max(size.x, size.y) / 2;
        if distance < blackhole.radius * blackhole_size_scale {
            f &= ~(FLAG_PHYSICAL | FLAG_DRAW_OBJECT | FLAG_DRAW_AABB);
            state.velocity = vec2f();
        }
    }

    const CONSTRAINTS = AABB(vec2f(-3200, -2000), vec2f(3200, 2000));

    let offset = state.position - initial_position;
    var new_aabb = AABB(aabb.min + offset, aabb.max + offset);
    if new_aabb.min.x < CONSTRAINTS.min.x {
        let overshoot = -new_aabb.min.x + CONSTRAINTS.min.x;
        new_aabb.min.x += overshoot * 0.5;
        new_aabb.max.x += overshoot * 0.5;
        state.velocity.x *= -1;
    }
    if new_aabb.max.x > CONSTRAINTS.max.x {
        let overshoot = new_aabb.max.x - CONSTRAINTS.max.x;
        new_aabb.min.x -= overshoot * 0.5;
        new_aabb.max.x -= overshoot * 0.5;
        state.velocity.x *= -1;
    }
    if new_aabb.min.y < CONSTRAINTS.min.y {
        let overshoot = -new_aabb.min.y + CONSTRAINTS.min.y;
        new_aabb.min.y += overshoot * 0.5;
        new_aabb.max.y += overshoot * 0.5;
        state.velocity.y *= -1;
    }
    if new_aabb.max.y > CONSTRAINTS.max.y {
        let overshoot = new_aabb.max.y - CONSTRAINTS.max.y;
        new_aabb.min.y -= overshoot * 0.5;
        new_aabb.max.y -= overshoot * 0.5;
        state.velocity.y *= -1;
    }
    integrated_flags[object_index].inner = f;
    integrated_aabbs[object_index] = new_aabb;
    integrated_velocities[object_index].inner = state.velocity;
}

struct ObjectPhaseState {
    position: vec2f,
    velocity: vec2f
}

fn integrate_euler_symplectic(state: ObjectPhaseState, index: u32, aabb: AABB, mass: f32) -> ObjectPhaseState {
    let a = global_acceleration + forces(state, index, aabb, mass) / mass;
    var new_state = state;
    new_state.velocity += a * dt;
    new_state.position += new_state.velocity * dt;
    return new_state;
}

fn forces(state: ObjectPhaseState, index: u32, aabb: AABB, mass: f32) -> vec2f {
    var total_force = vec2f();
    for (var bh_index: u32 = 0; bh_index < blackhole_count; bh_index += 1) {
        var blackhole = blackholes[bh_index];
        total_force += blackhole_gravity(blackhole, state.position, mass);
        total_force += frame_dragging(blackhole, state, mass);
    }
    total_force += collision_forces[index];
    return total_force;
}

fn blackhole_gravity(blackhole: BlackHole, position: vec2f, mass: f32) -> vec2f {
    let to_blackhole = blackhole.position - position;
    let direction = normalize(to_blackhole);
    let distance = length(to_blackhole);
    return direction * gravitational_constant * mass * blackhole.mass * blackhole_mass_scale / (distance * distance);
}

// Lense–Thirring formula for 2D
// NOTE: some terms are missing and have to be reintroduced for 3D
fn frame_dragging(blackhole: BlackHole, state: ObjectPhaseState, mass: f32) -> vec2f {
    let r_vec = blackhole.position - state.position;
    let r = length(r_vec);
    let J = blackhole.spin * blackhole.mass * blackhole.mass; // scalar angular momentum (Jz)
    let v_perp = vec2f(-state.velocity.y, state.velocity.x); // v rotated by +90 degrees
    return mass * (2.0 * gravitational_constant * J / pow(r, 3.0)) * v_perp;
}
