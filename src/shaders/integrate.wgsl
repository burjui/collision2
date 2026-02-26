#import common::{
    FLAG_DRAW_OBJECT, FLAG_PHYSICAL, FLAG_DRAW_AABB, BVH_NODE_TREE_FLAG,
    AABB, Mass, Velocity, Position, Flags, BvhNode,
    flat_invocation_index
}

@group(0) @binding(0) var<uniform> dt: f32;
@group(0) @binding(1) var<uniform> gravitational_constant: f32;
@group(0) @binding(2) var<uniform> global_force: vec2f;
@group(0) @binding(3) var<storage, read> masses: array<Mass>;
@group(0) @binding(4) var<storage, read> nodes: array<BvhNode>;

@group(1) @binding(0) var<uniform> blackhole_count: u32;
@group(1) @binding(1) var<uniform> blackhole_mass_scale: f32;
@group(1) @binding(2) var<uniform> blackhole_size_scale: f32;
@group(1) @binding(3) var<storage, read> blackholes: array<BlackHole>;

@group(2) @binding(0) var<storage, read> flags: array<Flags>;
@group(2) @binding(1) var<storage, read> aabbs: array<AABB>;
@group(2) @binding(2) var<storage, read> velocities: array<Velocity>;
@group(2) @binding(3) var<storage, read_write> integrated_flags: array<Flags>;
@group(2) @binding(4) var<storage, read_write> integrated_aabbs: array<AABB>;
@group(2) @binding(5) var<storage, read_write> integrated_velocities: array<Velocity>;

const WORKGROUP_SIZE: u32 = 64;

struct BlackHole {
    position: vec2f,
    radius: f32,
    mass: f32,
    spin: f32,
}

const STIFFNESS: f32 = 300000;
const RESTITUTION: f32 = 0.3;
const GAMMA_COEFF: f32 = (3.0 / 2.0) * (1.0 - RESTITUTION * RESTITUTION) / sqrt(5.0) * sqrt(STIFFNESS);

@compute @workgroup_size(WORKGROUP_SIZE)
fn integrate(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) nwg: vec3u,
) {
    let i = flat_invocation_index(gid, nwg, WORKGROUP_SIZE);
    if i >= arrayLength(&masses) {
        return;
    }

    let aabb = aabbs[i];
    let initial_position = (aabb.min + aabb.max) / 2;
    let initial_velocity = velocities[i].inner;

    var f = flags[i].inner;

    let mass = masses[i].inner;
    var state = ObjectPhaseState(initial_position, velocities[i].inner);
    if (f & FLAG_PHYSICAL) != 0 {
        state = integrate_euler_symplectic(state, i, aabb, mass);
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
    integrated_flags[i].inner = f;
    integrated_aabbs[i] = new_aabb;
    integrated_velocities[i].inner = state.velocity;
}

struct ObjectPhaseState {
    position: vec2f,
    velocity: vec2f
}

fn integrate_euler_symplectic(state: ObjectPhaseState, index: u32, aabb: AABB, mass: f32) -> ObjectPhaseState {
    let a = forces(state, index, aabb, mass) / mass;
    var new_state = state;
    new_state.velocity += a * dt;
    new_state.position += new_state.velocity * dt;
    return new_state;
}

struct InteractionResult {
    force: vec2f,
    collision_count: u32
}

fn forces(state: ObjectPhaseState, index: u32, aabb: AABB, mass: f32) -> vec2f {
    var total_force = global_force;
    for (var bh_index: u32 = 0; bh_index < blackhole_count; bh_index += 1) {
        var blackhole = blackholes[bh_index];
        total_force += blackhole_gravity(blackhole, state.position, mass);
        total_force += frame_dragging(blackhole, state);
    }
    total_force += collision_repulsion(index, aabb, state.velocity, mass);
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
fn frame_dragging(blackhole: BlackHole, state: ObjectPhaseState) -> vec2f {
    let r_vec = blackhole.position - state.position;
    let r = length(r_vec);
    let J = blackhole.spin; // scalar angular momentum (Jz)
    let v_perp = vec2f(-state.velocity.y, state.velocity.x); // v rotated by +90 degrees
    return (2.0 * gravitational_constant * J / pow(r, 3.0)) * v_perp;
}

fn collision_repulsion(index: u32, aabb: AABB, velocity: vec2f, mass: f32) -> vec2f {
    const MAX_STACK_DEPTH: u32 = 64; // 2 * max tree depth

    var stack: array<u32, MAX_STACK_DEPTH>;
    var sp: u32 = 0;
    var total_force = vec2f();

    stack[sp] = arrayLength(&nodes) - 1; // root
    sp++;

    while sp > 0 {
        let node_index = stack[sp - 1];
        sp--;

        let other_aabb = aabbs[node_index];
        if !aabb_overlaps(aabb, other_aabb) {
            continue;
        }

        let other_index = nodes[node_index].index;
        if (other_index & BVH_NODE_TREE_FLAG) != 0 {
            if sp >= MAX_STACK_DEPTH - 2 {
                break;
            }

            let i = other_index & ~BVH_NODE_TREE_FLAG;
            stack[sp] = i;
            stack[sp + 1] = i + 1;
            sp += 2;
        } else if other_index != index && (flags[other_index].inner & FLAG_PHYSICAL) != 0 {
            // TODO: pairwise force accumulation for precision
            total_force += collision_repulsion_pair(aabb, other_aabb, velocity, mass, other_index);
        }
    }

    return total_force;
}

fn collision_repulsion_pair(aabb: AABB, other_aabb: AABB, velocity: vec2f, mass: f32, other_index: u32) -> vec2f {
    let size = aabb.max - aabb.min;
    let other_size = other_aabb.max - other_aabb.min;
    let position = (aabb.min + aabb.max) / 2;
    let other_position = (other_aabb.min + other_aabb.max) / 2;
    let separation_vector = position - other_position;
    let distance = length(separation_vector);
    let r1 = 0.5 * size.x;
    let r2 = 0.5 * other_size.x;
    let interaction_distance = r1 + r2;
    if distance >= interaction_distance {
        return vec2f();
    }

    let penetration = interaction_distance - distance;
    let n = normalize(separation_vector);
    var v_ij_n = dot(velocity - velocities[other_index].inner, n);
    if penetration <= 0 && v_ij_n > 0 {
        v_ij_n = -RESTITUTION * v_ij_n;
    }
    let m1 = mass;
    let m2 = masses[other_index].inner;
    let m_eff = m1 * m2 / (m1 + m2);
    var f_damping = vec2f();
    if v_ij_n < 0 {
        f_damping = -GAMMA_COEFF * sqrt(m_eff) * v_ij_n * n;
    }
    let f_elastic = STIFFNESS * penetration * n;
    return f_elastic + f_damping;
}

fn aabb_overlaps(a: AABB, b: AABB) -> bool {
    return a.min.x < b.max.x &&
           a.max.x > b.min.x &&
           a.min.y < b.max.y &&
           a.max.y > b.min.y;
}
