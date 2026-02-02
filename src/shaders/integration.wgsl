#import common::{
    FLAG_DRAW_OBJECT, FLAG_PHYSICAL, FLAG_DRAW_AABB, BVH_NODE_TREE_FLAG,
    AABB, Mass, Velocity, Position, Flags, BvhNode,
    invocation_index
}

@group(0) @binding(0) var<uniform> dt: f32;
@group(0) @binding(1) var<storage, read> flags: array<Flags>;
@group(0) @binding(2) var<storage, read> masses: array<Mass>;
@group(0) @binding(3) var<storage, read> velocities: array<Velocity>;
@group(0) @binding(4) var<storage, read> aabbs: array<AABB>;
@group(0) @binding(5) var<storage, read> nodes: array<BvhNode>;
@group(0) @binding(6) var<storage, read> node_count: u32;
@group(1) @binding(0) var<storage, read_write> integrated_flags: array<Flags>;
@group(1) @binding(1) var<storage, read_write> integrated_velocities: array<Velocity>;
@group(1) @binding(2) var<storage, read_write> integrated_aabbs: array<AABB>;
@group(1) @binding(3) var<storage, read_write> errors: atomic<u32>;

const WORKGROUP_SIZE: u32 = 64;

struct BlackHole {
    position: vec2f,
    radius: f32,
    mass: f32,
    spin: f32
}

const BLACKHOLE_COUNT: u32 = 2;
const BLACKHOLES = array<BlackHole, BLACKHOLE_COUNT>(
    // BlackHole(vec2f(-200, 500),     2,  10,  0 * -50),
    BlackHole(vec2f(500, 200),      1,  10,  0 * -50),
    BlackHole(vec2f(),              2,  10,  3 * 100),
    // BlackHole(vec2f(-600, -300),    1,  20,  0 * -50),
    // BlackHole(vec2f(600, -700),     1,  10,  0 * -50),
);
const BLACKHOLE_MASS_SCALE: f32 = 1 * 1000;
const BLACKHOLE_SIZE_SCALE: f32 = 10;
const BLACKHOLE_DESTROY_MATTER: bool = true;
const GRAVITATIONAL_CONSTANT: f32 = 1 * 100000;

const GLOBAL_FORCE = vec2f();

@compute @workgroup_size(WORKGROUP_SIZE)
fn cs_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let i = invocation_index(gid, WORKGROUP_SIZE);
    if i >= arrayLength(&masses) {
        return;
    }

    let aabb = aabbs[i];
    let initial_position = (aabb.min + aabb.max) / 2;
    let initial_velocity = velocities[i].inner;

    var f = flags[i].inner;

    let mass = masses[i].inner;
    var state = State(initial_position, velocities[i].inner);
    if (f & FLAG_PHYSICAL) != 0 {
        state = integrate_euler_symplectic(state, i, aabb, mass);
    }

    let size = aabb.max - aabb.min;
    if BLACKHOLE_DESTROY_MATTER {
        for (var bh_index: u32 = 0; bh_index < BLACKHOLE_COUNT && (f & FLAG_PHYSICAL) != 0; bh_index++) {
            let blackhole = BLACKHOLES[bh_index];
            let distance = length(blackhole.position - state.position) - max(size.x, size.y) / 2;
            if distance < blackhole.radius * BLACKHOLE_SIZE_SCALE {
                f &= ~(FLAG_PHYSICAL | FLAG_DRAW_OBJECT | FLAG_DRAW_AABB);
                state.velocity = vec2f();
            }
        }
    }

    let offset = state.position - initial_position;
    var new_aabb = AABB(aabb.min + offset, aabb.max + offset);
    integrated_flags[i].inner = f;
    integrated_aabbs[i] = new_aabb;
    integrated_velocities[i].inner = state.velocity;
}

struct State {
    position: vec2f,
    velocity: vec2f
}

fn integrate_euler_symplectic(state: State, index: u32, aabb: AABB, mass: f32) -> State {
    let a = forces(state, index, aabb, mass) / mass;
    var new_state = state;
    new_state.velocity += a * dt;
    new_state.position += new_state.velocity * dt;
    return new_state;
}

fn forces(state: State, index: u32, aabb: AABB, mass: f32) -> vec2f {
    var f = GLOBAL_FORCE;
    for (var bh_index: u32 = 0; bh_index < BLACKHOLE_COUNT; bh_index += 1) {
        var blackhole = BLACKHOLES[bh_index];
        f += blackhole_gravity(blackhole, state.position, mass);
        f += frame_dragging(blackhole, state);
    }
    f += collision_repulsion(index, aabb, state.velocity, mass);
    return f;
}

fn blackhole_gravity(blackhole: BlackHole, position: vec2f, mass: f32) -> vec2f {
    let to_blackhole = blackhole.position - position;
    let direction = normalize(to_blackhole);
    let distance = length(to_blackhole);
    return direction * GRAVITATIONAL_CONSTANT * mass * blackhole.mass * BLACKHOLE_MASS_SCALE / (distance * distance);
}

// Lense–Thirring formula for 2D
// NOTE: some terms are missing and have to be reintroduced for 3D
fn frame_dragging(blackhole: BlackHole, state: State) -> vec2f {
    let r_vec = blackhole.position - state.position;
    let r = length(r_vec);
    let J = blackhole.spin; // scalar angular momentum (Jz)
    let v_perp = vec2f(-state.velocity.y, state.velocity.x); // v rotated by +90 degrees
    return (2.0 * GRAVITATIONAL_CONSTANT * J / pow(r, 3.0)) * v_perp;
}

fn collision_repulsion(index: u32, aabb: AABB, velocity: vec2f, mass: f32) -> vec2f {
    const MAX_STACK_DEPTH: u32 = 64; // 2 * max tree depth

    var stack: array<u32, MAX_STACK_DEPTH>;
    var sp = 0u;
    var f = vec2f();

    stack[sp] = node_count - 1; // root
    sp++;

    while sp > 0 {
        let node_index = stack[sp - 1];
        sp--;

        let other_index = nodes[node_index].index;
        if (other_index & BVH_NODE_TREE_FLAG) != 0 {
            if sp >= MAX_STACK_DEPTH - 2 {
                atomicAdd(&errors, 1);
                break;
            }

            let i = other_index & ~BVH_NODE_TREE_FLAG;
            stack[sp] = i;
            stack[sp + 1] = i + 1;
            sp += 2;
        } else if other_index != index && (flags[other_index].inner & FLAG_PHYSICAL) != 0 {
            let other_aabb = aabbs[other_index];
            if aabb_overlaps(aabb, other_aabb) {
                // TODO: pairwise force accumulation for precision
                f += collision_repulsion_pair(aabb, other_aabb, velocity, mass, other_index);
            }
        }
    }

    return f;
}

fn collision_repulsion_pair(aabb: AABB, other_aabb: AABB, velocity: vec2f, mass: f32, other_index: u32) -> vec2f {
    let size = aabb.max - aabb.min;
    let other_size = other_aabb.max - other_aabb.min;
    let position = (aabb.min + aabb.max) / 2;
    let other_position = (other_aabb.min + other_aabb.max) / 2;
    let separation_vector = position - other_position;
    let distance = length(separation_vector);
    let r1 = 0.5 * length(size);
    let r2 = 0.5 * length(other_size);
    if distance > (r1 + r2) {
        return vec2f();
    }

    const stiffness = 10000.0;

    let f_elastic = stiffness * pow((1 - distance / min(r1, r2)), 2);
    let direction = normalize(separation_vector);
    let vn = dot(velocity - velocities[other_index].inner, direction);
    let m1 = mass;
    let m2 = masses[other_index].inner;
    let m_eff = m1 * m2 / (m1 + m2);
    var f_damping: f32 = 0;
    if vn < 0 {
        const e: f32 = 2.71828;
        let gamma =
            (3.0 / 2.0) *
            (1.0 - e * e) / sqrt(5.0) *
            sqrt(stiffness * m_eff);
        f_damping = -gamma * vn;
    }
    return (f_elastic - f_damping) * direction;
}

fn aabb_overlaps(a: AABB, b: AABB) -> bool {
    return a.min.x < b.max.x &&
           a.max.x > b.min.x &&
           a.min.y < b.max.y &&
           a.max.y > b.min.y;
}