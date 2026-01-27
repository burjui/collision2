#import common::{
    FLAG_DRAW_OBJECT, FLAG_PHYSICAL, FLAG_DRAW_AABB,
    AABB, Mass, Velocity, Position, Flags, BvhNode,
    invocation_index
}

@group(0) @binding(0) var<uniform> dt: f32;
@group(0) @binding(1) var<storage, read_write> flags: array<Flags>;
@group(0) @binding(2) var<storage, read> masses: array<Mass>;
@group(0) @binding(3) var<storage, read_write> velocities: array<Velocity>;
@group(0) @binding(4) var<storage, read_write> aabbs: array<AABB>;
@group(0) @binding(5) var<storage, read> nodes: array<BvhNode>;
@group(0) @binding(6) var<storage, read_write> force_acc: array<vec2f>;

const WORKGROUP_SIZE: u32 = 64;

struct BlackHole {
    position: vec2f,
    radius: f32,
    mass: f32,
    spin: f32
}

const BLACKHOLE_COUNT: u32 = 5;
const BLACKHOLES = array<BlackHole, BLACKHOLE_COUNT>(
    BlackHole(vec2f(-200, 500),     2,  10,  0 * -50),
    BlackHole(vec2f(500, 200),      1,  20,  0 * -50),
    BlackHole(vec2f(),              2,  20,  0 * 50),
    BlackHole(vec2f(-600, -300),    1,  20,  0 * -50),
    BlackHole(vec2f(600, -700),     1,  10,  0 * -50),
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

    var f = flags[i].inner;
    if (f & FLAG_PHYSICAL) == 0 {
        return;
    }

    let aabb = aabbs[i];
    let start_position = (aabb.min + aabb.max) / 2;
    var state = State(start_position, velocities[i].inner);
    state = integrate_euler_symplectic(state);

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

    flags[i].inner = f;
    velocities[i].inner = state.velocity;
    let offset = state.position - start_position;
    aabbs[i] = AABB(aabb.min + offset, aabb.max + offset);
}

struct State {
    position: vec2f,
    velocity: vec2f
}

fn integrate_euler_symplectic(state: State) -> State {
    let a = forces(state);
    var new_state = state;
    new_state.velocity += a * dt;
    new_state.position += new_state.velocity * dt;
    return new_state;
}

fn forces(state: State) -> vec2f {
    var acc = GLOBAL_FORCE;
    for (var bh_index: u32 = 0; bh_index < BLACKHOLE_COUNT; bh_index += 1) {
        var blackhole = BLACKHOLES[bh_index];
        acc += blackhole_gravity(blackhole, state.position);
        acc += frame_dragging(blackhole, state);
    }
    return acc;
}

fn blackhole_gravity(blackhole: BlackHole, position: vec2f) -> vec2f {
    let to_blackhole = blackhole.position - position;
    let direction = normalize(to_blackhole);
    let distance = length(to_blackhole);
    let bh_gravity = direction * GRAVITATIONAL_CONSTANT * blackhole.mass * BLACKHOLE_MASS_SCALE / (distance * distance);
    return bh_gravity;
}

// TODO fix this steaming pile of nonsense :)
fn frame_dragging(blackhole: BlackHole, state: State) -> vec2f {
    let blackhole_vector = blackhole.position - state.position;
    let r = length(blackhole_vector);
    let angular_momentum = blackhole.spin * vec2f(1, 1);
    let v_cross_r = cross(vec3f(state.velocity, 0), vec3f(blackhole_vector, 0)).z;
    let v_cross_j = cross(vec3f(state.velocity, 0), vec3f(angular_momentum, 0)).z;
    let a = 2 * GRAVITATIONAL_CONSTANT / pow(r, 3) *
        (v_cross_j - 3 * blackhole_vector * angular_momentum * v_cross_r / pow(r, 2));
    return a;
}
