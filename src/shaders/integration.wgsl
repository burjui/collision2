#import common::{FLAG_DRAW_OBJECT, FLAG_PHYSICAL, AABB, Mass, Velocity, Position, Flags, invocation_index}

@group(0) @binding(0) var<uniform> dt: f32;
@group(0) @binding(1) var<storage, read> masses: array<Mass>;
@group(0) @binding(2) var<storage, read_write> flags: array<Flags>;
@group(0) @binding(3) var<storage, read_write> aabbs: array<AABB>;
@group(0) @binding(4) var<storage, read_write> velocities: array<Velocity>;

const WORKGROUP_SIZE: u32 = 64;

const BLACKHOLE_POSITION = vec2f();
const BLACKHOLE_MASS = 1000000.0;

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
    var v = velocities[i].inner;
    var x = (aabb.min + aabb.max) / 2;
    let size = aabb.max - aabb.min;
    let to_blackhole = BLACKHOLE_POSITION - x;
    let distance = length(to_blackhole) - max(size.x, size.y) / 2;
    // let direction = normalize(to_blackhole);
    // let bh_gravity = direction * 1000000 / (distance * distance);
    // let a = vec2f() + bh_gravity;
    // v = v + dt * a;
    // x += dt * v;

    let params = rk4_integrate(x, v);
    x = params.position;
    v = params.velocity;

    if distance < 100 {
        f &= ~(FLAG_PHYSICAL | FLAG_DRAW_OBJECT);
        v = vec2f();
    }

    flags[i].inner = f;
    velocities[i].inner = v;
    aabbs[i] = AABB(x - size / 2, x + size / 2);
}

struct IntegratedParameters {
    position: vec2f,
    velocity: vec2f
}

fn rk4_integrate(position: vec2f, velocity: vec2f) -> IntegratedParameters {
    // Treat position and velocity as state vector
    let state = vec4f(position, velocity);
    // RK4 integration
    let k1 = f(state);
    let k2 = f(state + 0.5 * dt * k1);
    let k3 = f(state + 0.5 * dt * k2);
    let k4 = f(state + dt * k3);
    let new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    return IntegratedParameters(new_state.xy, new_state.zw);
}

fn f(state: vec4f) -> vec4f {
    // state.xy = position, state.zw = velocity
    let acceleration = blackhole_gravity(state.xy);
    return vec4f(state.zw, acceleration);
}

fn blackhole_gravity(position: vec2f) -> vec2f {
    let to_blackhole = BLACKHOLE_POSITION - position;
    let direction = normalize(to_blackhole);
    let distance = length(to_blackhole);
    let bh_gravity = direction * BLACKHOLE_MASS / (distance * distance);
    return bh_gravity;
}
