#import common::{FLAG_DRAW_OBJECT, FLAG_PHYSICAL, AABB, Mass, Velocity, Position, Flags, BvhNode, invocation_index}

@group(0) @binding(0) var<uniform> dt: f32;
@group(0) @binding(1) var<storage, read_write> flags: array<Flags>;
@group(0) @binding(2) var<storage, read> masses: array<Mass>;
@group(0) @binding(3) var<storage, read_write> velocities: array<Velocity>;
@group(0) @binding(4) var<storage, read_write> aabbs: array<AABB>;
@group(0) @binding(5) var<storage, read> nodes: array<BvhNode>;
@group(0) @binding(6) var<storage, read_write> force_acc: array<vec2f>;

const WORKGROUP_SIZE: u32 = 64;

const BLACKHOLE_POSITION = vec2f();
const BLACKHOLE_MASS = 800000000.0;

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
    let start_x = (aabb.min + aabb.max) / 2;
    var params = IntegratedParameters(start_x, velocities[i].inner);
    params = integrate_yoshida(params);

    let size = aabb.max - aabb.min;
    let distance = length(BLACKHOLE_POSITION - params.x) - max(size.x, size.y) / 2;
    if distance < 100 {
        f &= ~(FLAG_PHYSICAL | FLAG_DRAW_OBJECT);
        params.v = vec2f();
    }

    flags[i].inner = f;
    velocities[i].inner = params.v;
    let offset = params.x - start_x;
    aabbs[i] = AABB(aabb.min + offset, aabb.max + offset);
}

fn forces(position: vec2f) -> vec2f {
    return blackhole_gravity(position);
}

fn blackhole_gravity(position: vec2f) -> vec2f {
    let to_blackhole = BLACKHOLE_POSITION - position;
    let direction = normalize(to_blackhole);
    let distance = length(to_blackhole);
    let bh_gravity = direction * BLACKHOLE_MASS / (distance * distance);
    return bh_gravity;
}


struct IntegratedParameters {
    x: vec2f,
    v: vec2f
}

fn integrate_yoshida(params: IntegratedParameters) -> IntegratedParameters {
    const w1 = 1.3512071919596578;
    const w0 = -1.7024143839193155; // = 1 - 2*w1
    var p = params;
    p = leapfrog_step(p, w1);
    p = leapfrog_step(p, w0);
    p = leapfrog_step(p, w1);
    return p;
}

// A single leapfrog (drift-kick-drift) step
fn leapfrog_step(params: IntegratedParameters, w: f32) -> IntegratedParameters {
    let half_step = w * dt * 0.5;
    // Drift (position half-step)
    var x = params.x + params.v * half_step;
    // Kick (full velocity step)
    let a = forces(x);
    let v = params.v + a * (w * dt);
    // Drift (position half-step)
    x = x + v * half_step;
    return IntegratedParameters(x, v);
}
