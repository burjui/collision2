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
}

const BLACKHOLE_COUNT: u32 = 5;
const BLACKHOLES = array<BlackHole, BLACKHOLE_COUNT>(
    BlackHole(vec2f(-200, 500), 1, 2),
    BlackHole(vec2f(200, -500), 1, 10),
    BlackHole(vec2f(-300, -200), 2, 200),
    BlackHole(vec2f(500, 200), 1, 2),
    BlackHole(vec2f(-600, -300), 1, 10),
);
const BLACKHOLE_MASS_SCALE: f32 = 50000000;
const BLACKHOLE_SIZE_SCALE: f32 = 3;
const BLACKHOLE_DESTROY_MATTER: bool = true;

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
    if BLACKHOLE_DESTROY_MATTER {
        for (var bh_index: u32 = 0; bh_index < BLACKHOLE_COUNT && (f & FLAG_PHYSICAL) != 0; bh_index++) {
            let blackhole = BLACKHOLES[bh_index];
            let distance = length(blackhole.position - params.x) - max(size.x, size.y) / 2;
            if distance < blackhole.radius * BLACKHOLE_SIZE_SCALE {
                f &= ~(FLAG_PHYSICAL | FLAG_DRAW_OBJECT | FLAG_DRAW_AABB);
                params.v = vec2f();
            }
        }
    }

    flags[i].inner = f;
    velocities[i].inner = params.v;
    let offset = params.x - start_x;
    aabbs[i] = AABB(aabb.min + offset, aabb.max + offset);
}

fn forces(position: vec2f) -> vec2f {
    var acc = vec2f();
    for (var bh_index: u32 = 0; bh_index < BLACKHOLE_COUNT; bh_index += 1) {
        acc += blackhole_gravity(BLACKHOLES[bh_index], position);
    }
    return acc;
}

fn blackhole_gravity(blackhole: BlackHole, position: vec2f) -> vec2f {
    let to_blackhole = blackhole.position - position;
    let direction = normalize(to_blackhole);
    let distance = length(to_blackhole);
    let bh_gravity = direction * blackhole.mass * BLACKHOLE_MASS_SCALE / (distance * distance);
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
