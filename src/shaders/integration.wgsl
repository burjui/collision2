#import common::{FLAG_SHOW, FLAG_PHYSICAL}

struct ComputeMass {
    inner: f32
}

struct ComputeVelocity {
    inner: vec2f
}

struct ComputePosition {
    inner: vec2f
}

struct ComputeFlags {
    inner: u32
}

@group(0) @binding(0) var<storage, read> dt: f32;
@group(0) @binding(1) var<storage, read> mass: array<ComputeMass>;
@group(0) @binding(2) var<storage, read_write> velocity: array<ComputeVelocity>;
@group(0) @binding(3) var<storage, read_write> position: array<ComputePosition>;
@group(0) @binding(4) var<storage, read_write> flags: array<ComputeFlags>;

const WORKGROUP_SIZE: u32 = 64;

@compute
@workgroup_size(WORKGROUP_SIZE)
fn cs_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let index = gid.x + gid.y * 65536 * WORKGROUP_SIZE;
    if index >= arrayLength(&mass) || (flags[index].inner & FLAG_PHYSICAL) == 0 {
        return;
    }
    const blackhole_position = vec2f(800, 400);
    let to_blackhole = blackhole_position - position[index].inner;
    let direction = normalize(to_blackhole);
    let distance = length(to_blackhole);
    var v = velocity[index].inner;
    var x = position[index].inner;
    let a = direction * 10000000 / (distance * distance);
    v = v + dt * a;// - v * dt * 0.5;
    x += dt * v;

    if distance < 100 {
        flags[index].inner &= ~(FLAG_PHYSICAL | FLAG_SHOW);
        v = vec2f();
    }
    velocity[index].inner = v;
    position[index].inner = x;
}
