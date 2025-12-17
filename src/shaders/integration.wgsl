#import common::FLAG_SHOW

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

@compute
@workgroup_size(64)
fn cs_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let index = gid.x;
    if index >= arrayLength(&mass) {
        return;
    }
    const blackhole_position = vec2f(800, 400);
    let to_blackhole = blackhole_position - position[index].inner;
    let direction = normalize(to_blackhole);
    let distance = length(to_blackhole);
    let gravity = direction * mass[index].inner * 10000000 / (distance * distance);
    let v = velocity[index].inner;
    velocity[index].inner = v + dt * gravity - v * dt * 0.5;
    position[index].inner += dt * velocity[index].inner;

    if distance < 100 {
        flags[index].inner &= ~FLAG_SHOW;
    }
}
