struct ComputeMass {
    inner: f32
}

struct ComputeVelocity {
    inner: vec2f
}

struct ComputePosition {
    inner: vec2f
}

@group(0) @binding(0) var<storage, read> dt: f32;
@group(0) @binding(1) var<storage, read> mass: array<ComputeMass>;
@group(0) @binding(2) var<storage, read> velocity: array<ComputeVelocity>;
@group(0) @binding(3) var<storage, read_write> position: array<ComputePosition>;

@compute
@workgroup_size(64)
fn cs_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let index = gid.x;
    if index >= arrayLength(&mass) {
        return;
    }
    position[index].inner += velocity[index].inner * dt;
}
