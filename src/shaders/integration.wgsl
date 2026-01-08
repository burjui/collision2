#import common::{FLAG_SHOW, FLAG_PHYSICAL, AABB, Mass, Velocity, Position, Flags, invocation_index}

@group(0) @binding(0) var<uniform> dt: f32;
@group(0) @binding(1) var<storage, read> masses: array<Mass>;
@group(0) @binding(2) var<storage, read_write> flags: array<Flags>;
@group(0) @binding(3) var<storage, read_write> aabbs: array<AABB>;
@group(0) @binding(4) var<storage, read_write> velocities: array<Velocity>;
@group(0) @binding(5) var<storage, read_write> processed: atomic<u32>;

const WORKGROUP_SIZE: u32 = 64;

@compute @workgroup_size(WORKGROUP_SIZE)
fn cs_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let i = invocation_index(gid, WORKGROUP_SIZE);
    if i >= arrayLength(&masses) {
        return;
    }

    atomicAdd(&processed, 1);

    var f = flags[i].inner;
    if (f & FLAG_PHYSICAL) == 0 {
        return;
    }

    const blackhole_position = vec2f(800, 400);

    let aabb = aabbs[i];
    var v = velocities[i].inner;
    var x = (aabb.min + aabb.max) / 2;
    let size = aabb.max - aabb.min;
    let to_blackhole = blackhole_position - x;
    let direction = normalize(to_blackhole);
    let distance = length(to_blackhole) - max(size.x, size.y) / 2;
    let bh_gravity = direction * 1000000 / (distance * distance);
    let a = vec2f();// + bh_gravity;
    v = v + dt * a;
    x += dt * v;

    // if distance < 100 {
    //     f &= ~(FLAG_PHYSICAL | FLAG_SHOW);
    //     v = vec2f();
    // }

    flags[i].inner = f;
    velocities[i].inner = v;
    aabbs[i] = AABB(x - size / 2, x + size / 2);
}
