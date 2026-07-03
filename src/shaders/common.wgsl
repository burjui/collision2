#define_import_path common

struct Camera {
    inner: mat4x4f
}

struct Flags {
    inner: u32
}

struct Velocity {
    inner: vec2f
}

struct Mass {
    inner: f32
}

struct Color {
    inner: vec4f
}

struct Shape {
    inner: u32
}

struct AABB {
    min: vec2f,
    max: vec2f
}

struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
}

struct CollisionCandidate {
    a: u32,
    b: u32,
}

struct CellPosition {
    cell: vec2u,
    offset: u32
}

const WORKGROUP_SIZE: u32 = 64;

const UNIT_QUAD_VERTICES = array<vec2f, 6>(
    vec2f(0.5, 0.5),
    vec2f(-0.5, 0.5),
    vec2f(-0.5, -0.5),
    vec2f(-0.5, -0.5),
    vec2f(0.5, -0.5),
    vec2f(0.5, 0.5),
);

const FLAG_DRAW_OBJECT: u32 = 1 << 0;
const FLAG_DRAW_AABB: u32 = 1 << 1;
const FLAG_PHYSICAL: u32 = 1 << 2;

const MAX_CANDIDATES_PER_OBJECT: u32 = 16;
const MAX_OBJECTS_PER_CELL: u32 = 4;

const MAX_DISPATCH_DIMENSION: u32 = 65535;

fn flat_invocation_index(gid: vec3u, nwg: vec3u, workgroup_size: u32) -> u32 {
    return gid.x +
          (gid.y * workgroup_size * nwg.x) +
          (gid.z * workgroup_size * nwg.x * workgroup_size * nwg.y);
}