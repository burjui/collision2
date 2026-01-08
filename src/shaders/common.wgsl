#define_import_path common

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

struct Camera {
    inner: mat4x4f
}

struct Velocity {
    inner: vec2f
}

struct Mass {
    inner: f32
}

struct Flags {
    inner: u32
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

const BVH_NODE_KIND_LEAF: u32 = 0;
const BVH_NODE_KIND_TREE: u32 = 0;

struct BvhNode {
    kind: u32,
    aabb: AABB,
    left: u32,
    right: u32
}

fn invocation_index(gid: vec3<u32>, workgroup_size: u32) -> u32 {
    return gid.x + gid.y * 65535 * workgroup_size;
}