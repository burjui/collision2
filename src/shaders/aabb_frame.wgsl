// Needs to be a separate shader to render all BVH AABBs

#import common::{ FLAG_DRAW_AABB, Camera, Flags, AABB }

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> flags: array<Flags>;
@group(0) @binding(2) var<storage, read> aabbs: array<AABB>;

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) flags: u32,
    @location(1) scale: f32,
    @location(2) quad_position: vec2f,
}

const UNIT_QUAD_VERTICES = array<vec2f, 5>(
    vec2f(0.5, 0.5),
    vec2f(-0.5, 0.5),
    vec2f(-0.5, -0.5),
    vec2f(0.5, -0.5),
    vec2f(0.5, 0.5),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) i: u32,
) -> VertexOutput {
    var out = VertexOutput();

    var flags_: u32 = FLAG_DRAW_AABB;
    if i < arrayLength(&flags) {
        flags_ = flags[i].inner;
        if (flags_ & FLAG_DRAW_AABB) == 0 {
            return out;
        }
    }

    let aabb = aabbs[i];
    let scale = (aabb.max - aabb.min);
    let center = ((aabb.min + aabb.max) / vec2(2f));
    let model = mat4x4f(
        scale.x, 0, 0, 0,
        0, scale.y, 0, 0,
        0, 0, 1, 0,
        center.x, center.y, 0, 1,
    );
    let vertex = UNIT_QUAD_VERTICES[vertex_index];
    out.clip_position = camera.inner * model * vec4f(vertex, 0, 1);
    out.flags = flags_;
    out.scale = max(scale.x, scale.y);
    out.quad_position = vertex;

    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    if (in.flags & FLAG_DRAW_AABB) == 0 {
        discard;
    }

    return FragmentOutput(vec4f(0.5, 0.5, 0.5, 1.0));
}
