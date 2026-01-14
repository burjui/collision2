#import common::{UNIT_QUAD_VERTICES, FLAG_DRAW_OBJECT, Camera, Flags, AABB, Color, Shape}

// TODO: point-based instancing

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) quad_position: vec2f,
    @location(1) color: vec4f,
    @location(2) shape: u32
};

const SHAPE_RECT: u32 = 0;
const SHAPE_CIRCLE: u32 = 1;

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> flags: array<Flags>;
@group(0) @binding(2) var<storage, read> aabbs: array<AABB>;
@group(0) @binding(4) var<storage, read> colors: array<Color>;
@group(0) @binding(5) var<storage, read> shapes: array<Shape>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) i: u32,
) -> VertexOutput {
    var out = VertexOutput();
    if (flags[i].inner & FLAG_DRAW_OBJECT) == 0 {
        return out;
    }

    let aabb = aabbs[i];
    let scale = aabb.max - aabb.min;
    let center = (aabb.min + aabb.max) / 2;
    let model = mat4x4f(
        scale.x, 0, 0, 0,
        0, scale.y, 0, 0,
        0, 0, 1, 0,
        center.x, center.y, 0, 1,
    );
    let vertex = UNIT_QUAD_VERTICES[vertex_index];
    out.clip_position = camera.inner * model * vec4f(vertex, 0, 1);
    out.quad_position = vertex;
    out.color = colors[i].inner;
    out.shape = shapes[i].inner;

    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var color = in.color;
    let d = sdf_cirle(in.quad_position);
    let w = fwidth(d) / 2; // fwidth has to be calculated before any branching
    if in.shape == SHAPE_CIRCLE {
        color.a = smoothstep(w, -w, d);
    }
    return FragmentOutput(color);
}

fn sdf_cirle(p: vec2f) -> f32 {
    return length(p) - 0.5;
}
