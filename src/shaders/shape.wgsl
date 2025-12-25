#import common::{FLAG_SHOW, Flags, AABB, Color, Shape}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) scaling_factor: f32,
    @location(1) quad_position: vec2f,
    @location(2) color: vec4f,
    @location(3) shape: u32
};

struct FragmentOutput {
    @location(0) color: vec4f
}

const SHAPE_RECT: u32 = 0;
const SHAPE_CIRCLE: u32 = 1;

@group(0) @binding(0) var<uniform> view_size: vec2f;
@group(0) @binding(1) var<storage, read> flags: array<Flags>;
@group(0) @binding(2) var<storage, read> aabbs: array<AABB>;
@group(0) @binding(4) var<storage, read> colors: array<Color>;
@group(0) @binding(5) var<storage, read> shapes: array<Shape>;

const VERTICES = array<vec2f, 6>(
    vec2f(1.0, 1.0),
    vec2f(-1.0, 1.0),
    vec2f(-1.0, -1.0),
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) i: u32,
) -> VertexOutput {
    var out: VertexOutput;
    if (flags[i].inner & FLAG_SHOW) == 0 {
        return out;
    }
    let aabb = aabbs[i];
    let size = aabb.max - aabb.min;
    let scale = size / view_size;
    out.scaling_factor = clamp(min(scale.x, scale.y), 0, 1);
    let center = (aabb.min + aabb.max) / 2;
    let translation = (center / view_size * vec2f(2, -2) + vec2(-1, 1)) / scale;
    let translation_matrix = transpose(mat4x4f(
        1, 0, 0, translation.x,
        0, 1, 0, translation.y,
        0, 0, 1, 0,
        0, 0, 0, 1
    ));
    let scale_matrix = mat4x4f(
        scale.x, 0, 0, 0,
        0, scale.y, 0, 0,
        0, 0,       1, 0,
        0, 0,       0, 1
    );
    let vertex = VERTICES[vertex_index];
    out.clip_position = scale_matrix * translation_matrix * vec4f(vertex, 0, 1);
    out.quad_position = vertex;
    out.color = colors[i].inner;
    out.shape = shapes[i].inner;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var color = in.color;
    if in.shape == SHAPE_CIRCLE {
        color.a = smoothstep(1.0, 1 - clamp(0.002 / in.scaling_factor, 0.002, 0.3), length(in.quad_position));
    }
    return FragmentOutput(color);
}
