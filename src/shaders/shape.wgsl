struct Uniforms {
    view_size: vec2f
}

struct VertexInput {
    @location(0) inner: vec2f
}

struct FlagsInput {
    @location(1) inner: u32
}

struct PositionInput {
    @location(2) inner: vec2f
}

struct SizeInput {
    @location(3) inner: vec2f
}

struct ColorInput {
    @location(4) inner: vec4f
}

struct ShapeInput {
    @location(5) inner: u32
}

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
const FLAG_SHOW: u32 = 1 << 0;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(
    vertex: VertexInput,
    flag: FlagsInput,
    position: PositionInput,
    size: SizeInput,
    color: ColorInput,
    shape: ShapeInput
) -> VertexOutput {
    var out: VertexOutput;
    if (flag.inner & FLAG_SHOW) == 0 {
        return out;
    }
    let scale = size.inner / uniforms.view_size;
    out.scaling_factor = clamp(min(scale.x, scale.y), 0, 1);
    let translation = (position.inner / uniforms.view_size * vec2f(2, -2) + vec2(-1, 1)) / scale;
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
    out.clip_position = scale_matrix * translation_matrix * vec4f(vertex.inner, 0, 1);
    out.quad_position = vertex.inner;
    out.color = color.inner;
    out.shape = shape.inner;

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
