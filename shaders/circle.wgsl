struct Uniforms {
    view_size: vec2f
}

struct VertexInput {
    @location(0) position: vec2f
};

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) scaling_factor: f32,
    @location(1) quad_position: vec2f,
    @location(2) color: vec4f,
};

struct InstanceInput {
    @location(1) position: vec2f,
    @location(2) radius: f32,
    @location(3) color: vec4f
}

struct FragmentOutput {
    @location(0) color: vec4f
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    let scale = vec2f(instance.radius) * 2 / uniforms.view_size;
    out.scaling_factor = clamp(min(scale.x, scale.y), 0, 1);
    let translation = (instance.position / uniforms.view_size * vec2f(2, -2) + vec2(-1, 1)) / scale;
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
    out.clip_position = scale_matrix * translation_matrix * vec4f(vertex.position, 0, 1);
    out.quad_position = vertex.position;
    out.color = instance.color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var color = in.color;
    color.a = smoothstep(1.0, 1 - clamp(0.002 / in.scaling_factor, 0.002, 0.3), length(in.quad_position));
    return FragmentOutput(color);
}
