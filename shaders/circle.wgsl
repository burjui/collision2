struct Uniforms {
    transform: mat4x4f,
}

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) position: vec2f
};

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) quad_position: vec2f,
};

struct FragmentOutput {
    @location(0) color: vec4f
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.transform * vec4f(in.position, 0.0, 1.1);
    out.quad_position = vec4f(in.position, 0.0, 1.1).xy;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var color = vec4f(1.0, 0.0, 0.0, 1.0);
    color.a = smoothstep(1.0, 0.99, length(in.quad_position));
    return FragmentOutput(color);
}
