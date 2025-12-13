struct Uniforms {
    transform: mat4x4f,
    scaling: f32
}

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) position: vec2f
};

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) quad_position: vec2f,
    @location(1) color: vec4f
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
    out.clip_position = uniforms.transform * vec4f(vertex.position + instance.position, 0.0, 1.1);
    out.quad_position = vec4f(vertex.position, 0.0, 1.1).xy;
    out.color = instance.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var color = in.color;
    color.a = smoothstep(1.0, 1.0 - 0.01/uniforms.scaling, length(in.quad_position));
    return FragmentOutput(color);
}
