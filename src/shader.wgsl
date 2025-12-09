struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) quad_position: vec2f,
};

struct FragmentOutput {
    @location(0) color: vec4f
}

const QUAD_VERTICES_LENGTH: u32 = 6;
const QUAD = array<vec2<f32>,6>(
    vec2f(1.0, 1.0),
    vec2f(-1.0, 1.0),
    vec2f(-1.0, -1.0),
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0)
);

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4f(QUAD[in.vertex_index % QUAD_VERTICES_LENGTH], 0.0, 1.1);
    out.quad_position = out.clip_position.xy;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var color = vec4f(1.0, 0.0, 0.0, 1.0);
    color.a = smoothstep(1.0, 0.99, length(in.quad_position.xy));
    return FragmentOutput(color);
}
