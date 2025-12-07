struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    // @location(0) position: vec3f,
    // @location(1) color: vec4f,
    // @location(2) quad_pos : vec2f
};

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) color: vec4f,
    @location(1) quad_pos: vec2f,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var v: vec2f;
    switch in.vertex_index {
        case 0 { v = vec2f(1.0, 1.0); }
        case 1 { v = vec2f(-1.0, 1.0); }
        case 2 { v = vec2f(-1.0, -1.0); }
        case 3 { v = vec2f(-1.0, -1.0); }
        case 4 { v = vec2f(1.0, -1.0); }
        case 5 { v = vec2f(1.0, 1.0); }
        default { v = vec2f(0.0, 0.0); }
    }
    var out: VertexOutput;
    out.clip_position = vec4f(v, 0.0, 1.1);
    out.color = vec4f(1.0, 0.0, 0.0, 1.0);
    out.quad_pos = out.clip_position.xy;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    var color = in.color;
    color.a = max(0.0, smoothstep(0.0, 0.01, 1.0 - length(in.quad_pos.xy)));
    return color;
}
