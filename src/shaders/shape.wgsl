#import common::{ UNIT_QUAD_VERTICES, FLAG_DRAW_OBJECT, Camera, Flags, AABB, Color, Shape, Velocity}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) quad_position: vec2f,
    @location(1) color: vec4f,
    @location(2) shape: u32
};

const SHAPE_RECT: u32 = 0;
const SHAPE_CIRCLE: u32 = 1;

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> size_factor: f32;
@group(0) @binding(2) var<storage, read> flags: array<Flags>;
@group(0) @binding(3) var<storage, read> aabbs: array<AABB>;
@group(0) @binding(4) var<storage, read> colors: array<Color>;
@group(0) @binding(5) var<storage, read> shapes: array<Shape>;
@group(0) @binding(6) var<storage, read> velocities: array<Velocity>;

const COLORING_SPEED_LIMIT: f32 = pow(80.0, 2.0);

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
    var scale = (aabb.max - aabb.min) * size_factor;
    var v = velocities[i].inner;
    let relative_speed = length(v) / COLORING_SPEED_LIMIT;
    out.color = velocity_to_color(v, sqrt(relative_speed));
    scale *= sqrt(sqrt(relative_speed)) * 1.5;
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
        color.a *= smoothstep(w, -w, d);
    }
    return FragmentOutput(color);
}

fn sdf_cirle(p: vec2f) -> f32 {
    return length(p) - 0.5;
}

fn velocity_to_color(velocity: vec2f, relative_speed: f32) -> vec4f {
    let t = clamp(relative_speed, 0.0, 1.0);
    let lambda = mix(700.0, 380.0, t);
    let rgb = wavelength_to_rgb(lambda);
    let intensity = spectral_intensity(lambda);
    return vec4f(rgb * intensity, 0.1);
}

fn spectral_intensity(lambda: f32) -> f32 {
    if (lambda < 420.0) {
        return 0.3 + 0.7 * (lambda - 380.0) / (420.0 - 380.0);
    }
    if (lambda > 645.0) {
        return 0.3 + 0.7 * (700.0 - lambda) / (700.0 - 645.0);
    }
    return 1.0;
}

fn wavelength_to_rgb(lambda: f32) -> vec3f {
    var r: f32 = 0.0;
    var g: f32 = 0.0;
    var b: f32 = 0.0;

    if (lambda >= 380.0 && lambda < 440.0) {
        r = -(lambda - 440.0) / (440.0 - 380.0);
        b = 1.0;
    } else if (lambda < 490.0) {
        g = (lambda - 440.0) / (490.0 - 440.0);
        b = 1.0;
    } else if (lambda < 510.0) {
        g = 1.0;
        b = -(lambda - 510.0) / (510.0 - 490.0);
    } else if (lambda < 580.0) {
        r = (lambda - 510.0) / (580.0 - 510.0);
        g = 1.0;
    } else if (lambda < 645.0) {
        r = 1.0;
        g = -(lambda - 645.0) / (645.0 - 580.0);
    } else if (lambda <= 700.0) {
        r = 1.0;
    }

    return vec3f(r, g, b);
}
