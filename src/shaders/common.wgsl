#define_import_path common

const FLAG_SHOW: u32 = 1 << 0;
const FLAG_PHYSICAL: u32 = 1 << 1;

struct Velocity {
    inner: vec2f
}

struct Mass {
    inner: f32
}

struct Flags {
    inner: u32
}

struct Position {
    inner: vec2f,
}

struct Size {
    inner: vec2f
}

struct Color {
    inner: vec4f
}

struct Shape {
    inner: u32
}
