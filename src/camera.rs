use nalgebra::Vector2;
use winit::dpi::PhysicalSize;

pub fn orthographic_camera(
    view_size: PhysicalSize<f32>,
    world_height: f32,
    zoom: f32,
    offset: Vector2<f32>,
) -> [[f32; 4]; 4] {
    let aspect = view_size.width / view_size.height;
    let world_width = world_height * aspect;
    let left = -world_width * 0.5;
    let right = world_width * 0.5;
    let bottom = -world_height * 0.5;
    let top = world_height * 0.5;
    let sx = zoom * 2.0 / (right - left);
    let sy = zoom * 2.0 / (top - bottom);
    let tx = -offset.x * 2.0 / (right - left);
    let ty = offset.y * 2.0 / (top - bottom);
    [
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [tx, ty, 0.0, 1.0],
    ]
}
