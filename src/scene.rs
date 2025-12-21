use itertools::Itertools as _;
use nalgebra::Vector2;
use rand::random;
use winit::dpi::PhysicalSize;

use crate::{
    objects::{ObjectPrototype, Objects},
    shaders::{
        common::{FLAG_PHYSICAL, FLAG_SHOW},
        shape::{SHAPE_CIRCLE, SHAPE_RECT},
    },
};

pub fn create_scene(window_size: PhysicalSize<u32>, objects: &mut Objects) {
    let window_size = Vector2::new(window_size.width as f32, window_size.height as f32);
    let circles = {
        const RADIUS: f32 = 0.3;
        // const VELOCITY_MAX: f32 = 0.01;
        let shape_count: Vector2<usize> = (window_size / (RADIUS * 2.0)).try_cast().unwrap();
        (0..shape_count.x).cartesian_product(0..shape_count.y).map(move |(i, j)| {
            let (i, j) = (i as f32, j as f32);
            let position = [RADIUS * (i * 2.0 + 1.0), RADIUS * (j * 2.0 + 1.0)];
            ObjectPrototype {
                flags: FLAG_SHOW | FLAG_PHYSICAL,
                position,
                velocity: [
                    // random_range(-VELOCITY_MAX..VELOCITY_MAX),
                    // random_range(-VELOCITY_MAX..VELOCITY_MAX),
                    0.0, 0.0,
                ],
                mass: 1.0,
                size: [RADIUS * 2.0, RADIUS * 2.0],
                color: [random(), random(), random(), 1.0],
                shape: SHAPE_CIRCLE,
            }
        })
    };

    const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    let top = ObjectPrototype {
        flags: FLAG_SHOW,
        position: [window_size.x / 2.0, 0.5],
        velocity: [0.0, 0.0],
        mass: f32::INFINITY,
        size: [window_size.x, 1.0],
        color: RED,
        shape: SHAPE_RECT,
    };
    let bottom = ObjectPrototype {
        flags: FLAG_SHOW,
        position: [window_size.x / 2.0, window_size.y - 0.5],
        velocity: [0.0, 0.0],
        mass: f32::INFINITY,
        size: [window_size.x, 1.0],
        color: RED,
        shape: SHAPE_RECT,
    };
    let left = ObjectPrototype {
        flags: FLAG_SHOW,
        velocity: [0.0, 0.0],
        position: [0.5, window_size.y / 2.0],
        mass: f32::INFINITY,
        size: [1.0, window_size.y],
        color: RED,
        shape: SHAPE_RECT,
    };
    let right = ObjectPrototype {
        flags: FLAG_SHOW,
        position: [window_size.x - 0.5, window_size.y / 2.0],
        velocity: [0.0, 0.0],
        mass: f32::INFINITY,
        size: [1.0, window_size.y],
        color: RED,
        shape: SHAPE_RECT,
    };
    objects.extend(circles);
    objects.push(top);
    objects.push(bottom);
    objects.push(left);
    objects.push(right);
}
