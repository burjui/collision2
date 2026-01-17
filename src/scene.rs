use std::ops::{Range, RangeInclusive};

use color::{AlphaColor, palette::css};
use itertools::Itertools as _;
use nalgebra::Vector2;
use rand::random_range;

use crate::{
    objects::{ObjectPrototype, Objects},
    shaders::{
        common::{AABB, FLAG_DRAW_AABB, FLAG_DRAW_OBJECT, FLAG_PHYSICAL},
        shape::{SHAPE_CIRCLE, SHAPE_RECT},
    },
};

pub fn create_scene(objects: &mut Objects, world_aabb: AABB) {
    let world_size = world_aabb.size();

    println!("World size: {}x{}", world_size.x, world_size.y);

    let circles = {
        const RADIUS: f32 = 2.0;
        const MARGIN: f32 = -0.7;
        const POSITION_RAND_FACTOR: f32 = 1.0;
        const VELOCITY_RAND_MAX: f32 = 300.0;
        const VELOCITY_RAND_RANGE_X: RangeInclusive<f32> = -VELOCITY_RAND_MAX / 3.0..=VELOCITY_RAND_MAX;
        const VELOCITY_RAND_RANGE_Y: RangeInclusive<f32> = -VELOCITY_RAND_MAX..=VELOCITY_RAND_MAX;
        const _COLOR_RAND_RANGE: Range<f32> = 0.8..1.0;
        const EFFECTIVE_RADIUS: f32 = RADIUS + MARGIN;
        let shape_count_f32 = world_size / (EFFECTIVE_RADIUS * 2.0);
        let shape_count: Vector2<usize> = shape_count_f32.try_cast().unwrap();
        (0..shape_count.x).cartesian_product(0..shape_count.y).map(move |(i, j)| {
            let (i, j) = (i as f32, j as f32);
            let range = -RADIUS * POSITION_RAND_FACTOR..=RADIUS * POSITION_RAND_FACTOR;
            let position = world_aabb.min()
                + Vector2::new(EFFECTIVE_RADIUS * (i * 2.0 + 1.0), EFFECTIVE_RADIUS * (j * 2.0 + 1.0))
                + Vector2::new(random_range(range.clone()), random_range(range));
            ObjectPrototype {
                flags: FLAG_DRAW_OBJECT | FLAG_DRAW_AABB | FLAG_PHYSICAL,
                position: position.into(),
                velocity: [random_range(VELOCITY_RAND_RANGE_X), random_range(VELOCITY_RAND_RANGE_Y)],
                mass: 2.0,
                size: [RADIUS * 2.0, RADIUS * 2.0],
                color: AlphaColor::new([
                    0.4 + 0.6 * i / (shape_count_f32.x - 1.0),
                    0.8 * j / (shape_count_f32.x - 1.0),
                    0.3 * i / (shape_count_f32.x - 1.0) * j / (shape_count_f32.y - 1.0),
                    1.0,
                ]),
                shape: SHAPE_CIRCLE,
            }
        })
    };

    let border_thickness = world_aabb.size().y / 400.0;
    let top = ObjectPrototype {
        flags: FLAG_DRAW_OBJECT,
        position: [0.0, world_aabb.max().y - border_thickness / 2.0],
        velocity: [0.0, 0.0],
        mass: f32::INFINITY,
        size: [world_size.x, border_thickness],
        color: css::RED,
        shape: SHAPE_RECT,
    };
    let bottom = ObjectPrototype {
        flags: FLAG_DRAW_OBJECT,
        position: [0.0, world_aabb.min().y + border_thickness / 2.0],
        velocity: [0.0, 0.0],
        mass: f32::INFINITY,
        size: [world_size.x, border_thickness],
        color: css::RED,
        shape: SHAPE_RECT,
    };
    let left = ObjectPrototype {
        flags: FLAG_DRAW_OBJECT,
        velocity: [0.0, 0.0],
        position: [world_aabb.min().x + border_thickness / 2.0, 0.0],
        mass: f32::INFINITY,
        size: [border_thickness, world_size.y],
        color: css::RED,
        shape: SHAPE_RECT,
    };
    let right = ObjectPrototype {
        flags: FLAG_DRAW_OBJECT,
        position: [world_aabb.max().x - border_thickness / 2.0, 0.0],
        velocity: [0.0, 0.0],
        mass: f32::INFINITY,
        size: [border_thickness, world_size.y],
        color: css::RED,
        shape: SHAPE_RECT,
    };

    objects.extend(circles);
    objects.push(top);
    objects.push(bottom);
    objects.push(left);
    objects.push(right);
}
