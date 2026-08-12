use color::{AlphaColor, Srgb, palette::css};
use image::{Rgba, imageops::FilterType};
use itertools::Itertools as _;
use nalgebra::Vector2;
use rand::random_range;

use crate::{
    config::CONFIG,
    objects::{ObjectPrototype, Objects},
    shaders::{
        common::{AABB, FLAG_COLLISION, FLAG_DRAW_AABB, FLAG_DRAW_OBJECT, FLAG_PHYSICAL, FLAG_VELOCITY_COLOR},
        render_shape::SHAPE_RECT,
    },
};

pub fn create_scene(objects: &mut Objects, world_aabb: AABB) {
    let world_center = world_aabb.min() + world_aabb.size() / 2.0;
    let scene_aabb = AABB {
        min: (world_center - (world_aabb.size() * CONFIG.scene_scale) / 2.0).into(),
        max: (world_center + (world_aabb.size() * CONFIG.scene_scale) / 2.0).into(),
    };
    let particles = {
        let effective_radius: f32 = CONFIG.particle_radius + CONFIG.particle_padding;
        let shape_count_f32 = scene_aabb.size() / (effective_radius * 2.0);
        let shape_count: Vector2<u32> = shape_count_f32.try_cast().unwrap();
        let image = CONFIG.image.clone().map(|path| {
            let image = image::open(path).unwrap();
            println!("Image size: {}x{}", image.width(), image.height());
            image.resize_exact(shape_count.x, shape_count.y, FilterType::Gaussian).into_rgba8()
        });

        (0..shape_count.x).cartesian_product(0..shape_count.y).map(move |(x, y)| {
            let (i, j) = (x as f32, y as f32);
            let postition_randomization_range = -CONFIG.particle_radius * CONFIG.particle_position_rand
                ..=CONFIG.particle_radius * CONFIG.particle_position_rand;
            let position = scene_aabb.min()
                + Vector2::new(effective_radius * (i * 2.0 + 1.0), effective_radius * (j * 2.0 + 1.0))
                + Vector2::new(
                    random_range(postition_randomization_range.clone()),
                    random_range(postition_randomization_range),
                );
            let color = if let Some(image) = &image {
                let Rgba(color) = image.get_pixel(x, shape_count.y - 1 - y);
                AlphaColor::new(color.map(|component| component as f32 / 255.0))
            } else {
                AlphaColor::new([
                    0.4 + 0.6 * i / (shape_count_f32.x - 1.0),
                    0.8 * j / (shape_count_f32.x - 1.0),
                    0.3 * i / (shape_count_f32.x - 1.0) * j / (shape_count_f32.y - 1.0),
                    1.0,
                ])
            };
            let velocity_color = if CONFIG.image.is_some() { 0 } else { FLAG_VELOCITY_COLOR };
            let offset = Vector2::from(CONFIG.scene_offset());
            ObjectPrototype {
                flags: FLAG_DRAW_OBJECT | FLAG_DRAW_AABB | FLAG_PHYSICAL | FLAG_COLLISION | velocity_color,
                position: (position + offset).into(),
                velocity: CONFIG.kick(),
                mass: CONFIG.particle_mass,
                size: [CONFIG.particle_radius * 2.0, CONFIG.particle_radius * 2.0],
                color,
                shape: CONFIG.particle_shape as u32,
            }
        })
    };
    objects.extend(particles);

    let _borders = world_borders(world_aabb);
    // for border in _borders {
    //     objects.push(border);
    // }
}

fn world_borders(world_aabb: AABB) -> Vec<ObjectPrototype> {
    const FLAGS: u32 = FLAG_DRAW_OBJECT;
    const MASS: f32 = 10000.0;
    const COLOR: AlphaColor<Srgb> = css::RED;

    let world_size = world_aabb.size();
    let border_thickness = world_size.y / 1000.0;
    let top = ObjectPrototype {
        flags: FLAGS,
        position: [
            world_size.x / 2.0,
            world_aabb.max().y - border_thickness / 2.0 + world_size.y / 2.0,
        ],
        velocity: [0.0, 0.0],
        mass: MASS,
        size: [world_size.x, border_thickness],
        color: COLOR,
        shape: SHAPE_RECT,
    };
    let bottom = ObjectPrototype {
        flags: FLAGS,
        position: [0.0, world_aabb.min().y + border_thickness / 2.0 - world_size.y / 2.0],
        velocity: [0.0, 0.0],
        mass: MASS,
        size: [world_size.x, border_thickness],
        color: COLOR,
        shape: SHAPE_RECT,
    };
    let left = ObjectPrototype {
        flags: FLAGS,
        velocity: [0.0, 0.0],
        position: [world_aabb.min().x + border_thickness / 2.0 - world_size.x / 2.0, 0.0],
        mass: MASS,
        size: [border_thickness, world_size.y],
        color: COLOR,
        shape: SHAPE_RECT,
    };
    let right = ObjectPrototype {
        flags: FLAGS,
        position: [world_aabb.max().x - border_thickness / 2.0 + world_size.x / 2.0, 0.0],
        velocity: [0.0, 0.0],
        mass: MASS,
        size: [border_thickness, world_size.y],
        color: COLOR,
        shape: SHAPE_RECT,
    };
    vec![top, bottom, left, right]
}
