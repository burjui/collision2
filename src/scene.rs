use color::AlphaColor;
use image::{Rgba, imageops::FilterType};
use nalgebra::Vector2;
use rand::random_range;

use crate::{
    config::AppConfig,
    objects::{ObjectPrototype, Objects},
    shaders::common::{AABB, FLAG_COLLIDE, FLAG_DRAW_AABB, FLAG_DRAW_OBJECT, FLAG_INTEGRATE, FLAG_VELOCITY_COLOR},
};

pub fn create_scene(app_config: &AppConfig, objects: &mut Objects, world_aabb: AABB) {
    let world_center = world_aabb.min() + world_aabb.size() / 2.0;
    let scene_aabb = AABB {
        min: (world_center - (world_aabb.size() * app_config.scene_scale) / 2.0).into(),
        max: (world_center + (world_aabb.size() * app_config.scene_scale) / 2.0).into(),
    };
    let effective_radius: f32 = app_config.particle_radius + app_config.particle_padding;
    let shape_count_f32 = scene_aabb.size() / (effective_radius * 2.0);
    let shape_count: Vector2<u32> = shape_count_f32.try_cast().unwrap();
    let mut image = None;
    if !app_config.image.is_empty() {
        image = Some({
            let image = image::open(&app_config.image).unwrap();
            println!("Image size: {}x{}", image.width(), image.height());
            image.resize_exact(shape_count.x, shape_count.y, FilterType::Gaussian).into_rgba8()
        })
    }
    objects.reserve((shape_count.x * shape_count.y).try_into().unwrap());

    for y in 0..shape_count.y {
        for x in 0..shape_count.x {
            let (i, j) = (x as f32, y as f32);
            let postition_randomization_range = -app_config.particle_radius * app_config.particle_position_rand
                ..=app_config.particle_radius * app_config.particle_position_rand;
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
            let mut velocity_color = FLAG_VELOCITY_COLOR;
            if !app_config.image.is_empty() {
                velocity_color = 0;
            }
            let offset = Vector2::from(app_config.scene_offset());
            objects.push(ObjectPrototype {
                flags: FLAG_DRAW_OBJECT | FLAG_DRAW_AABB | FLAG_INTEGRATE | FLAG_COLLIDE | velocity_color,
                position: (position + offset).into(),
                velocity: app_config.kick(),
                mass: app_config.particle_mass,
                size: [app_config.particle_radius * 2.0, app_config.particle_radius * 2.0],
                color,
                shape: app_config.particle_shape as u32,
            });
        }
    }
}
