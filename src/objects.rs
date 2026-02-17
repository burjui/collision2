use color::{AlphaColor, Srgb};
use nalgebra::Vector2;

use crate::shaders::common::{AABB, Color, Flags, Mass, Shape, Velocity};

pub struct ObjectPrototype {
    pub flags: u32,
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub mass: f32,
    pub size: [f32; 2],
    pub color: AlphaColor<Srgb>,
    pub shape: u32,
}

#[derive(Default)]
pub struct Objects {
    pub flags: Vec<Flags>,
    pub aabbs: Vec<AABB>,
    pub velocities: Vec<Velocity>,
    pub masses: Vec<Mass>,
    pub colors: Vec<Color>,
    pub shapes: Vec<Shape>,
}

impl Objects {
    pub fn push(&mut self, prototype: ObjectPrototype) {
        self.flags.push(Flags::new(prototype.flags));
        let position = Vector2::from(prototype.position);
        let size = Vector2::from(prototype.size);
        self.aabbs.push(AABB::new((position - size / 2.0).into(), (position + size / 2.0).into()));
        self.velocities.push(Velocity::new(prototype.velocity));
        self.masses.push(Mass::new(prototype.mass));
        self.colors.push(Color::new(prototype.color.components));
        self.shapes.push(Shape::new(prototype.shape));
    }

    pub fn extend(&mut self, iter: impl IntoIterator<Item = ObjectPrototype>) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for prototype in iter {
            self.push(prototype);
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        self.flags.reserve(additional);
        self.aabbs.reserve(additional);
        self.velocities.reserve(additional);
        self.masses.reserve(additional);
        self.colors.reserve(additional);
        self.shapes.reserve(additional);
    }
}
