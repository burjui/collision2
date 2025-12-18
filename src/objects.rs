use crate::shaders::common::{Color, Flags, Mass, Position, Shape, Size, Velocity};

pub struct ObjectPrototype {
    pub flags: u32,
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub mass: f32,
    pub size: [f32; 2],
    pub color: [f32; 4],
    pub shape: u32,
}

#[derive(Default)]
pub struct Objects {
    pub flags: Vec<Flags>,
    pub position: Vec<Position>,
    pub velocity: Vec<Velocity>,
    pub mass: Vec<Mass>,
    pub size: Vec<Size>,
    pub color: Vec<Color>,
    pub shape: Vec<Shape>,
}

impl Objects {
    pub fn push(&mut self, prototype: ObjectPrototype) {
        self.flags.push(Flags::new(prototype.flags));
        self.position.push(Position::new(prototype.position));
        self.velocity.push(Velocity::new(prototype.velocity));
        self.mass.push(Mass::new(prototype.mass));
        self.size.push(Size::new(prototype.size));
        self.color.push(Color::new(prototype.color));
        self.shape.push(Shape::new(prototype.shape));
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
        self.position.reserve(additional);
        self.velocity.reserve(additional);
        self.mass.reserve(additional);
        self.size.reserve(additional);
        self.color.reserve(additional);
        self.shape.reserve(additional);
    }

    pub fn len(&self) -> usize {
        self.flags.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
