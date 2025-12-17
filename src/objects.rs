use crate::shaders::{
    integration::{ComputeMass, ComputeVelocity},
    shape::{ColorInput, FlagsInput, PositionInput, ShapeInput, SizeInput},
};

pub struct ObjectPrototype {
    pub mass: f32,
    pub velocity: [f32; 2],
    pub flags: u32,
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub color: [f32; 4],
    pub shape: u32,
}

pub struct Objects {
    pub mass: Vec<ComputeMass>,
    pub velocity: Vec<ComputeVelocity>,
    pub flags: Vec<FlagsInput>,
    pub position: Vec<PositionInput>,
    pub size: Vec<SizeInput>,
    pub color: Vec<ColorInput>,
    pub shape: Vec<ShapeInput>,
}

impl Objects {
    pub fn new() -> Self {
        Self {
            mass: vec![],
            velocity: vec![],
            flags: vec![],
            position: vec![],
            size: vec![],
            color: vec![],
            shape: vec![],
        }
    }

    pub fn push(&mut self, prototype: ObjectPrototype) {
        self.mass.push(ComputeMass::new(prototype.mass));
        self.velocity.push(ComputeVelocity::new(prototype.velocity));
        self.flags.push(FlagsInput::new(prototype.flags));
        self.position.push(PositionInput::new(prototype.position));
        self.size.push(SizeInput::new(prototype.size));
        self.color.push(ColorInput::new(prototype.color));
        self.shape.push(ShapeInput::new(prototype.shape));
    }

    pub fn extend(&mut self, iter: impl IntoIterator<Item = ObjectPrototype>) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for prototype in iter {
            self.push(prototype);
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        self.mass.reserve(additional);
        self.velocity.reserve(additional);
        self.flags.reserve(additional);
        self.position.reserve(additional);
        self.size.reserve(additional);
        self.color.reserve(additional);
        self.shape.reserve(additional);
    }

    pub fn len(&self) -> usize {
        self.flags.len()
    }
}
