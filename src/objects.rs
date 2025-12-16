use crate::shape_shaders::shape;

pub struct ObjectPrototype {
    pub flags: u32,
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub color: [f32; 4],
    pub shape: u32,
}

pub struct Objects {
    pub flags: Vec<shape::FlagsInput>,
    pub position: Vec<shape::PositionInput>,
    pub size: Vec<shape::SizeInput>,
    pub color: Vec<shape::ColorInput>,
    pub shape: Vec<shape::ShapeInput>,
}

impl Objects {
    pub fn new() -> Self {
        Self {
            flags: vec![],
            position: vec![],
            size: vec![],
            color: vec![],
            shape: vec![],
        }
    }

    pub fn push(&mut self, prototype: ObjectPrototype) {
        self.flags.push(shape::FlagsInput { inner: prototype.flags });
        self.position.push(shape::PositionInput {
            inner: prototype.position,
        });
        self.size.push(shape::SizeInput { inner: prototype.size });
        self.color.push(shape::ColorInput { inner: prototype.color });
        self.shape.push(shape::ShapeInput { inner: prototype.shape });
    }

    pub fn reserve(&mut self, additional: usize) {
        self.flags.reserve(additional);
        self.position.reserve(additional);
        self.size.reserve(additional);
        self.color.reserve(additional);
        self.shape.reserve(additional);
    }

    pub fn extend(&mut self, iter: impl IntoIterator<Item = ObjectPrototype>) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for prototype in iter {
            self.push(prototype);
        }
    }

    pub fn len(&self) -> usize {
        self.flags.len()
    }
}
