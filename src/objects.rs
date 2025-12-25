use nalgebra::Vector2;
use wgpu::BufferUsages;

use crate::{
    gpu_buffer::GpuBuffer,
    shaders::common::{AABB, Color, Flags, Mass, Shape, Velocity},
};

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
        self.colors.push(Color::new(prototype.color));
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

    pub fn len(&self) -> usize {
        self.flags.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn to_buffers(self, device: &wgpu::Device, queue: &wgpu::Queue) -> ObjectBuffers {
        let access_mode = BufferUsages::COPY_DST;
        let flags = GpuBuffer::new(self.len(), "flags buffer", BufferUsages::STORAGE | access_mode, device);
        let aabbs = GpuBuffer::new(self.len(), "aabb buffer", BufferUsages::STORAGE | access_mode, device);
        let velocities = GpuBuffer::new(self.len(), "velocity buffer", BufferUsages::STORAGE | access_mode, device);
        let masses = GpuBuffer::new(self.len(), "mass buffer", BufferUsages::STORAGE | access_mode, device);
        let colors = GpuBuffer::new(self.len(), "color buffer", BufferUsages::STORAGE | access_mode, device);
        let shapes = GpuBuffer::new(self.len(), "shape buffer", BufferUsages::STORAGE | access_mode, device);
        flags.write(queue, &self.flags);
        aabbs.write(queue, &self.aabbs);
        velocities.write(queue, &self.velocities);
        masses.write(queue, &self.masses);
        colors.write(queue, &self.colors);
        shapes.write(queue, &self.shapes);
        ObjectBuffers {
            flags,
            aabbs,
            velocities,
            masses,
            colors,
            shapes,
        }
    }
}

pub struct ObjectBuffers {
    pub flags: GpuBuffer<Flags>,
    pub aabbs: GpuBuffer<AABB>,
    pub velocities: GpuBuffer<Velocity>,
    pub masses: GpuBuffer<Mass>,
    pub colors: GpuBuffer<Color>,
    pub shapes: GpuBuffer<Shape>,
}
