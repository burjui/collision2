use color::{AlphaColor, Srgb};
use itertools::Itertools;
use nalgebra::Vector2;
use wgpu::BufferUsages;

use crate::{
    gpu_buffer::GpuBuffer,
    shaders::common::{AABB, BvhNode, Color, Flags, Mass, Shape, Velocity},
};

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

    pub fn len(&self) -> usize {
        self.flags.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn to_buffers(self, device: &wgpu::Device, queue: &wgpu::Queue) -> ObjectBuffers {
        let storage_copy_dst: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let storage_copy_src: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

        // These are twice the size to hold the BVH AABBs and nodes
        // TODO: come up with a scheme that saves memory
        let aabbs = GpuBuffer::new(self.len() * 2, "aabb buffer", storage_copy_dst, device);
        let bvh_nodes = GpuBuffer::new(self.len() * 2, "bvh node buffer", storage_copy_dst, device);

        let flags = GpuBuffer::new(self.len(), "flags buffer", storage_copy_dst, device);
        let velocities = GpuBuffer::new(self.len(), "velocity buffer", storage_copy_dst, device);
        let masses = GpuBuffer::new(self.len(), "mass buffer", storage_copy_dst, device);
        let colors = GpuBuffer::new(self.len(), "color buffer", storage_copy_dst, device);
        let shapes = GpuBuffer::new(self.len(), "shape buffer", storage_copy_dst, device);

        aabbs.write(queue, &self.aabbs);

        let bvh_leaves = (0..u32::try_from(self.len()).unwrap()).map(BvhNode::new).collect_vec();
        bvh_nodes.write(queue, &bvh_leaves);

        let integrated_velocities = GpuBuffer::new(self.len(), "integrated velocity buffer", storage_copy_src, device);
        let integrated_aabbs = GpuBuffer::new(self.len(), "integrated aabb buffer", storage_copy_src, device);

        flags.write(queue, &self.flags);
        velocities.write(queue, &self.velocities);
        masses.write(queue, &self.masses);
        colors.write(queue, &self.colors);
        shapes.write(queue, &self.shapes);

        ObjectBuffers {
            flags,
            aabbs,
            bvh_nodes,
            velocities,
            integrated_velocities,
            integrated_aabbs,
            masses,
            colors,
            shapes,
        }
    }
}

pub struct ObjectBuffers {
    pub flags: GpuBuffer<Flags>,
    pub aabbs: GpuBuffer<AABB>,
    pub bvh_nodes: GpuBuffer<BvhNode>,
    pub velocities: GpuBuffer<Velocity>,
    pub integrated_velocities: GpuBuffer<Velocity>,
    pub integrated_aabbs: GpuBuffer<AABB>,
    pub masses: GpuBuffer<Mass>,
    pub colors: GpuBuffer<Color>,
    pub shapes: GpuBuffer<Shape>,
}
