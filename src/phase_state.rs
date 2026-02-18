use std::array::from_fn;

use wgpu::{BufferUsages, Device, Queue};

use crate::{
    gpu_buffer::GpuBuffer,
    shaders::common::{AABB, Flags, Velocity},
};

/// Set of object phase states (change every frame)
/// NOTE: it doesn't contain BVH nodes because these are not used by the renderer
#[derive(Clone)]
pub struct PhaseState {
    // TODO: split AABBs of objects and nodes
    aabbs: GpuBuffer<AABB>,
    velocities: GpuBuffer<Velocity>,
    flags: GpuBuffer<Flags>,
}

impl PhaseState {
    fn new(
        index: usize,
        device: &Device,
        queue: &Queue,
        aabbs: &[AABB],
        velocities: &[Velocity],
        flags: &[Flags],
        node_count: usize,
    ) -> Self {
        let aabbs_name = format!("aabb buffer #{index}");
        let velocities_name = format!("velocity buffer #{index}");
        let flags_name = format!("flags buffer #{index}");
        let (velocities, flags) = if index == 0 {
            (
                GpuBuffer::from_data(velocities, &velocities_name, BufferUsages::STORAGE, device),
                GpuBuffer::from_data(flags, &flags_name, BufferUsages::STORAGE, device),
            )
        } else {
            (
                GpuBuffer::new(velocities.len(), &velocities_name, BufferUsages::STORAGE, device),
                GpuBuffer::new(flags.len(), &flags_name, BufferUsages::STORAGE, device),
            )
        };
        let aabbs_buffer =
            GpuBuffer::new(node_count, &aabbs_name, BufferUsages::STORAGE | BufferUsages::COPY_DST, device);
        aabbs_buffer.write(queue, aabbs);
        Self {
            aabbs: aabbs_buffer,
            velocities,
            flags,
        }
    }

    pub fn aabbs(&self) -> &GpuBuffer<AABB> {
        &self.aabbs
    }

    pub fn velocities(&self) -> &GpuBuffer<Velocity> {
        &self.velocities
    }

    pub fn flags(&self) -> &GpuBuffer<Flags> {
        &self.flags
    }
}

pub struct PhaseStateRing {
    states: [PhaseState; Self::CAPACITY],
    oldest_index: usize,
    current_index: usize,
}

impl PhaseStateRing {
    pub const CAPACITY: usize = {
        const CAPACITY: usize = 4;
        assert!(CAPACITY > 1);
        CAPACITY
    };

    pub fn new(
        device: &Device,
        queue: &Queue,
        initial_flags: &[Flags],
        initial_aabbs: &[AABB],
        initial_velocities: &[Velocity],
        node_count: usize,
    ) -> Self {
        Self {
            states: from_fn(|i| {
                PhaseState::new(i, device, queue, initial_aabbs, initial_velocities, initial_flags, node_count)
            }),
            oldest_index: 0,
            current_index: 0,
        }
    }

    pub fn oldest(&self) -> &PhaseState {
        &self.states[self.oldest_index]
    }

    pub fn current(&self) -> &PhaseState {
        &self.states[self.current_index]
    }

    pub fn next(&self) -> &PhaseState {
        &self.states[next_index(self.current_index)]
    }

    pub fn current_index(&self) -> usize {
        self.current_index
    }

    pub fn advance(&mut self) {
        self.current_index = next_index(self.current_index);
        if self.current_index == self.oldest_index {
            self.oldest_index = next_index(self.oldest_index);
        }
    }
}

fn next_index(index: usize) -> usize {
    (index + 1) % PhaseStateRing::CAPACITY
}
