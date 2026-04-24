use wgpu::{BufferUsages, Device, Queue};

use crate::{
    shaders::common::{AABB, Flags, Velocity},
    typed_buffer::TypedBuffer,
};

/// Set of object phase states (change every frame)
/// NOTE: it doesn't contain BVH nodes because these are not used by the renderer
#[derive(Clone)]
pub struct PhaseState {
    // TODO: split AABBs of objects and nodes
    aabbs: TypedBuffer<AABB>,
    velocities: TypedBuffer<Velocity>,
    flags: TypedBuffer<Flags>,
}

impl PhaseState {
    fn new(
        device: &Device,
        queue: &Queue,
        index: usize,
        initial_aabbs: &[AABB],
        initial_velocities: &[Velocity],
        initial_flags: &[Flags],
        node_count: usize,
    ) -> Self {
        let aabbs_name = format!("aabbs #{index}");
        let velocities_name = format!("velocities #{index}");
        let flags_name = format!("flags #{index}");
        let (velocities, flags) = if index == 0 {
            (
                TypedBuffer::from_data(
                    device,
                    initial_velocities,
                    &velocities_name,
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                ),
                TypedBuffer::from_data(device, initial_flags, &flags_name, BufferUsages::STORAGE),
            )
        } else {
            (
                TypedBuffer::new(
                    device,
                    initial_velocities.len(),
                    &velocities_name,
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                ),
                TypedBuffer::new(device, initial_flags.len(), &flags_name, BufferUsages::STORAGE),
            )
        };
        let aabbs = TypedBuffer::new(device, node_count, &aabbs_name, BufferUsages::STORAGE | BufferUsages::COPY_DST);
        aabbs.write(queue, initial_aabbs);
        Self {
            aabbs,
            velocities,
            flags,
        }
    }

    pub fn aabbs(&self) -> &TypedBuffer<AABB> {
        &self.aabbs
    }

    pub fn velocities(&self) -> &TypedBuffer<Velocity> {
        &self.velocities
    }

    pub fn flags(&self) -> &TypedBuffer<Flags> {
        &self.flags
    }
}

pub struct PhaseStateRing {
    states: Vec<PhaseState>,
    frame_index: usize,
    compute_index: usize,
}

impl PhaseStateRing {
    pub const N_FRAMES: usize = 2;
    pub const N_COMPUTE: usize = 3;

    pub const CAPACITY: usize = {
        assert!(Self::N_FRAMES >= 1);
        assert!(Self::N_COMPUTE >= 2);
        Self::N_FRAMES + Self::N_COMPUTE
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
            states: (0..Self::CAPACITY)
                .map(|i| {
                    PhaseState::new(device, queue, i, initial_aabbs, initial_velocities, initial_flags, node_count)
                })
                .collect(),
            frame_index: 0,
            compute_index: 0,
        }
    }

    pub fn current_frame(&self) -> &PhaseState {
        &self.states[self.frame_index]
    }

    pub fn current_frame_index(&self) -> usize {
        self.frame_index
    }

    pub fn current_compute(&self) -> &PhaseState {
        &self.states[self.compute_index]
    }

    pub fn next_compute(&self) -> &PhaseState {
        &self.states[Self::next_index(self.compute_index)]
    }

    pub fn current_compute_index(&self) -> usize {
        self.compute_index
    }

    pub fn advance_frame(&mut self) {
        let next_index = Self::next_index(self.frame_index);
        if next_index != self.compute_index {
            self.frame_index = next_index;
        }
    }

    pub fn advance_compute(&mut self) {
        self.compute_index = Self::next_index(self.compute_index);
        if Self::next_index(self.compute_index) == self.frame_index {
            self.frame_index = Self::next_index(self.frame_index);
        }
    }

    fn next_index(index: usize) -> usize {
        (index + 1) % Self::CAPACITY
    }
}
