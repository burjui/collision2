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
                GpuBuffer::from_data(
                    velocities,
                    &velocities_name,
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                    device,
                ),
                GpuBuffer::from_data(flags, &flags_name, BufferUsages::STORAGE, device),
            )
        } else {
            (
                GpuBuffer::new(
                    velocities.len(),
                    &velocities_name,
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                    device,
                ),
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
                    PhaseState::new(i, device, queue, initial_aabbs, initial_velocities, initial_flags, node_count)
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
