use wgpu::{BufferUsages, Device};

use crate::{
    device_buffer::DeviceBuffer,
    shaders::common::{AABB, Flags, Velocity},
};

/// Set of object phase states (change every frame)
#[derive(Clone)]
pub struct PhaseState {
    // TODO: split AABBs of objects and nodes
    aabbs: DeviceBuffer<AABB>,
    velocities: DeviceBuffer<Velocity>,
    flags: DeviceBuffer<Flags>,
}

impl PhaseState {
    fn new(
        index: usize,
        device: &Device,
        object_count: u32,
        initial_aabbs: &[AABB],
        initial_velocities: &[Velocity],
        initial_flags: &[Flags],
    ) -> Self {
        let aabbs_name = format!("aabbs #{index}");
        let velocities_name = format!("velocities #{index}");
        let flags_name = format!("flags #{index}");
        let (aabbs, velocities, flags) = if index == 0 {
            (
                DeviceBuffer::from_data(
                    device,
                    initial_aabbs,
                    &aabbs_name,
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                ),
                DeviceBuffer::from_data(
                    device,
                    initial_velocities,
                    &velocities_name,
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                ),
                DeviceBuffer::from_data(device, initial_flags, &flags_name, BufferUsages::STORAGE),
            )
        } else {
            (
                DeviceBuffer::new(device, object_count, &aabbs_name, BufferUsages::STORAGE | BufferUsages::COPY_SRC),
                DeviceBuffer::new(
                    device,
                    object_count,
                    &velocities_name,
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                ),
                DeviceBuffer::new(device, object_count, &flags_name, BufferUsages::STORAGE),
            )
        };
        Self {
            aabbs,
            velocities,
            flags,
        }
    }

    pub fn aabbs(&self) -> &DeviceBuffer<AABB> {
        &self.aabbs
    }

    pub fn velocities(&self) -> &DeviceBuffer<Velocity> {
        &self.velocities
    }

    pub fn flags(&self) -> &DeviceBuffer<Flags> {
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
        object_count: u32,
        initial_flags: &[Flags],
        initial_aabbs: &[AABB],
        initial_velocities: &[Velocity],
    ) -> Self {
        Self {
            states: (0..Self::CAPACITY)
                .map(|i| PhaseState::new(i, device, object_count, initial_aabbs, initial_velocities, initial_flags))
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
