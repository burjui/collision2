use wgpu::{BufferUsages, Device};

use crate::{
    command_timings::CommandTimings,
    config::CONFIG,
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
    command_timings: CommandTimings,
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
            command_timings: CommandTimings::new(device, 20),
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

    pub fn command_timings(&mut self) -> &mut CommandTimings {
        &mut self.command_timings
    }
}

#[derive(Clone, Copy)]
pub struct PhaseStateRingConfig {
    pub n_frames: usize,
    pub n_compute: usize,
}

impl PhaseStateRingConfig {
    pub fn capacity(&self) -> usize {
        self.n_frames + self.n_compute
    }
}

pub struct PhaseStateRing {
    capacity: usize,
    states: Vec<PhaseState>,
    frame_index: usize,
    compute_index: usize,
}

impl PhaseStateRing {
    pub fn new(
        config: PhaseStateRingConfig,
        device: &Device,
        object_count: u32,
        initial_flags: &[Flags],
        initial_aabbs: &[AABB],
        initial_velocities: &[Velocity],
    ) -> Self {
        assert!(config.n_frames >= if CONFIG.headless { 0 } else { 1 });
        assert!(config.n_compute >= 2);
        let capacity = config.capacity();
        Self {
            capacity,
            states: (0..capacity)
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
        &self.states[self.next_index(self.compute_index)]
    }

    pub fn current_compute_index(&self) -> usize {
        self.compute_index
    }

    pub fn advance_frame(&mut self) {
        let next_index = self.next_index(self.frame_index);
        if next_index != self.compute_index {
            self.frame_index = next_index;
        }
    }

    pub fn advance_compute(&mut self) {
        self.compute_index = self.next_index(self.compute_index);
        if self.next_index(self.compute_index) == self.frame_index {
            self.frame_index = self.next_index(self.frame_index);
        }
    }

    fn next_index(&self, index: usize) -> usize {
        (index + 1) % self.capacity
    }
}
