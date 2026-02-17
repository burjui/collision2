use std::array::from_fn;

use wgpu::{BufferUsages, Device};

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
    fn new(index: usize, device: &Device, aabbs: &[AABB], velocities: &[Velocity], flags: &[Flags]) -> Self {
        let aabbs_name = format!("aabb buffer #{index}");
        let velocities_name = format!("velocity buffer #{index}");
        let flags_name = format!("flags buffer #{index}");
        let (aabbs, velocities, flags) = if index == 0 {
            (
                GpuBuffer::from_data(aabbs, &aabbs_name, BufferUsages::STORAGE, device),
                GpuBuffer::from_data(velocities, &velocities_name, BufferUsages::STORAGE, device),
                GpuBuffer::from_data(flags, &flags_name, BufferUsages::STORAGE, device),
            )
        } else {
            (
                GpuBuffer::new(aabbs.len(), &aabbs_name, BufferUsages::STORAGE, device),
                GpuBuffer::new(velocities.len(), &velocities_name, BufferUsages::STORAGE, device),
                GpuBuffer::new(flags.len(), &flags_name, BufferUsages::STORAGE, device),
            )
        };
        Self {
            aabbs,
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

const N_PHASE_STATES: usize = 2;
const _: () = assert!(N_PHASE_STATES > 1);

fn next_index(index: usize) -> usize {
    (index + 1) % N_PHASE_STATES
}

pub struct PhaseStateRing {
    states: [PhaseState; N_PHASE_STATES],
    oldest_index: usize,
    current_index: usize,
}

impl PhaseStateRing {
    pub const CAPACITY: usize = N_PHASE_STATES;

    pub fn new(
        device: &Device,
        initial_flags: &[Flags],
        initial_aabbs: &[AABB],
        initial_velocities: &[Velocity],
    ) -> Self {
        Self {
            states: from_fn(|i| PhaseState::new(i, device, initial_aabbs, initial_velocities, initial_flags)),
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
