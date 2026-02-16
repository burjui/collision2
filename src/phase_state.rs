use std::iter::{once, repeat_with};

use itertools::Itertools as _;
use wgpu::BufferUsages;

use crate::{
    gpu_buffer::GpuBuffer,
    shaders::common::{AABB, Flags, Velocity},
};

#[derive(Clone)]
pub struct PhaseStates {
    id: usize,
    device: wgpu::Device,
    aabbs: GpuBuffer<AABB>,
    velocities: GpuBuffer<Velocity>,
    flags: GpuBuffer<Flags>,
}

impl PhaseStates {
    fn new(
        device: &wgpu::Device,
        aabbs: GpuBuffer<AABB>,
        velocities: GpuBuffer<Velocity>,
        flags: GpuBuffer<Flags>,
    ) -> Self {
        Self {
            id: 0,
            device: device.clone(),
            aabbs,
            velocities,
            flags,
        }
    }

    fn duplicate(&self) -> Self {
        let id = self.id + 1;
        let storage_copy_dst: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_DST;

        let aabbs_name = format!("aabb buffer #{id}");
        let aabbs = GpuBuffer::new(self.aabbs.len(), &aabbs_name, storage_copy_dst, &self.device);

        let velocities_name = format!("velocity buffer #{id}");
        let velocities = GpuBuffer::new(self.velocities.len(), &velocities_name, storage_copy_dst, &self.device);

        let flags_name = format!("flags buffer #{id}");
        let flags = GpuBuffer::new(self.flags.len(), &flags_name, storage_copy_dst, &self.device);

        Self {
            id,
            device: self.device.clone(),
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

const N_PHASE_STATES: usize = 3;

pub struct PhaseStateBuffers {
    states: [PhaseStates; N_PHASE_STATES],
    oldest: usize,
    latest: usize,
}

impl PhaseStateBuffers {
    pub fn new(
        device: &wgpu::Device,
        aabbs: GpuBuffer<AABB>,
        velocities: GpuBuffer<Velocity>,
        flags: GpuBuffer<Flags>,
    ) -> Self {
        let first = PhaseStates::new(device, aabbs, velocities, flags);
        Self {
            states: once(first.clone())
                .chain(repeat_with(|| first.duplicate()))
                .take(N_PHASE_STATES)
                .collect_array()
                .unwrap(),
            oldest: 0,
            latest: 0,
        }
    }

    pub fn oldest(&self) -> PhaseStates {
        self.states[self.oldest].clone()
    }

    pub fn next_pair(&mut self) -> (PhaseStates, PhaseStates) {
        let src = self.states[self.latest].clone();
        self.latest = (self.latest + 1) % self.states.len();
        let dst = self.states[self.latest].clone();
        if self.latest == self.oldest {
            self.oldest = (self.oldest + 1) % self.states.len();
        }
        (src, dst)
    }

    pub fn pair_count(&self) -> usize {
        self.states.len() - 1
    }
}
