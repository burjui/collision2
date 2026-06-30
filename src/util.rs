use std::array::from_fn;

use nalgebra::Vector2;
use wgpu::ComputePass;

use crate::{
    phase_state::PhaseStateRing,
    shaders::common::{AABB, MAX_DISPATCH_DIMENSION, Velocity, WORKGROUP_SIZE},
};

impl AABB {
    pub fn min(&self) -> Vector2<f32> {
        self.min.into()
    }

    pub fn max(&self) -> Vector2<f32> {
        self.max.into()
    }

    pub fn size(&self) -> Vector2<f32> {
        self.max() - self.min()
    }
}

impl Default for Velocity {
    fn default() -> Self {
        Self { inner: [0.0, 0.0] }
    }
}

/// NOTE: uses immediates for thread offset
pub fn dispatch_compute(compute_pass: &mut ComputePass, n_threads: u32) {
    let mut nwg = n_threads.div_ceil(WORKGROUP_SIZE);
    let mut thread_offset: u32 = 0;
    while nwg > MAX_DISPATCH_DIMENSION {
        compute_pass.set_immediates(0, bytemuck::cast_slice(&[thread_offset]));
        compute_pass.dispatch_workgroups(MAX_DISPATCH_DIMENSION, 1, 1);
        nwg -= MAX_DISPATCH_DIMENSION;
        thread_offset += MAX_DISPATCH_DIMENSION * WORKGROUP_SIZE;
    }
    compute_pass.dispatch_workgroups(nwg, 1, 1);
}

pub struct PhaseStateCache<T> {
    data: [Option<T>; PhaseStateRing::CAPACITY],
    phase_state_index: Option<usize>,
}

impl<T> Default for PhaseStateCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> PhaseStateCache<T> {
    pub fn new() -> Self {
        Self {
            data: from_fn(|_| None),
            phase_state_index: None,
        }
    }

    pub fn update(&mut self, phase_state_index: usize, f: impl FnOnce() -> T) {
        self.data[phase_state_index].get_or_insert_with(f);
        self.phase_state_index = Some(phase_state_index);
    }

    pub fn get_current(&self) -> &T {
        self.data[self.phase_state_index.expect("forgot to call update()?")].as_ref().expect("forgot to call update()?")
    }
}
