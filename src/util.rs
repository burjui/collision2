use std::iter::from_fn;

use nalgebra::Vector2;
use wgpu::ComputePass;

use crate::{
    phase_state::PhaseStateRingConfig,
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
    let nwg = n_threads.div_ceil(WORKGROUP_SIZE);
    let n_indirect_dispatches = nwg.div_ceil(MAX_DISPATCH_DIMENSION);
    for i in 0..n_indirect_dispatches {
        let thread_offset = i * MAX_DISPATCH_DIMENSION * WORKGROUP_SIZE;
        compute_pass.set_immediates(0, bytemuck::cast_slice(&[thread_offset]));
        let current_nwg = (nwg - MAX_DISPATCH_DIMENSION * i).min(MAX_DISPATCH_DIMENSION);
        compute_pass.dispatch_workgroups(current_nwg, 1, 1);
    }
}

pub struct PhaseStateCache<T> {
    data: Vec<Option<T>>,
    phase_state_index: Option<usize>,
}

impl<T> PhaseStateCache<T> {
    pub fn new(phase_state_ring_config: PhaseStateRingConfig) -> Self {
        Self {
            data: from_fn(|| Some(None)).take(phase_state_ring_config.capacity()).collect(),
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
