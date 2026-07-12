use wgpu::ComputePass;

use crate::command_timings::CommandTimings;

pub trait ComputeStage {
    const LABEL: &'static str;

    fn compute(&self, compute_pass: &mut ComputePass, timings: &mut CommandTimings) {
        timings.measure_compute(compute_pass, Self::LABEL, |compute_pass| self.compute_impl(compute_pass));
    }

    fn compute_impl(&self, compute_pass: &mut ComputePass);
}
