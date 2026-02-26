use nalgebra::Vector2;

use crate::shaders::common::AABB;

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

pub fn dispatch_dimensions(object_count: usize, workgroup_size: u32) -> (u32, u32, u32) {
    const MAX_DIMENSION: u32 = 65535;
    let total_workgroups = u32::try_from(object_count)
        .ok()
        .and_then(|c| c.checked_add(workgroup_size - 1))
        .map(|c| c / workgroup_size)
        .expect("object_count too large or overflow");
    let x = total_workgroups.min(MAX_DIMENSION);
    let y = (total_workgroups.div_ceil(x)).min(MAX_DIMENSION);
    let z = total_workgroups.div_ceil(x * y);
    (x, y, z)
}
