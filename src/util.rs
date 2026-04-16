use nalgebra::Vector2;

use crate::shaders::common::{AABB, Velocity};

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

pub fn dispatch_dimensions(object_count: u32, workgroup_size: u32) -> (u32, u32, u32) {
    const MAX_DIMENSION: u32 = 65535;
    let total_workgroups = object_count.div_ceil(workgroup_size);
    let x = total_workgroups.min(MAX_DIMENSION);
    let y = (total_workgroups.div_ceil(x)).min(MAX_DIMENSION);
    let z = total_workgroups.div_ceil(x * y);
    (x, y, z)
}
