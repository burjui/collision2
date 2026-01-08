use nalgebra::Vector2;

use crate::shaders::common::AABB;

pub trait AabbExt {
    fn min(&self) -> Vector2<f32>;
    fn max(&self) -> Vector2<f32>;

    fn size(&self) -> Vector2<f32> {
        self.max() - self.min()
    }
}

impl AabbExt for AABB {
    fn min(&self) -> Vector2<f32> {
        self.min.into()
    }

    fn max(&self) -> Vector2<f32> {
        self.max.into()
    }
}
