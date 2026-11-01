use std::{fmt::Debug, str::FromStr};

use anyhow::Context;
use config::FileFormat;
use serde::Deserialize;

use crate::shaders::render_shape::{SHAPE_CIRCLE, SHAPE_RECT};

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub print_config: bool,
    pub print_adapter: bool,

    pub dt: f32,
    pub headless: bool,
    pub sim_time_limit: f32,
    pub exit_at_limit: bool,
    pub n_frames: usize,
    pub n_compute: usize,
    pub printouts: bool,
    pub fps: f32,
    pub world_size: String,

    pub accel: String,
    pub kick: String,
    pub kick_radius: f32,

    pub blackhole: bool,
    pub blackhole_radius: f32,
    pub blackhole_mass: f32,
    pub blackhole_spin: f32,

    pub particle_radius: f32,
    pub particle_mass: f32,
    pub particle_padding: f32,
    pub particle_shape: Shape,
    pub particle_position_rand: f32,

    pub scene_scale: f32,
    pub scene_offset: String,

    pub image: String,

    pub stiffness: f32,
    pub restitution: f32,

    pub output_path: String,

    pub spectrum_width: f32,
}

impl AppConfig {
    pub fn new(path: Option<&str>) -> anyhow::Result<AppConfig> {
        let mut builder = config::Config::builder().add_source(config::File::new("defaults.toml", FileFormat::Toml));
        if let Some(path) = path {
            builder = builder.add_source(config::File::new(path, FileFormat::Toml));
        }
        builder.add_source(config::Environment::default()).build()?.try_deserialize().context("load config")
    }

    pub fn world_size(&self) -> [f32; 2] {
        parse_pair(&self.world_size, "x")
    }

    pub fn accel(&self) -> [f32; 2] {
        parse_pair(&self.accel, ",")
    }

    pub fn kick(&self) -> [f32; 2] {
        parse_pair(&self.kick, ",")
    }

    pub fn scene_offset(&self) -> [f32; 2] {
        parse_pair(&self.scene_offset, ",")
    }
}

fn parse_pair<T: FromStr>(s: &str, delimiter: &str) -> [T; 2]
where
    <T as FromStr>::Err: Debug,
{
    let mut iter = s.split(delimiter).map(|d| d.parse().unwrap());
    [iter.next().unwrap(), iter.next().unwrap()]
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, Deserialize)]
pub enum Shape {
    #[serde(rename = "circle")]
    Circle = SHAPE_CIRCLE,
    #[serde(rename = "rect")]
    Rect = SHAPE_RECT,
}
