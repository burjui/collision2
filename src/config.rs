use std::sync::LazyLock;

use anyhow::Context;
use serde::Deserialize;

use crate::shaders::render_shape::{SHAPE_CIRCLE, SHAPE_RECT};

pub static CONFIG: LazyLock<Config> = LazyLock::new(|| Config::new().unwrap());

#[derive(Debug, Deserialize)]
pub struct Config {
    #[serde(default = "default_dt")]
    pub dt: f32,
    #[serde(default = "default_headless")]
    pub headless: bool,
    pub sim_time_limit: Option<f32>,
    #[serde(default = "default_n_frames")]
    pub n_frames: usize,
    #[serde(default = "default_n_compute")]
    pub n_compute: usize,
    #[serde(default = "default_printouts")]
    pub printouts: bool,
    #[serde(default = "default_fps")]
    pub fps: f32,
    #[serde(default = "default_particle_radius")]
    pub particle_radius: f32,
    #[serde(default = "default_particle_padding")]
    pub particle_padding: f32,
    #[serde(default = "default_particle_shape")]
    pub particle_shape: Shape,
    #[serde(default = "default_image")]
    pub image: Option<String>,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, Deserialize)]
pub enum Shape {
    #[serde(rename = "circle")]
    Circle = SHAPE_CIRCLE,
    #[serde(rename = "rect")]
    Rect = SHAPE_RECT,
}

fn default_dt() -> f32 {
    0.002
}

fn default_headless() -> bool {
    false
}

fn default_n_frames() -> usize {
    2
}

fn default_n_compute() -> usize {
    3
}

fn default_printouts() -> bool {
    true
}

fn default_fps() -> f32 {
    30.0
}

fn default_particle_radius() -> f32 {
    3.0
}

fn default_particle_padding() -> f32 {
    1.0
}

fn default_particle_shape() -> Shape {
    Shape::Circle
}

fn default_image() -> Option<String> {
    None
}

impl Config {
    pub fn new() -> anyhow::Result<Config> {
        config::Config::builder()
            .add_source(config::Environment::default())
            .build()?
            .try_deserialize()
            .context("load config")
    }
}
