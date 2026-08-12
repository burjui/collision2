use std::{fmt::Debug, str::FromStr, sync::LazyLock};

use anyhow::Context;
use itertools::Itertools as _;
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
    #[serde(default = "default_exit_at_limit")]
    pub exit_at_limit: bool,
    #[serde(default = "default_n_frames")]
    pub n_frames: usize,
    #[serde(default = "default_n_compute")]
    pub n_compute: usize,
    #[serde(default = "default_printouts")]
    pub printouts: bool,
    #[serde(default = "default_fps")]
    pub fps: f32,
    #[serde(default = "default_world_size")]
    pub world_size: String,

    #[serde(default = "default_accel")]
    pub accel: String,
    #[serde(default = "default_kick")]
    pub kick: String,

    #[serde(default = "default_blackhole")]
    pub blackhole: bool,
    #[serde(default = "default_blackhole_radius")]
    pub blackhole_radius: f32,
    #[serde(default = "default_blackhole_mass")]
    pub blackhole_mass: f32,
    #[serde(default = "default_blackhole_spin")]
    pub blackhole_spin: f32,

    #[serde(default = "default_particle_radius")]
    pub particle_radius: f32,
    #[serde(default = "default_particle_mass")]
    pub particle_mass: f32,
    #[serde(default = "default_particle_padding")]
    pub particle_padding: f32,
    #[serde(default = "default_particle_shape")]
    pub particle_shape: Shape,
    #[serde(default = "default_particle_position_rand")]
    pub particle_position_rand: f32,

    #[serde(default = "default_scene_scale")]
    pub scene_scale: f32,
    #[serde(default = "default_scene_offset")]
    pub scene_offset: String,

    #[serde(default = "default_image")]
    pub image: Option<String>,

    #[serde(default = "default_stiffness")]
    pub stiffness: f32,
    #[serde(default = "default_restitution")]
    pub restitution: f32,

    pub output_path: Option<String>,

    #[serde(default = "default_spectrum_width")]
    pub spectrum_width: f32,
}

impl Config {
    pub fn new() -> anyhow::Result<Config> {
        config::Config::builder()
            .add_source(config::Environment::default())
            .build()?
            .try_deserialize()
            .context("load config")
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
    s.split(delimiter).map(|d| d.parse().unwrap()).collect_array().unwrap()
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

fn default_exit_at_limit() -> bool {
    true
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
    60.0
}

fn default_world_size() -> String {
    "6400x4000".to_string()
}

fn default_accel() -> String {
    "0.0,-100.0".to_string()
}

fn default_kick() -> String {
    "0.0,0.0".to_string()
}

fn default_blackhole() -> bool {
    false
}

fn default_blackhole_radius() -> f32 {
    20.0
}

fn default_blackhole_mass() -> f32 {
    2.0
}

fn default_blackhole_spin() -> f32 {
    150.0
}

fn default_particle_radius() -> f32 {
    3.0
}

fn default_particle_mass() -> f32 {
    3.0
}

fn default_particle_padding() -> f32 {
    1.0
}

fn default_particle_shape() -> Shape {
    Shape::Circle
}

fn default_particle_position_rand() -> f32 {
    0.1
}

fn default_scene_offset() -> String {
    "0.0,0.0".to_string()
}

fn default_scene_scale() -> f32 {
    0.6
}

fn default_image() -> Option<String> {
    None
}

fn default_stiffness() -> f32 {
    100000.0
}

fn default_restitution() -> f32 {
    0.0
}

fn default_spectrum_width() -> f32 {
    320.0
}
