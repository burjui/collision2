use std::sync::LazyLock;

use anyhow::Context;
use serde::Deserialize;

pub static CONFIG: LazyLock<Config> = LazyLock::new(|| Config::new().unwrap());

#[derive(Debug, Deserialize)]
pub struct Config {
    #[serde(default = "default_dt")]
    pub dt: f32,
    #[serde(default = "default_headless")]
    pub headless: bool,
    pub sim_time_limit: Option<f32>,
}

fn default_dt() -> f32 {
    0.002
}

fn default_headless() -> bool {
    false
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
