use std::sync::LazyLock;

use anyhow::Context;
use serde::Deserialize;

pub static CONFIG: LazyLock<Config> = LazyLock::new(|| Config::new().unwrap());

#[derive(Debug, Deserialize)]
pub struct Config {
    #[serde(default = "default_dt")]
    pub dt: f32,
}

fn default_dt() -> f32 {
    0.002
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
