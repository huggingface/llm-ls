use std::path::Path;

use config::Config;
use serde::{Deserialize, Serialize};
use tokio::fs::write;

use crate::error::Result;

#[derive(Deserialize, Serialize)]
pub(crate) struct LlmLsConfig {
    /// .gitignore-like glob patterns to exclude from indexing
    pub(crate) ignored_paths: Vec<String>,
}

impl Default for LlmLsConfig {
    fn default() -> Self {
        Self {
            ignored_paths: vec![".git/".into(), ".idea/".into(), ".DS_Store/".into()],
        }
    }
}

pub async fn load_config(cache_path: &str) -> Result<LlmLsConfig> {
    let config_file_path = Path::new(cache_path).join("config.yaml");
    if config_file_path.exists() {
        Ok(Config::builder()
            .add_source(config::File::with_name(&format!("{cache_path}/config")))
            .add_source(config::Environment::with_prefix("LLM_LS"))
            .build()?
            .try_deserialize()?)
    } else {
        let config = LlmLsConfig::default();
        write(config_file_path, serde_yaml::to_string(&config)?.as_bytes()).await?;
        Ok(config)
    }
}
