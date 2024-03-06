use std::path::Path;

use config::Config;
use serde::{Deserialize, Serialize};
use tokio::fs::write;

use crate::error::Result;

#[derive(Clone, Deserialize, Serialize)]
pub(crate) struct ModelConfig {
    pub(crate) id: String,
    pub(crate) revision: String,
    pub(crate) embeddings_size: usize,
    pub(crate) max_input_size: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            id: "intfloat/multilingual-e5-small".to_string(),
            revision: "main".to_string(),
            embeddings_size: 384,
            max_input_size: 512,
        }
    }
}

#[derive(Deserialize, Serialize)]
pub(crate) struct LlmLsConfig {
    pub(crate) model: ModelConfig,
    /// .gitignore-like glob patterns to exclude from indexing
    pub(crate) ignored_paths: Vec<String>,
}

impl Default for LlmLsConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            ignored_paths: vec![".git".into(), ".idea".into(), ".DS_Store".into()],
        }
    }
}

/// Loads configuration from a file and environment variables.
///
/// If the file does not exist, it will be created with the default configuration.
///
/// # Arguments
///
/// * `cache_path` - Path to the directory where the configuration file will be stored.
pub(crate) async fn load_config(cache_path: &str) -> Result<LlmLsConfig> {
    let config_file_path = Path::new(cache_path).join("config.yaml");
    let config = if config_file_path.exists() {
        Config::builder()
            .add_source(config::File::with_name(&format!("{cache_path}/config")))
            .add_source(config::Environment::with_prefix("LLM_LS"))
            .build()?
            .try_deserialize()?
    } else {
        let config = LlmLsConfig::default();
        write(config_file_path, serde_yaml::to_string(&config)?.as_bytes()).await?;
        config
    };
    Ok(config)
}
