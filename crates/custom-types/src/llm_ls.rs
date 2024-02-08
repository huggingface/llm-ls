use std::{fmt::Display, path::PathBuf};

use lsp_types::TextDocumentPositionParams;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AcceptCompletionParams {
    pub request_id: Uuid,
    pub accepted_completion: u32,
    pub shown_completions: Vec<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RejectCompletionParams {
    pub request_id: Uuid,
    pub shown_completions: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Ide {
    Neovim,
    VSCode,
    JetBrains,
    Emacs,
    Jupyter,
    Sublime,
    VisualStudio,
    #[default]
    Unknown,
}

impl Display for Ide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.serialize(f)
    }
}

fn parse_ide<'de, D>(d: D) -> std::result::Result<Ide, D::Error>
where
    D: Deserializer<'de>,
{
    Deserialize::deserialize(d).map(|b: Option<_>| b.unwrap_or(Ide::Unknown))
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    #[default]
    HuggingFace,
    Ollama,
    OpenAi,
    Tgi,
}

impl Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.serialize(f)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FimParams {
    pub enabled: bool,
    pub prefix: String,
    pub middle: String,
    pub suffix: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum TokenizerConfig {
    Local {
        path: PathBuf,
    },
    HuggingFace {
        repository: String,
        api_token: Option<String>,
    },
    Download {
        url: String,
        to: PathBuf,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GetCompletionsParams {
    #[serde(flatten)]
    pub text_document_position: TextDocumentPositionParams,
    #[serde(default)]
    #[serde(deserialize_with = "parse_ide")]
    pub ide: Ide,
    pub fim: FimParams,
    pub api_token: Option<String>,
    pub model: String,
    pub backend: Backend,
    pub tokens_to_clear: Vec<String>,
    pub tokenizer_config: Option<TokenizerConfig>,
    pub context_window: usize,
    pub tls_skip_verify_insecure: bool,
    pub request_body: Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Completion {
    pub generated_text: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GetCompletionsResult {
    pub request_id: Uuid,
    pub completions: Vec<Completion>,
}
