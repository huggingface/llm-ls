use std::path::PathBuf;

use lsp_types::{request::Request, TextDocumentPositionParams};
use serde::{Deserialize, Deserializer, Serialize};
use uuid::Uuid;

#[derive(Debug)]
pub(crate) enum GetCompletions {}

impl Request for GetCompletions {
    type Params = GetCompletionsParams;
    type Result = GetCompletionsResult;
    const METHOD: &'static str = "llm-ls/getCompletions";
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RequestParams {
    pub(crate) max_new_tokens: u32,
    pub(crate) temperature: f32,
    pub(crate) do_sample: bool,
    pub(crate) top_p: f32,
    pub(crate) stop_tokens: Option<Vec<String>>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Ide {
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

fn parse_ide<'de, D>(d: D) -> std::result::Result<Ide, D::Error>
where
    D: Deserializer<'de>,
{
    Deserialize::deserialize(d).map(|b: Option<_>| b.unwrap_or(Ide::Unknown))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct FimParams {
    pub(crate) enabled: bool,
    pub(crate) prefix: String,
    pub(crate) middle: String,
    pub(crate) suffix: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub(crate) enum TokenizerConfig {
    Local { path: PathBuf },
    HuggingFace { repository: String },
    Download { url: String, to: PathBuf },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GetCompletionsParams {
    #[serde(flatten)]
    pub(crate) text_document_position: TextDocumentPositionParams,
    pub(crate) request_params: RequestParams,
    #[serde(default)]
    #[serde(deserialize_with = "parse_ide")]
    pub(crate) ide: Ide,
    pub(crate) fim: FimParams,
    pub(crate) api_token: Option<String>,
    pub(crate) model: String,
    pub(crate) tokens_to_clear: Vec<String>,
    pub(crate) tokenizer_config: Option<TokenizerConfig>,
    pub(crate) context_window: usize,
    pub(crate) tls_skip_verify_insecure: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Completion {
    pub(crate) generated_text: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct GetCompletionsResult {
    request_id: Uuid,
    pub(crate) completions: Vec<Completion>,
}
