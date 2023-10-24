use std::path::PathBuf;

use lsp_client::{client::LspClient, msg::RequestId, server::Server};
use lsp_types::{
    request::Request, DidOpenTextDocumentParams, InitializeParams, TextDocumentIdentifier,
    TextDocumentItem, TextDocumentPositionParams,
};
use serde::{Deserialize, Deserializer, Serialize};
use tracing::info;
use tracing_subscriber::EnvFilter;
use url::Url;
use uuid::Uuid;

const A_FILE: &str = r#"int main() {
  int a = 42;
  int forty_two_times_two = 
}
"#;

#[derive(Debug)]
enum GetCompletions {}

impl Request for GetCompletions {
    type Params = GetCompletionsParams;
    type Result = GetCompletionsResult;
    const METHOD: &'static str = "llm-ls/getCompletions";
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RequestParams {
    max_new_tokens: u32,
    temperature: f32,
    do_sample: bool,
    top_p: f32,
    stop_tokens: Option<Vec<String>>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum Ide {
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
struct FimParams {
    enabled: bool,
    prefix: String,
    middle: String,
    suffix: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum TokenizerConfig {
    Local { path: PathBuf },
    HuggingFace { repository: String },
    Download { url: String, to: PathBuf },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct GetCompletionsParams {
    #[serde(flatten)]
    text_document_position: TextDocumentPositionParams,
    request_params: RequestParams,
    #[serde(default)]
    #[serde(deserialize_with = "parse_ide")]
    ide: Ide,
    fim: FimParams,
    api_token: Option<String>,
    model: String,
    tokens_to_clear: Vec<String>,
    tokenizer_config: Option<TokenizerConfig>,
    context_window: usize,
    tls_skip_verify_insecure: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Completion {
    generated_text: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct GetCompletionsResult {
    request_id: Uuid,
    completions: Vec<Completion>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_target(true)
        .with_line_number(true)
        .with_env_filter(
            EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .json()
        .flatten_event(true)
        .with_current_span(false)
        .with_span_list(true)
        .init();

    let llm_ls_path = "/Users/mc/Documents/work/extensions/llm-ls/target/debug/llm-ls";
    let (conn, server) = Server::build().binary_path(llm_ls_path.into()).start()?;
    let client = LspClient::new(conn, server);
    client.send_request::<lsp_types::request::Initialize>(InitializeParams::default());
    client.send_notification::<lsp_types::notification::DidOpenTextDocument>(
        DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri: Url::parse("file://some/path/file.c").unwrap(),
                language_id: "c".into(),
                version: 0,
                text: A_FILE.into(),
            },
        },
    );
    let response = client.send_request::<GetCompletions>(GetCompletionsParams {
        api_token: Some("".into()),
        context_window: 2000,
        fim: FimParams {
            enabled: true,
            prefix: "<fim_prefix>".into(),
            middle: "<fim_middle>".into(),
            suffix: "<fim_suffix>".into(),
        },
        ide: Ide::Neovim,
        model: "bigcode/starcoder".into(),
        request_params: RequestParams {
            max_new_tokens: 100,
            temperature: 0.2,
            do_sample: true,
            top_p: 0.95,
            stop_tokens: None,
        },
        text_document_position: TextDocumentPositionParams {
            position: lsp_types::Position {
                line: 2,
                character: 28,
            },
            text_document: TextDocumentIdentifier {
                uri: Url::parse("file://some/path/file.c").unwrap(),
            },
        },
        tls_skip_verify_insecure: false,
        tokens_to_clear: vec!["<|endoftext|>".into()],
        tokenizer_config: None,
    });
    let (_, result): (RequestId, GetCompletionsResult) = response.extract()?;
    info!("got result: {result:?}");
    client.shutdown();
    client.exit();
    Ok(())
}
