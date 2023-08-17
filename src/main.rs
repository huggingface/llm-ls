use reqwest::header::AUTHORIZATION;
use ropey::Rope;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Display;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::{Error, Result};
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use tracing::info;
use tracing_appender::rolling;
use tracing_subscriber::EnvFilter;

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RequestParams {
    max_new_tokens: u32,
    temperature: f32,
    do_sample: bool,
    top_p: f32,
    stop_token: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct FimParams {
    enabled: bool,
    prefix: String,
    middle: String,
    suffix: String,
}

#[derive(Serialize)]
struct APIRequest {
    inputs: String,
    parameters: RequestParams,
}

#[derive(Debug, Deserialize)]
struct Generation {
    generated_text: String,
}

#[derive(Debug, Deserialize)]
struct APIError {
    error: String,
}

impl Display for APIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum APIResponse {
    Generations(Vec<Generation>),
    Error(APIError),
}

#[derive(Debug)]
struct Backend {
    client: Client,
    document_map: Arc<RwLock<HashMap<String, Rope>>>,
    http_client: reqwest::Client,
}

fn internal_error<E: Display>(err: E) -> Error {
    Error {
        code: tower_lsp::jsonrpc::ErrorCode::InternalError,
        message: err.to_string().into(),
        data: None,
    }
}

fn get_cache_dir_path() -> Result<PathBuf> {
    let home_dir = home::home_dir().ok_or(internal_error("Failed to find home dir"))?;
    Ok(home_dir.join(".cache/llm_ls"))
}

fn build_prompt(pos: Position, text: &Rope) -> Result<String> {
    let offset = text
        .try_line_to_char(pos.line as usize)
        .map_err(internal_error)?
        + pos.character as usize;
    if offset == 0 {
        Ok("".to_owned())
    } else {
        Ok(text.slice(0..offset).to_string())
    }
}

async fn request_completion(
    http_client: &reqwest::Client,
    model: &str,
    request_params: RequestParams,
    api_token: Option<String>,
    prompt: String,
) -> Result<Vec<Generation>> {
    let mut req = http_client.post(model).json(&APIRequest {
        inputs: prompt,
        parameters: request_params,
    });

    if let Some(api_token) = api_token.clone() {
        req = req.header(AUTHORIZATION, format!("Bearer {api_token}"))
    }

    let res = req.send().await.map_err(internal_error)?;

    match res.json().await.map_err(internal_error)? {
        APIResponse::Generations(gens) => Ok(gens),
        APIResponse::Error(err) => Err(internal_error(err)),
    }
}

#[derive(Deserialize, Serialize)]
struct Completion {
    generated_text: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct CompletionParams {
    #[serde(flatten)]
    text_document_position: TextDocumentPositionParams,
    request_params: RequestParams,
    fim: FimParams,
    api_token: Option<String>,
    model: String,
}

impl Backend {
    async fn get_completions(&self, params: CompletionParams) -> Result<Option<Vec<Completion>>> {
        info!("get_completions {params:?}");
        let document_map = self.document_map.read().await;
        let text = document_map
            .get(params.text_document_position.text_document.uri.as_str())
            .ok_or(internal_error("failed to find document"))?;
        let prompt = build_prompt(params.text_document_position.position, text)?;
        let result = request_completion(
            &self.http_client,
            &params.model,
            params.request_params,
            params.api_token,
            prompt.clone(),
        )
        .await?;
        if result.len() > 0 {
            let generated_text = ropey::Rope::from_str(&result[0].generated_text);

            let offset = prompt.len();
            Ok(Some(vec![Completion {
                generated_text: generated_text.slice(offset..).to_string(),
            }]))
        } else {
            Ok(None)
        }
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        tokio::fs::create_dir_all(get_cache_dir_path()?)
            .await
            .map_err(internal_error)?;
        Ok(InitializeResult {
            server_info: Some(ServerInfo {
                name: "llm-ls".to_owned(),
                version: None,
            }),
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "{llm-ls} initialized")
            .await;
        let _ = info!("initialized\n");
    }

    // TODO:
    // textDocument/didClose

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "{llm-ls} file opened")
            .await;
        let rope = ropey::Rope::from_str(&params.text_document.text);
        let uri = params.text_document.uri.to_string();
        self.document_map
            .write()
            .await
            .entry(uri)
            .or_insert(rope.clone());
        let _ = info!("file opened\n");
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "{llm-ls} file changed")
            .await;
        let rope = ropey::Rope::from_str(&params.content_changes[0].text);
        let uri = params.text_document.uri.to_string();
        self.document_map
            .write()
            .await
            .entry(uri)
            .or_insert(rope.clone());
        let _ = info!("file changed\n");
    }

    async fn did_save(&self, _: DidSaveTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "{llm-ls} file saved")
            .await;
        let _ = info!("file saved\n");
    }

    async fn did_close(&self, _: DidCloseTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "{llm-ls} file closed")
            .await;
        let _ = info!("file closed\n");
    }

    async fn shutdown(&self) -> Result<()> {
        let _ = info!("shutdown");
        Ok(())
    }
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let log_file = rolling::never(
        get_cache_dir_path().expect("failed to find home dir"),
        "llm-ls.log",
    );
    let builder = tracing_subscriber::fmt()
        .with_writer(log_file)
        .with_target(true)
        .with_line_number(true)
        .with_env_filter(
            EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("info")),
        );

    builder
        .json()
        .flatten_event(true)
        .with_current_span(false)
        .with_span_list(true)
        .init();

    let http_client = reqwest::Client::new();

    let (service, socket) = LspService::build(|client| Backend {
        client,
        document_map: Arc::new(RwLock::new(HashMap::new())),
        http_client,
    })
    .custom_method("llm-ls/getCompletions", Backend::get_completions)
    .finish();

    Server::new(stdin, stdout, socket).serve(service).await;
}
