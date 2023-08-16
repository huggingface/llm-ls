use reqwest::header::AUTHORIZATION;
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::{Error, Result};
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RequestParams {
    max_new_tokens: u32,
    temperature: f32,
    do_sample: bool,
    top_p: f32,
    stop_token: String,
}

#[derive(Debug, Deserialize)]
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
    http_client: reqwest::Client,
    configuration: Arc<RwLock<Configuration>>,
}

fn internal_error<E: Display>(err: E) -> Error {
    Error {
        code: tower_lsp::jsonrpc::ErrorCode::InternalError,
        message: err.to_string().into(),
        data: None,
    }
}

async fn log(msg: &str) -> Result<()> {
    let current_time = chrono::Utc::now();
    let pid = std::process::id();
    let mut log_line = format!("[{current_time}] {{{pid}}} ");
    log_line.push_str(msg);
    tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(get_cache_dir_path()?.join("llm-ls.log"))
        .await
        .map_err(internal_error)?
        .write_all(log_line.as_bytes())
        .await
        .map_err(internal_error)?;
    Ok(())
}

fn get_cache_dir_path() -> Result<PathBuf> {
    let home_dir = home::home_dir().ok_or(internal_error("Failed to find home dir"))?;
    Ok(home_dir.join(".cache/llm_ls"))
}

async fn request_completion(
    http_client: &reqwest::Client,
    config: &Configuration,
) -> Result<Vec<Generation>> {
    let mut req = http_client.post(&config.model).json(&APIRequest {
        inputs: "Hello my name is ".to_owned(),
        parameters: config.request_params.clone(),
    });

    if let Some(api_token) = config.api_token.clone() {
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

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    #[serde(flatten)]
    text_document_position: TextDocumentPositionParams,
}

#[derive(Debug, Deserialize)]
struct Configuration {
    request_params: RequestParams,
    fim: FimParams,
    api_token: Option<String>,
    model: String,
}

impl Backend {
    async fn get_completions(&self, context: CompletionRequest) -> Result<Option<Vec<Completion>>> {
        log(&format!("get_completions {context:?}\n"))
            .await
            .map_err(internal_error)?;
        let _ = context;
        let result =
            request_completion(&self.http_client, &*self.configuration.read().await).await?;
        if result.len() > 0 {
            let generated_text = result[0].generated_text.clone();

            log(&format!("completion result: {generated_text}\n"))
                .await
                .map_err(internal_error)?;

            Ok(Some(vec![Completion { generated_text }]))
        } else {
            Ok(None)
        }
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, ip: InitializeParams) -> Result<InitializeResult> {
        tokio::fs::create_dir_all(get_cache_dir_path()?)
            .await
            .map_err(internal_error)?;
        if let Some(init_options) = ip.initialization_options {
            let configuration = serde_json::from_value(init_options).map_err(internal_error)?;
            log(&format!("configuration {configuration:?}\n"))
                .await
                .map_err(internal_error)?;
            let mut new_configuration = self.configuration.write().await;
            *new_configuration = configuration;
        };
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                completion_provider: Some(CompletionOptions {
                    resolve_provider: Some(false),
                    trigger_characters: Some(vec![
                        ".".to_owned(),
                        "(".to_owned(),
                        "{".to_owned(),
                        ":".to_owned(),
                        ":".to_owned(),
                    ]),
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "{llm-ls} initialized")
            .await;
        let _ = log("initialized\n").await;
    }

    // TODO:
    // handle textDocument/didOpen, textDocument/didChange, textDocument/didClose

    async fn did_open(&self, _: DidOpenTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "{llm-ls} file opened")
            .await;
        let _ = log("file opened\n").await;
    }

    async fn did_change(&self, _: DidChangeTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "{llm-ls} file changed")
            .await;
        let _ = log("file changed\n").await;
    }

    async fn did_save(&self, _: DidSaveTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "{llm-ls} file saved")
            .await;
        let _ = log("file saved\n").await;
    }

    async fn did_close(&self, _: DidCloseTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "{llm-ls} file closed")
            .await;
        let _ = log("file closed\n").await;
    }

    async fn shutdown(&self) -> Result<()> {
        let _ = log("shutdown\n").await;
        Ok(())
    }
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let http_client = reqwest::Client::new();

    let (service, socket) = LspService::build(|client| Backend {
        client,
        http_client,
        configuration: Arc::new(RwLock::new(Configuration {
            request_params: RequestParams {
                max_new_tokens: 60,
                temperature: 0.2,
                do_sample: true,
                top_p: 0.95,
                stop_token: "<|endoftext|>".to_owned(),
            },
            fim: FimParams {
                enabled: true,
                prefix: "<fim_prefix>".to_owned(),
                middle: "<fim_middle>".to_owned(),
                suffix: "<fim_suffix>".to_owned(),
            },
            api_token: None,
            model: "https://api-inference.huggingface.co/models/bigcode/starcoder".to_owned(),
        })),
    })
    .custom_method("llm-ls/getCompletions", Backend::get_completions)
    .finish();

    Server::new(stdin, stdout, socket).serve(service).await;
}
