mod language_comments;

use language_comments::build_language_comments;
use reqwest::header::AUTHORIZATION;
use ropey::Rope;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::{Error, Result};
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use tracing::{error, info};
use tracing_appender::rolling;
use tracing_subscriber::EnvFilter;

#[derive(Clone, Debug, Deserialize)]
pub struct LanguageComment {
    open: String,
    close: Option<String>,
}

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

#[derive(Debug, Serialize)]
struct APIParams {
    max_new_tokens: u32,
    temperature: f32,
    do_sample: bool,
    top_p: f32,
    stop: Vec<String>,
    return_full_text: bool,
}

impl From<RequestParams> for APIParams {
    fn from(params: RequestParams) -> Self {
        Self {
            max_new_tokens: params.max_new_tokens,
            temperature: params.temperature,
            do_sample: params.do_sample,
            top_p: params.top_p,
            stop: vec![params.stop_token.clone()],
            return_full_text: false,
        }
    }
}

#[derive(Serialize)]
struct APIRequest {
    inputs: String,
    parameters: APIParams,
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
struct Document {
    language_id: String,
    text: Rope,
}

impl Document {
    fn new(language_id: String, text: Rope) -> Self {
        Self { language_id, text }
    }
}

#[derive(Debug)]
struct Backend {
    cache_dir: PathBuf,
    client: Client,
    document_map: Arc<RwLock<HashMap<String, Document>>>,
    http_client: reqwest::Client,
    workspace_folders: Arc<RwLock<Option<Vec<WorkspaceFolder>>>>,
    language_comments: HashMap<String, LanguageComment>,
    tokenizer_map: Arc<RwLock<HashMap<String, Tokenizer>>>,
}

#[derive(Debug, Deserialize, Serialize)]
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
    tokenizer_path: Option<String>,
    context_window: usize,
}

fn internal_error<E: Display>(err: E) -> Error {
    let err_msg = err.to_string();
    error!(err_msg);
    Error {
        code: tower_lsp::jsonrpc::ErrorCode::InternalError,
        message: err_msg.into(),
        data: None,
    }
}

fn file_path_comment(
    file_url: Url,
    file_language_id: &str,
    workspace_folders: Option<&Vec<WorkspaceFolder>>,
    language_comments: &HashMap<String, LanguageComment>,
) -> String {
    let mut file_path = file_url.path().to_owned();
    let path_in_workspace = if let Some(workspace_folders) = workspace_folders {
        for workspace_folder in workspace_folders {
            let workspace_folder_path = workspace_folder.uri.path();
            if file_path.starts_with(workspace_folder_path) {
                file_path = file_path.replace(workspace_folder_path, "");
                break;
            }
        }
        file_path
    } else {
        file_path
    };
    let lc = match language_comments.get(file_language_id) {
        Some(id) => id.clone(),
        None => LanguageComment {
            open: "//".to_owned(),
            close: None,
        },
    };
    let close = if let Some(close) = lc.close {
        format!(" {close}")
    } else {
        "".to_owned()
    };
    format!("{} {path_in_workspace}{close}\n", lc.open)
}

fn build_prompt(
    pos: Position,
    text: &Rope,
    fim: &FimParams,
    file_path: String,
    tokenizer: Tokenizer,
    context_window: usize,
) -> Result<String> {
    if fim.enabled {
        let mut token_count = context_window;
        let mut before_iter = text.lines_at(pos.line as usize + 1).reversed();
        let mut after_iter = text.lines_at(pos.line as usize);
        let mut before_line = before_iter.next();
        let col = pos.character as usize;
        if let Some(line) = before_line {
            before_line = Some(line.slice(0..col));
        }
        let mut after_line = after_iter.next();
        if let Some(line) = after_line {
            after_line = Some(line.slice(col..));
        }
        let mut before = vec![];
        let mut after = String::new();
        while before_line.is_some() || after_line.is_some() {
            if let Some(before_line) = before_line {
                let before_line = before_line.to_string();
                let tokens = tokenizer
                    .encode(before_line.clone(), false)
                    .map_err(internal_error)?
                    .len();
                if tokens > token_count {
                    break;
                }
                token_count -= tokens;
                before.push(before_line);
            }
            if let Some(after_line) = after_line {
                let after_line = after_line.to_string();
                let tokens = tokenizer
                    .encode(after_line.clone(), false)
                    .map_err(internal_error)?
                    .len();
                if tokens > token_count {
                    break;
                }
                token_count -= tokens;
                after.push_str(&after_line);
            }
            before_line = before_iter.next();
            after_line = after_iter.next();
        }
        Ok(format!(
            "{}{}{}{}{}{}",
            file_path,
            fim.prefix,
            before.into_iter().rev().collect::<Vec<_>>().join(""),
            fim.suffix,
            after,
            fim.middle
        ))
    } else {
        let mut token_count = context_window;
        let mut before = vec![];
        let mut first = true;
        for mut line in text.lines_at(pos.line as usize).reversed() {
            if first {
                line = line.slice(0..pos.character as usize);
                first = false;
            }
            let line = line.to_string();
            let tokens = tokenizer
                .encode(line.clone(), false)
                .map_err(internal_error)?
                .len();
            if tokens > token_count {
                break;
            }
            token_count -= tokens;
            before.push(line);
        }
        Ok(format!(
            "{}{}",
            file_path,
            &before.into_iter().rev().collect::<Vec<_>>().join("")
        ))
    }
}

async fn request_completion(
    http_client: &reqwest::Client,
    model: &str,
    request_params: RequestParams,
    api_token: Option<&String>,
    prompt: String,
) -> Result<Vec<Generation>> {
    let mut req = http_client.post(build_url(model)).json(&APIRequest {
        inputs: prompt,
        parameters: request_params.into(),
    });

    if let Some(api_token) = api_token {
        req = req.header(AUTHORIZATION, format!("Bearer {api_token}"))
    }

    let res = req.send().await.map_err(internal_error)?;

    match res.json().await.map_err(internal_error)? {
        APIResponse::Generations(gens) => Ok(gens),
        APIResponse::Error(err) => Err(internal_error(err)),
    }
}

fn parse_generations(generations: Vec<Generation>, stop_token: &str) -> Vec<Completion> {
    generations
        .into_iter()
        .map(|g| Completion {
            generated_text: g.generated_text.replace(stop_token, ""),
        })
        .collect()
}

async fn download_tokenizer_file(
    http_client: &reqwest::Client,
    model: &str,
    api_token: Option<&String>,
    to: impl AsRef<Path>,
) -> Result<()> {
    if to.as_ref().exists() {
        return Ok(());
    }
    tokio::fs::create_dir_all(
        to.as_ref()
            .parent()
            .ok_or_else(|| internal_error("tokenizer path has no parent"))?,
    )
    .await
    .map_err(internal_error)?;
    let mut req = http_client.get(format!(
        "https://huggingface.co/{model}/resolve/main/tokenizer.json"
    ));
    if let Some(api_token) = api_token {
        req = req.header(AUTHORIZATION, format!("Bearer {api_token}"))
    }
    let res = req
        .send()
        .await
        .map_err(internal_error)?
        .error_for_status()
        .map_err(internal_error)?;
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(to)
        .await
        .map_err(internal_error)?;
    file.write_all(&res.bytes().await.map_err(internal_error)?)
        .await
        .map_err(internal_error)?;
    Ok(())
}

async fn get_tokenizer(
    model: &str,
    tokenizer_map: &mut HashMap<String, Tokenizer>,
    tokenizer_path: Option<&String>,
    http_client: &reqwest::Client,
    cache_dir: impl AsRef<Path>,
    api_token: Option<&String>,
) -> Result<Tokenizer> {
    if model.starts_with("http://") || model.starts_with("https://") {
        let tokenizer = match tokenizer_path {
            Some(path) => Tokenizer::from_file(path).map_err(internal_error)?,
            None => return Err(internal_error("`tokenizer_path` is null")),
        };
        Ok(tokenizer)
    } else {
        match tokenizer_map.get(model) {
            Some(tokenizer) => Ok(tokenizer.clone()),
            None => {
                let path = cache_dir.as_ref().join(model).join("tokenizer.json");
                download_tokenizer_file(http_client, model, api_token, &path).await?;
                let tokenizer = Tokenizer::from_file(path).map_err(internal_error)?;
                tokenizer_map.insert(model.to_owned(), tokenizer.clone());
                Ok(tokenizer)
            }
        }
    }
}

fn build_url(model: &str) -> String {
    if model.starts_with("http://") || model.starts_with("https://") {
        model.to_owned()
    } else {
        format!("https://api-inference.huggingface.co/models/{model}")
    }
}

impl Backend {
    async fn get_completions(&self, params: CompletionParams) -> Result<Vec<Completion>> {
        info!("get_completions {params:?}");
        let document_map = self.document_map.read().await;

        let document = document_map
            .get(params.text_document_position.text_document.uri.as_str())
            .ok_or_else(|| internal_error("failed to find document"))?;
        let file_path = file_path_comment(
            params.text_document_position.text_document.uri,
            &document.language_id,
            self.workspace_folders.read().await.as_ref(),
            &self.language_comments,
        );
        let tokenizer = get_tokenizer(
            &params.model,
            &mut *self.tokenizer_map.write().await,
            params.tokenizer_path.as_ref(),
            &self.http_client,
            &self.cache_dir,
            params.api_token.as_ref(),
        )
        .await?;
        let prompt = build_prompt(
            params.text_document_position.position,
            &document.text,
            &params.fim,
            file_path,
            tokenizer,
            params.context_window,
        )?;
        let stop_token = params.request_params.stop_token.clone();
        let result = request_completion(
            &self.http_client,
            &params.model,
            params.request_params,
            params.api_token.as_ref(),
            prompt,
        )
        .await?;

        Ok(parse_generations(result, &stop_token))
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, params: InitializeParams) -> Result<InitializeResult> {
        *self.workspace_folders.write().await = params.workspace_folders;
        Ok(InitializeResult {
            server_info: Some(ServerInfo {
                name: "llm-ls".to_owned(),
                version: Some("0.1.0".to_owned()),
            }),
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                ..Default::default()
            },
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "{llm-ls} initialized")
            .await;
        let _ = info!("initialized");
    }

    // TODO:
    // textDocument/didClose

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "{llm-ls} file opened")
            .await;
        let rope = ropey::Rope::from_str(&params.text_document.text);
        let uri = params.text_document.uri.to_string();
        *self
            .document_map
            .write()
            .await
            .entry(uri.clone())
            .or_insert(Document::new("unknown".to_owned(), Rope::new())) =
            Document::new(params.text_document.language_id, rope);
        info!("{uri} opened");
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "{llm-ls} file changed")
            .await;
        let rope = ropey::Rope::from_str(&params.content_changes[0].text);
        let uri = params.text_document.uri.to_string();
        let mut document_map = self.document_map.write().await;
        let doc = document_map
            .entry(uri.clone())
            .or_insert(Document::new("unknown".to_owned(), Rope::new()));
        doc.text = rope;
        info!("{uri} changed");
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "{llm-ls} file saved")
            .await;
        let uri = params.text_document.uri.to_string();
        info!("{uri} saved");
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, "{llm-ls} file closed")
            .await;
        let uri = params.text_document.uri.to_string();
        info!("{uri} closed");
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

    let home_dir = home::home_dir().ok_or(()).expect("failed to find home dir");
    let cache_dir = home_dir.join(".cache/llm_ls");
    tokio::fs::create_dir_all(&cache_dir)
        .await
        .expect("failed to create cache dir");

    let log_file = rolling::never(&cache_dir, "llm-ls.log");
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
        cache_dir,
        client,
        document_map: Arc::new(RwLock::new(HashMap::new())),
        http_client,
        workspace_folders: Arc::new(RwLock::new(None)),
        language_comments: build_language_comments(),
        tokenizer_map: Arc::new(RwLock::new(HashMap::new())),
    })
    .custom_method("llm-ls/getCompletions", Backend::get_completions)
    .finish();

    Server::new(stdin, stdout, socket).serve(service).await;
}
