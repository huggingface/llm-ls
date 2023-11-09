use document::Document;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, USER_AGENT};
use ropey::Rope;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::{Error, Result};
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use tracing::{debug, error, info, info_span, warn, Instrument};
use tracing_appender::rolling;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

mod document;
mod language_id;

const MAX_WARNING_REPEAT: Duration = Duration::from_secs(3_600);
const NAME: &str = "llm-ls";
const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
enum CompletionType {
    Empty,
    SingleLine,
    MultiLine,
}

impl Display for CompletionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompletionType::Empty => write!(f, "empty"),
            CompletionType::SingleLine => write!(f, "single_line"),
            CompletionType::MultiLine => write!(f, "multi_line"),
        }
    }
}

fn should_complete(document: &Document, position: Position) -> CompletionType {
    let row = position.line as usize;
    let column = position.character as usize;
    if let Some(tree) = &document.tree {
        let current_node = tree.root_node().descendant_for_point_range(
            tree_sitter::Point { row, column },
            tree_sitter::Point { row, column },
        );
        if let Some(node) = current_node {
            if node == tree.root_node() {
                return CompletionType::MultiLine;
            }
            let start = node.start_position();
            let end = node.end_position();
            let mut start_offset = document.text.line_to_char(start.row) + start.column;
            let mut end_offset = document.text.line_to_char(end.row) + end.column - 1;
            let start_char = document.text.char(start_offset);
            if !start_char.is_whitespace() {
                start_offset += 1;
            }
            let end_char = document.text.char(end_offset);
            if !end_char.is_whitespace() {
                end_offset -= 1;
            }
            if start_offset >= end_offset {
                return CompletionType::SingleLine;
            }
            let slice = document.text.slice(start_offset..end_offset);
            if slice.to_string().trim().is_empty() {
                return CompletionType::MultiLine;
            }
        }
    }
    let start_idx = document.text.line_to_char(row);
    let next_char = document.text.char(start_idx + column);
    if next_char.is_whitespace() {
        CompletionType::SingleLine
    } else {
        CompletionType::Empty
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum TokenizerConfig {
    Local { path: PathBuf },
    HuggingFace { repository: String },
    Download { url: String, to: PathBuf },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct RequestParams {
    max_new_tokens: u32,
    temperature: f32,
    do_sample: bool,
    top_p: f32,
    stop_tokens: Option<Vec<String>>,
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
    #[allow(dead_code)]
    #[serde(skip_serializing)]
    stop: Option<Vec<String>>,
    return_full_text: bool,
}

impl From<RequestParams> for APIParams {
    fn from(params: RequestParams) -> Self {
        Self {
            max_new_tokens: params.max_new_tokens,
            temperature: params.temperature,
            do_sample: params.do_sample,
            top_p: params.top_p,
            stop: params.stop_tokens,
            return_full_text: false,
        }
    }
}

#[derive(Serialize)]
struct APIRequest {
    inputs: String,
    parameters: APIParams,
}

#[derive(Debug, Serialize, Deserialize)]
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
    Generation(Generation),
    Generations(Vec<Generation>),
    Error(APIError),
}

struct Backend {
    cache_dir: PathBuf,
    client: Client,
    document_map: Arc<RwLock<HashMap<String, Document>>>,
    http_client: reqwest::Client,
    unsafe_http_client: reqwest::Client,
    workspace_folders: Arc<RwLock<Option<Vec<WorkspaceFolder>>>>,
    tokenizer_map: Arc<RwLock<HashMap<String, Arc<Tokenizer>>>>,
    unauthenticated_warn_at: Arc<RwLock<Instant>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Completion {
    generated_text: String,
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

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct AcceptedCompletion {
    request_id: Uuid,
    accepted_completion: u32,
    shown_completions: Vec<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct RejectedCompletion {
    request_id: Uuid,
    shown_completions: Vec<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompletionParams {
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

#[derive(Debug, Deserialize, Serialize)]
struct CompletionResult {
    request_id: Uuid,
    completions: Vec<Completion>,
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

fn build_prompt(
    pos: Position,
    text: &Rope,
    fim: &FimParams,
    tokenizer: Option<Arc<Tokenizer>>,
    context_window: usize,
) -> Result<String> {
    let t = Instant::now();
    if fim.enabled {
        let mut token_count = context_window;
        let mut before_iter = text.lines_at(pos.line as usize + 1).reversed();
        let mut after_iter = text.lines_at(pos.line as usize);
        let mut before_line = before_iter.next();
        if let Some(line) = before_line {
            let col = (pos.character as usize).clamp(0, line.len_chars());
            before_line = Some(line.slice(0..col));
        }
        let mut after_line = after_iter.next();
        if let Some(line) = after_line {
            let col = (pos.character as usize).clamp(0, line.len_chars());
            after_line = Some(line.slice(col..));
        }
        let mut before = vec![];
        let mut after = String::new();
        while before_line.is_some() || after_line.is_some() {
            if let Some(before_line) = before_line {
                let before_line = before_line.to_string();
                let tokens = if let Some(tokenizer) = tokenizer.clone() {
                    tokenizer
                        .encode(before_line.clone(), false)
                        .map_err(internal_error)?
                        .len()
                } else {
                    before_line.len()
                };
                if tokens > token_count {
                    break;
                }
                token_count -= tokens;
                before.push(before_line);
            }
            if let Some(after_line) = after_line {
                let after_line = after_line.to_string();
                let tokens = if let Some(tokenizer) = tokenizer.clone() {
                    tokenizer
                        .encode(after_line.clone(), false)
                        .map_err(internal_error)?
                        .len()
                } else {
                    after_line.len()
                };
                if tokens > token_count {
                    break;
                }
                token_count -= tokens;
                after.push_str(&after_line);
            }
            before_line = before_iter.next();
            after_line = after_iter.next();
        }
        let prompt = format!(
            "{}{}{}{}{}",
            fim.prefix,
            before.into_iter().rev().collect::<Vec<_>>().join(""),
            fim.suffix,
            after,
            fim.middle
        );
        let time = t.elapsed().as_millis();
        info!(prompt, build_prompt_ms = time, "built prompt in {time} ms");
        Ok(prompt)
    } else {
        let mut token_count = context_window;
        let mut before = vec![];
        let mut first = true;
        for mut line in text.lines_at(pos.line as usize + 1).reversed() {
            if first {
                let col = (pos.character as usize).clamp(0, line.len_chars());
                line = line.slice(0..col);
                first = false;
            }
            let line = line.to_string();
            let tokens = if let Some(tokenizer) = tokenizer.clone() {
                tokenizer
                    .encode(line.clone(), false)
                    .map_err(internal_error)?
                    .len()
            } else {
                line.len()
            };
            if tokens > token_count {
                break;
            }
            token_count -= tokens;
            before.push(line);
        }
        let prompt = before.into_iter().rev().collect::<Vec<_>>().join("");
        let time = t.elapsed().as_millis();
        info!(prompt, build_prompt_ms = time, "built prompt in {time} ms");
        Ok(prompt)
    }
}

async fn request_completion(
    http_client: &reqwest::Client,
    ide: Ide,
    model: &str,
    request_params: RequestParams,
    api_token: Option<&String>,
    prompt: String,
) -> Result<Vec<Generation>> {
    let t = Instant::now();
    let res = http_client
        .post(build_url(model))
        .json(&APIRequest {
            inputs: prompt,
            parameters: request_params.into(),
        })
        .headers(build_headers(api_token, ide)?)
        .send()
        .await
        .map_err(internal_error)?;

    let generations = match res.json().await.map_err(internal_error)? {
        APIResponse::Generation(gen) => vec![gen],
        APIResponse::Generations(gens) => gens,
        APIResponse::Error(err) => return Err(internal_error(err)),
    };
    let time = t.elapsed().as_millis();
    info!(
        model,
        compute_generations_ms = time,
        generations = serde_json::to_string(&generations).map_err(internal_error)?,
        "{model} computed generations in {time} ms"
    );
    Ok(generations)
}

fn parse_generations(
    generations: Vec<Generation>,
    tokens_to_clear: &[String],
    completion_type: CompletionType,
) -> Vec<Completion> {
    generations
        .into_iter()
        .map(|g| {
            let mut generated_text = g.generated_text;
            for token in tokens_to_clear {
                generated_text = generated_text.replace(token, "")
            }
            match completion_type {
                CompletionType::Empty => {
                    warn!("completion type should not be empty when post processing completions");
                    Completion { generated_text }
                }
                CompletionType::SingleLine => Completion {
                    generated_text: generated_text
                        .split_once('\n')
                        .unwrap_or((&generated_text, ""))
                        .0
                        .to_owned(),
                },
                CompletionType::MultiLine => Completion { generated_text },
            }
        })
        .collect()
}

async fn download_tokenizer_file(
    http_client: &reqwest::Client,
    url: &str,
    api_token: Option<&String>,
    to: impl AsRef<Path>,
    ide: Ide,
) -> Result<()> {
    if to.as_ref().exists() {
        return Ok(());
    }
    tokio::fs::create_dir_all(
        to.as_ref()
            .parent()
            .ok_or_else(|| internal_error("invalid tokenizer path"))?,
    )
    .await
    .map_err(internal_error)?;
    let headers = build_headers(api_token, ide)?;
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(to)
        .await
        .map_err(internal_error)?;
    let http_client = http_client.clone();
    let url = url.to_owned();
    tokio::spawn(async move {
        let res = match http_client.get(url).headers(headers).send().await {
            Ok(res) => res,
            Err(err) => {
                error!("error sending download request for the tokenzier file: {err}");
                return;
            }
        };
        let res = match res.error_for_status() {
            Ok(res) => res,
            Err(err) => {
                error!("API replied with error to the tokenizer file download: {err}");
                return;
            }
        };
        let bytes = match res.bytes().await {
            Ok(bytes) => bytes,
            Err(err) => {
                error!("error while streaming tokenizer file bytes: {err}");
                return;
            }
        };
        match file.write_all(&bytes).await {
            Ok(_) => (),
            Err(err) => {
                error!("error writing the tokenizer file to disk: {err}");
            }
        };
    })
    .await
    .map_err(internal_error)?;
    Ok(())
}

async fn get_tokenizer(
    model: &str,
    tokenizer_map: &mut HashMap<String, Arc<Tokenizer>>,
    tokenizer_config: Option<TokenizerConfig>,
    http_client: &reqwest::Client,
    cache_dir: impl AsRef<Path>,
    api_token: Option<&String>,
    ide: Ide,
) -> Result<Option<Arc<Tokenizer>>> {
    if let Some(tokenizer) = tokenizer_map.get(model) {
        return Ok(Some(tokenizer.clone()));
    }
    if let Some(config) = tokenizer_config {
        let tokenizer = match config {
            TokenizerConfig::Local { path } => match Tokenizer::from_file(path) {
                Ok(tokenizer) => Some(Arc::new(tokenizer)),
                Err(err) => {
                    error!("error loading tokenizer from file: {err}");
                    None
                }
            },
            TokenizerConfig::HuggingFace { repository } => {
                let path = cache_dir.as_ref().join(model).join("tokenizer.json");
                let url =
                    format!("https://huggingface.co/{repository}/resolve/main/tokenizer.json");
                download_tokenizer_file(http_client, &url, api_token, &path, ide).await?;
                match Tokenizer::from_file(path) {
                    Ok(tokenizer) => Some(Arc::new(tokenizer)),
                    Err(err) => {
                        error!("error loading tokenizer from file: {err}");
                        None
                    }
                }
            }
            TokenizerConfig::Download { url, to } => {
                download_tokenizer_file(http_client, &url, api_token, &to, ide).await?;
                match Tokenizer::from_file(to) {
                    Ok(tokenizer) => Some(Arc::new(tokenizer)),
                    Err(err) => {
                        error!("error loading tokenizer from file: {err}");
                        None
                    }
                }
            }
        };
        if let Some(tokenizer) = tokenizer.clone() {
            tokenizer_map.insert(model.to_owned(), tokenizer.clone());
        }
        Ok(tokenizer)
    } else {
        Ok(None)
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
    async fn get_completions(&self, params: CompletionParams) -> Result<CompletionResult> {
        let request_id = Uuid::new_v4();
        let span = info_span!("completion_request", %request_id);
        async move {
            let document_map = self.document_map.read().await;

            let document = document_map
                .get(params.text_document_position.text_document.uri.as_str())
                .ok_or_else(|| internal_error("failed to find document"))?;
            info!(
                document_url = %params.text_document_position.text_document.uri,
                cursor_line = ?params.text_document_position.position.line,
                cursor_character = ?params.text_document_position.position.character,
                language_id = %document.language_id,
                model = params.model,
                ide = %params.ide,
                max_new_tokens = params.request_params.max_new_tokens,
                temperature = params.request_params.temperature,
                do_sample = params.request_params.do_sample,
                top_p = params.request_params.top_p,
                stop_tokens = ?params.request_params.stop_tokens,
                "received completion request for {}",
                params.text_document_position.text_document.uri
            );
            if params.api_token.is_none() {
                let now = Instant::now();
                let unauthenticated_warn_at = self.unauthenticated_warn_at.read().await;
                if now.duration_since(*unauthenticated_warn_at) > MAX_WARNING_REPEAT {
                    drop(unauthenticated_warn_at);
                    self.client.show_message(MessageType::WARNING, "You are currently unauthenticated and will get rate limited. To reduce rate limiting, login with your API Token and consider subscribing to PRO: https://huggingface.co/pricing#pro").await;
                    let mut unauthenticated_warn_at = self.unauthenticated_warn_at.write().await;
                    *unauthenticated_warn_at = Instant::now();
                }
            }
            let completion_type = should_complete(document, params.text_document_position.position);
            info!(%completion_type, "completion type: {completion_type:?}");
            if completion_type == CompletionType::Empty {
                return Ok(CompletionResult { request_id, completions: vec![]});
            }

            let tokenizer = get_tokenizer(
                &params.model,
                &mut *self.tokenizer_map.write().await,
                params.tokenizer_config,
                &self.http_client,
                &self.cache_dir,
                params.api_token.as_ref(),
                params.ide,
            )
            .await?;
            let prompt = build_prompt(
                params.text_document_position.position,
                &document.text,
                &params.fim,
                tokenizer,
                params.context_window,
            )?;

            let http_client = if params.tls_skip_verify_insecure {
                info!("tls verification is disabled");
                &self.unsafe_http_client
            } else {
                &self.http_client
            };
            let result = request_completion(
                http_client,
                params.ide,
                &params.model,
                params.request_params,
                params.api_token.as_ref(),
                prompt,
            )
            .await?;

            let completions = parse_generations(result, &params.tokens_to_clear, completion_type);
            Ok(CompletionResult { request_id, completions })
        }.instrument(span).await
    }

    async fn accept_completion(&self, accepted: AcceptedCompletion) -> Result<()> {
        info!(
            request_id = %accepted.request_id,
            accepted_position = accepted.accepted_completion,
            shown_completions = serde_json::to_string(&accepted.shown_completions).map_err(internal_error)?,
            "accepted completion"
        );
        Ok(())
    }

    async fn reject_completion(&self, rejected: RejectedCompletion) -> Result<()> {
        info!(
            request_id = %rejected.request_id,
            shown_completions = serde_json::to_string(&rejected.shown_completions).map_err(internal_error)?,
            "rejected completion"
        );
        Ok(())
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, params: InitializeParams) -> Result<InitializeResult> {
        *self.workspace_folders.write().await = params.workspace_folders;
        Ok(InitializeResult {
            server_info: Some(ServerInfo {
                name: "llm-ls".to_owned(),
                version: Some(VERSION.to_owned()),
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
            .log_message(MessageType::INFO, "llm-ls initialized")
            .await;
        info!("initialized language server");
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri.to_string();
        match Document::open(
            &params.text_document.language_id,
            &params.text_document.text,
        )
        .await
        {
            Ok(document) => {
                self.document_map
                    .write()
                    .await
                    .insert(uri.clone(), document);
                info!("{uri} opened");
            }
            Err(err) => error!("error opening {uri}: {err}"),
        }
        self.client
            .log_message(MessageType::INFO, format!("{uri} opened"))
            .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri.to_string();
        self.client
            .log_message(MessageType::INFO, format!("{uri} changed"))
            .await;
        let mut document_map = self.document_map.write().await;
        let doc = document_map.get_mut(&uri);
        if let Some(doc) = doc {
            match doc.change(&params.content_changes[0].text).await {
                Ok(()) => info!("{uri} changed"),
                Err(err) => error!("error when changing {uri}: {err}"),
            }
        } else {
            warn!("textDocument/didChange {uri}: document not found");
        }
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        let uri = params.text_document.uri.to_string();
        self.client
            .log_message(MessageType::INFO, format!("{uri} saved"))
            .await;
        info!("{uri} saved");
    }

    // TODO:
    // textDocument/didClose
    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri.to_string();
        self.client
            .log_message(MessageType::INFO, format!("{uri} closed"))
            .await;
        info!("{uri} closed");
    }

    async fn shutdown(&self) -> Result<()> {
        debug!("shutdown");
        Ok(())
    }
}

fn build_headers(api_token: Option<&String>, ide: Ide) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    let user_agent = format!("{NAME}/{VERSION}; rust/unknown; ide/{ide:?}");
    headers.insert(
        USER_AGENT,
        HeaderValue::from_str(&user_agent).map_err(internal_error)?,
    );

    if let Some(api_token) = api_token {
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_token}")).map_err(internal_error)?,
        );
    }

    Ok(headers)
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
            EnvFilter::try_from_env("LLM_LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("warn")),
        );

    builder
        .json()
        .flatten_event(true)
        .with_current_span(false)
        .with_span_list(true)
        .init();

    let http_client = reqwest::Client::new();
    let unsafe_http_client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .expect("failed to build reqwest unsafe client");

    let (service, socket) = LspService::build(|client| Backend {
        cache_dir,
        client,
        document_map: Arc::new(RwLock::new(HashMap::new())),
        http_client,
        unsafe_http_client,
        workspace_folders: Arc::new(RwLock::new(None)),
        tokenizer_map: Arc::new(RwLock::new(HashMap::new())),
        unauthenticated_warn_at: Arc::new(RwLock::new(
            Instant::now()
                .checked_sub(MAX_WARNING_REPEAT)
                .expect("instant to be in bounds"),
        )),
    })
    .custom_method("llm-ls/getCompletions", Backend::get_completions)
    .custom_method("llm-ls/acceptCompletion", Backend::accept_completion)
    .custom_method("llm-ls/rejectCompletion", Backend::reject_completion)
    .finish();

    Server::new(stdin, stdout, socket).serve(service).await;
}
