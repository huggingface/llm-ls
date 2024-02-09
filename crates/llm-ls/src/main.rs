use clap::Parser;
use custom_types::llm_ls::{
    AcceptCompletionParams, Backend, Completion, FimParams, GetCompletionsParams,
    GetCompletionsResult, Ide, TokenizerConfig,
};
use ropey::Rope;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result as LspResult;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use tracing::{debug, error, info, info_span, warn, Instrument};
use tracing_appender::rolling;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use crate::backend::{build_body, build_headers, parse_generations};
use crate::document::Document;
use crate::error::{internal_error, Error, Result};

mod backend;
mod document;
mod error;
mod language_id;

const MAX_WARNING_REPEAT: Duration = Duration::from_secs(3_600);
pub const NAME: &str = "llm-ls";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

fn get_position_idx(rope: &Rope, row: usize, col: usize) -> Result<usize> {
    Ok(rope.try_line_to_char(row)?
        + col.min(
            rope.get_line(row.min(rope.len_lines().saturating_sub(1)))
                .ok_or(Error::OutOfBoundLine(row))?
                .len_chars()
                .saturating_sub(1),
        ))
}

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

fn should_complete(document: &Document, position: Position) -> Result<CompletionType> {
    let row = position.line as usize;
    let column = position.character as usize;
    if document.text.len_chars() == 0 {
        warn!("Document is empty");
        return Ok(CompletionType::Empty);
    }
    if let Some(tree) = &document.tree {
        let current_node = tree.root_node().descendant_for_point_range(
            tree_sitter::Point { row, column },
            tree_sitter::Point {
                row,
                column: column + 1,
            },
        );
        if let Some(node) = current_node {
            if node == tree.root_node() {
                return Ok(CompletionType::MultiLine);
            }
            let start = node.start_position();
            let end = node.end_position();
            let mut start_offset = get_position_idx(&document.text, start.row, start.column)?;
            let mut end_offset = get_position_idx(&document.text, end.row, end.column)? - 1;
            let start_char = document
                .text
                .get_char(start_offset.min(document.text.len_chars().saturating_sub(1)))
                .ok_or(Error::OutOfBoundIndexing(start_offset))?;
            let end_char = document
                .text
                .get_char(end_offset.min(document.text.len_chars().saturating_sub(1)))
                .ok_or(Error::OutOfBoundIndexing(end_offset))?;
            if !start_char.is_whitespace() {
                start_offset += 1;
            }
            if !end_char.is_whitespace() {
                end_offset -= 1;
            }
            if start_offset >= end_offset {
                return Ok(CompletionType::SingleLine);
            }
            let slice = document
                .text
                .get_slice(start_offset..end_offset)
                .ok_or(Error::OutOfBoundSlice(start_offset, end_offset))?;
            if slice.to_string().trim().is_empty() {
                return Ok(CompletionType::MultiLine);
            }
        }
    }
    let start_idx = document.text.try_line_to_char(row)?;
    // XXX: We treat the end of a document as a newline
    let next_char = document.text.get_char(start_idx + column).unwrap_or('\n');
    if next_char.is_whitespace() {
        Ok(CompletionType::SingleLine)
    } else {
        Ok(CompletionType::Empty)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestParams {
    max_new_tokens: u32,
    temperature: f32,
    do_sample: bool,
    top_p: f32,
    stop_tokens: Option<Vec<String>>,
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
pub struct Generation {
    generated_text: String,
}

#[derive(Debug, Deserialize)]
pub struct APIError {
    error: String,
}

impl std::error::Error for APIError {
    fn description(&self) -> &str {
        &self.error
    }
}

impl Display for APIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum APIResponse {
    Generation(Generation),
    Generations(Vec<Generation>),
    Error(APIError),
}

struct LlmService {
    cache_dir: PathBuf,
    client: Client,
    document_map: Arc<RwLock<HashMap<String, Document>>>,
    http_client: reqwest::Client,
    unsafe_http_client: reqwest::Client,
    workspace_folders: Arc<RwLock<Option<Vec<WorkspaceFolder>>>>,
    tokenizer_map: Arc<RwLock<HashMap<String, Arc<Tokenizer>>>>,
    unauthenticated_warn_at: Arc<RwLock<Instant>>,
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
        let mut remaining_token_count = context_window - 3; // account for FIM tokens
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
                    tokenizer.encode(before_line.clone(), false)?.len()
                } else {
                    before_line.len()
                };
                if tokens > remaining_token_count {
                    break;
                }
                remaining_token_count -= tokens;
                before.push(before_line);
            }
            if let Some(after_line) = after_line {
                let after_line = after_line.to_string();
                let tokens = if let Some(tokenizer) = tokenizer.clone() {
                    tokenizer.encode(after_line.clone(), false)?.len()
                } else {
                    after_line.len()
                };
                if tokens > remaining_token_count {
                    break;
                }
                remaining_token_count -= tokens;
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
        let mut remaining_token_count = context_window;
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
                tokenizer.encode(line.clone(), false)?.len()
            } else {
                line.len()
            };
            if tokens > remaining_token_count {
                break;
            }
            remaining_token_count -= tokens;
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
    prompt: String,
    params: &GetCompletionsParams,
) -> Result<Vec<Generation>> {
    let t = Instant::now();

    let json = build_body(
        &params.backend,
        params.model.clone(),
        prompt,
        params.request_body.clone(),
    );
    let headers = build_headers(&params.backend, params.api_token.as_ref(), params.ide)?;
    let res = http_client
        .post(build_url(params.backend.clone(), &params.model))
        .json(&json)
        .headers(headers)
        .send()
        .await?;

    let model = &params.model;
    let generations = parse_generations(&params.backend, res.text().await?.as_str())?;
    let time = t.elapsed().as_millis();
    info!(
        model,
        compute_generations_ms = time,
        generations = serde_json::to_string(&generations)?,
        "{model} computed generations in {time} ms"
    );
    Ok(generations)
}

fn format_generations(
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
    tokio::fs::create_dir_all(to.as_ref().parent().ok_or(Error::InvalidTokenizerPath)?).await?;
    let headers = build_headers(&Backend::default(), api_token, ide)?;
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(to)
        .await?;
    let http_client = http_client.clone();
    let url = url.to_owned();
    // TODO:
    // - create oneshot channel to send result of tokenizer download to display error message
    // to user?
    // - retry logic?
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
    .await?;
    Ok(())
}

async fn get_tokenizer(
    model: &str,
    tokenizer_map: &mut HashMap<String, Arc<Tokenizer>>,
    tokenizer_config: Option<&TokenizerConfig>,
    http_client: &reqwest::Client,
    cache_dir: impl AsRef<Path>,
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
            TokenizerConfig::HuggingFace {
                repository,
                api_token,
            } => {
                let (org, repo) = repository
                    .split_once('/')
                    .ok_or(Error::InvalidRepositoryId)?;
                let path = cache_dir
                    .as_ref()
                    .join(org)
                    .join(repo)
                    .join("tokenizer.json");
                let url =
                    format!("https://huggingface.co/{repository}/resolve/main/tokenizer.json");
                download_tokenizer_file(http_client, &url, api_token.as_ref(), &path, ide).await?;
                match Tokenizer::from_file(path) {
                    Ok(tokenizer) => Some(Arc::new(tokenizer)),
                    Err(err) => {
                        error!("error loading tokenizer from file: {err}");
                        None
                    }
                }
            }
            TokenizerConfig::Download { url, to } => {
                download_tokenizer_file(http_client, url, None, &to, ide).await?;
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

fn build_url(backend: Backend, model: &str) -> String {
    match backend {
        Backend::HuggingFace { url } => format!("{url}/models/{model}"),
        Backend::Ollama { url } => url,
        Backend::OpenAi { url } => url,
        Backend::Tgi { url } => url,
    }
}

impl LlmService {
    async fn get_completions(
        &self,
        params: GetCompletionsParams,
    ) -> LspResult<GetCompletionsResult> {
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
                backend = ?params.backend,
                ide = %params.ide,
                request_body = serde_json::to_string(&params.request_body).map_err(internal_error)?,
                "received completion request for {}",
                params.text_document_position.text_document.uri
            );
            if params.api_token.is_none() && params.backend.is_using_inference_api() {
                let now = Instant::now();
                let unauthenticated_warn_at = self.unauthenticated_warn_at.read().await;
                if now.duration_since(*unauthenticated_warn_at) > MAX_WARNING_REPEAT {
                    drop(unauthenticated_warn_at);
                    self.client.show_message(MessageType::WARNING, "You are currently unauthenticated and will get rate limited. To reduce rate limiting, login with your API Token and consider subscribing to PRO: https://huggingface.co/pricing#pro").await;
                    let mut unauthenticated_warn_at = self.unauthenticated_warn_at.write().await;
                    *unauthenticated_warn_at = Instant::now();
                }
            }
            let completion_type = should_complete(document, params.text_document_position.position)?;
            info!(%completion_type, "completion type: {completion_type:?}");
            if completion_type == CompletionType::Empty {
                return Ok(GetCompletionsResult { request_id, completions: vec![]});
            }

            let tokenizer = get_tokenizer(
                &params.model,
                &mut *self.tokenizer_map.write().await,
                params.tokenizer_config.as_ref(),
                &self.http_client,
                &self.cache_dir,
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
                prompt,
                &params,
            )
            .await?;

            let completions = format_generations(result, &params.tokens_to_clear, completion_type);
            Ok(GetCompletionsResult { request_id, completions })
        }.instrument(span).await
    }

    async fn accept_completion(&self, accepted: AcceptCompletionParams) -> LspResult<()> {
        info!(
            request_id = %accepted.request_id,
            accepted_position = accepted.accepted_completion,
            shown_completions = serde_json::to_string(&accepted.shown_completions).map_err(internal_error)?,
            "accepted completion"
        );
        Ok(())
    }

    async fn reject_completion(&self, rejected: AcceptCompletionParams) -> LspResult<()> {
        info!(
            request_id = %rejected.request_id,
            shown_completions = serde_json::to_string(&rejected.shown_completions).map_err(internal_error)?,
            "rejected completion"
        );
        Ok(())
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for LlmService {
    async fn initialize(&self, params: InitializeParams) -> LspResult<InitializeResult> {
        *self.workspace_folders.write().await = params.workspace_folders;
        Ok(InitializeResult {
            server_info: Some(ServerInfo {
                name: "llm-ls".to_owned(),
                version: Some(VERSION.to_owned()),
            }),
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::INCREMENTAL,
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
        if params.content_changes.is_empty() {
            return;
        }

        // ignore the output scheme
        if params.text_document.uri.scheme() == "output" {
            return;
        }

        let mut document_map = self.document_map.write().await;
        self.client
            .log_message(MessageType::LOG, format!("{uri} changed"))
            .await;
        let doc = document_map.get_mut(&uri);
        if let Some(doc) = doc {
            for change in &params.content_changes {
                if let Some(range) = change.range {
                    match doc.change(range, &change.text).await {
                        Ok(()) => info!("{uri} changed"),
                        Err(err) => error!("error when changing {uri}: {err}"),
                    }
                } else {
                    warn!("Could not update document, got change request with missing range");
                }
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

    async fn shutdown(&self) -> LspResult<()> {
        debug!("shutdown");
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Wether to use a tcp socket for data transfer
    #[arg(long = "port")]
    socket: Option<usize>,

    /// Wether to use stdio transport for data transfer, ignored because it is the default
    /// behaviour
    #[arg(short, long, default_value_t = true)]
    stdio: bool,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

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

    let (service, socket) = LspService::build(|client| LlmService {
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
    .custom_method("llm-ls/getCompletions", LlmService::get_completions)
    .custom_method("llm-ls/acceptCompletion", LlmService::accept_completion)
    .custom_method("llm-ls/rejectCompletion", LlmService::reject_completion)
    .finish();

    if let Some(port) = args.socket {
        let addr = format!("127.0.0.1:{port}");
        let listener = TcpListener::bind(&addr)
            .await
            .unwrap_or_else(|_| panic!("failed to bind tcp listener to {addr}"));
        let (stream, _) = listener
            .accept()
            .await
            .unwrap_or_else(|_| panic!("failed to accept new connections on {addr}"));
        let (read, write) = tokio::io::split(stream);
        Server::new(read, write, socket).serve(service).await;
    } else {
        let (stdin, stdout) = (tokio::io::stdin(), tokio::io::stdout());
        Server::new(stdin, stdout, socket).serve(service).await;
    }
}
