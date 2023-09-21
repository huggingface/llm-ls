use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, USER_AGENT};
use ropey::Rope;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::{Error, Result};
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use tracing::{debug, error, info};
use tracing_appender::rolling;
use tracing_subscriber::EnvFilter;

const NAME: &str = "llm-ls";
const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Deserialize, Serialize)]
enum TokenizerConfig {
    Local { path: PathBuf },
    HuggingFace { repository: String },
    Download { url: String, to: PathBuf },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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
    Generation(Generation),
    Generations(Vec<Generation>),
    Error(APIError),
}

#[derive(Debug)]
struct Document {
    #[allow(dead_code)]
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
    unsafe_http_client: reqwest::Client,
    workspace_folders: Arc<RwLock<Option<Vec<WorkspaceFolder>>>>,
    tokenizer_map: Arc<RwLock<HashMap<String, Arc<Tokenizer>>>>,
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
        info!(build_prompt_ms = time, "built prompt in {time} ms");
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
        info!(build_prompt_ms = time, "built prompt in {time} ms");
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

    match res.json().await.map_err(internal_error)? {
        APIResponse::Generation(gen) => Ok(vec![gen]),
        APIResponse::Generations(gens) => Ok(gens),
        APIResponse::Error(err) => Err(internal_error(err)),
    }
}

fn parse_generations(generations: Vec<Generation>, tokens_to_clear: &[String]) -> Vec<Completion> {
    generations
        .into_iter()
        .map(|g| {
            let mut generated_text = g.generated_text;
            for token in tokens_to_clear {
                generated_text = generated_text.replace(token, "")
            }
            Completion { generated_text }
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
            .ok_or_else(|| internal_error("tokenizer path has no parent"))?,
    )
    .await
    .map_err(internal_error)?;
    let res = http_client
        .get(url)
        .headers(build_headers(api_token, ide)?)
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
            TokenizerConfig::Local { path } => {
                Arc::new(Tokenizer::from_file(path).map_err(internal_error)?)
            }
            TokenizerConfig::HuggingFace { repository } => {
                let path = cache_dir.as_ref().join(model).join("tokenizer.json");
                let url =
                    format!("https://huggingface.co/{repository}/resolve/main/tokenizer.json");
                download_tokenizer_file(http_client, &url, api_token, &path, ide).await?;
                Arc::new(Tokenizer::from_file(path).map_err(internal_error)?)
            }
            TokenizerConfig::Download { url, to } => {
                download_tokenizer_file(http_client, &url, api_token, &to, ide).await?;
                Arc::new(Tokenizer::from_file(to).map_err(internal_error)?)
            }
        };
        tokenizer_map.insert(model.to_owned(), tokenizer.clone());
        Ok(Some(tokenizer))
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
    async fn get_completions(&self, params: CompletionParams) -> Result<Vec<Completion>> {
        info!("get_completions {params:?}");
        let document_map = self.document_map.read().await;

        let document = document_map
            .get(params.text_document_position.text_document.uri.as_str())
            .ok_or_else(|| internal_error("failed to find document"))?;
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

        Ok(parse_generations(result, &params.tokens_to_clear))
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
            .log_message(MessageType::INFO, "{llm-ls} initialized")
            .await;
        let _ = info!("initialized language server");
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
        let _ = debug!("shutdown");
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
    })
    .custom_method("llm-ls/getCompletions", Backend::get_completions)
    .finish();

    Server::new(stdin, stdout, socket).serve(service).await;
}
