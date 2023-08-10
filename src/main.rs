use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;
use tower_lsp::jsonrpc::{Error, Result};
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

#[derive(Serialize)]
struct RequestParams {
    max_new_tokens: u32,
    temperature: f32,
    do_sample: bool,
    top_p: f32,
    stop_token: String,
}

#[derive(Serialize)]
struct APIRequest {
    inputs: String,
    parameters: RequestParams,
}

#[derive(Deserialize)]
struct APIResponse {
    generated_text: String,
}

#[derive(Debug)]
struct Backend {
    client: Client,
    http_client: reqwest::Client,
}

fn internal_error<E: Display>(err: E) -> Error {
    Error {
        code: tower_lsp::jsonrpc::ErrorCode::InternalError,
        message: err.to_string(),
        data: None,
    }
}

fn get_cache_dir_path() -> Result<PathBuf> {
    let home_dir = home::home_dir().ok_or(internal_error("Failed to find home dir"))?;
    Ok(home_dir.join(".cache/ccserver"))
}

async fn request_completion(http_client: &reqwest::Client) -> Result<Vec<APIResponse>> {
    http_client
        .post("https://api-inference.huggingface.co/models/bigcode/starcoder")
        .json(&APIRequest {
            inputs: "Hello my name is ".to_owned(),
            parameters: RequestParams {
                max_new_tokens: 60,
                temperature: 0.2,
                do_sample: true,
                top_p: 0.95,
                stop_token: "\n".to_owned(),
            },
        })
        .send()
        .await
        .map_err(internal_error)?
        .json()
        .await
        .map_err(internal_error)?
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        tokio::fs::create_dir_all(get_cache_dir_path()?)
            .await
            .map_err(internal_error)?;
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
            .log_message(MessageType::INFO, "{ccserver} initialized")
            .await;
        if let Ok(cache_dir) = get_cache_dir_path() {
            tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(cache_dir.join("ccserver.log"))
                .await
                .unwrap()
                .write_all(b"initialized\n")
                .await
                .unwrap();
        }
    }

    // XXX: tbd if we use code action or completion
    async fn completion(&self, _: CompletionParams) -> Result<Option<CompletionResponse>> {
        let result = request_completion(&self.http_client).await?;
        if result.len() > 0 {
            let generated_text = result[0].generated_text.clone();

            tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(get_cache_dir_path()?.join("ccserver.log"))
                .await
                .unwrap()
                .write_all(format!("completion request: {generated_text}\n").as_bytes())
                .await
                .unwrap();

            Ok(Some(CompletionResponse::Array(vec![CompletionItem {
                label: "ccserver completion".to_owned(),
                insert_text: Some(generated_text.clone()),
                kind: Some(CompletionItemKind::TEXT),
                detail: Some(generated_text),
                ..Default::default()
            }])))
        } else {
            Ok(None)
        }
    }

    async fn shutdown(&self) -> Result<()> {
        tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(get_cache_dir_path()?.join("ccserver.log"))
            .await
            .unwrap()
            .write_all(b"shutdown\n")
            .await
            .unwrap();
        Ok(())
    }
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let http_client = reqwest::Client::new();

    let (service, socket) = LspService::new(|client| Backend {
        client,
        http_client,
    });
    Server::new(stdin, stdout, socket).serve(service).await;
}
