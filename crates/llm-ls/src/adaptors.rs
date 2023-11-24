use super::{
    internal_error, APIError, APIResponse, CompletionParams, Generation, Ide, NAME, VERSION,
};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, USER_AGENT};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tower_lsp::jsonrpc;

fn build_tgi_body(prompt: String, params: CompletionParams) -> Value {
    serde_json::json!({
        "inputs": prompt,
        "parameters": params.request_params,
    })
}

fn build_tgi_headers(api_token: Option<&String>, ide: Ide) -> Result<HeaderMap, jsonrpc::Error> {
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

fn parse_tgi_text(text: reqwest::Result<String>) -> Result<Vec<Generation>, jsonrpc::Error> {
    let generations =
        match serde_json::from_str(&text.unwrap_or_default()).map_err(internal_error)? {
            APIResponse::Generation(gen) => vec![gen],
            APIResponse::Generations(_) => {
                return Err(internal_error(
                    "TGI parser unexpectedly encountered api-inference",
                ))
            }
            APIResponse::Error(err) => return Err(internal_error(err)),
        };
    Ok(generations)
}

fn build_api_body(prompt: String, params: CompletionParams) -> Value {
    build_tgi_body(prompt, params)
}

fn build_api_headers(api_token: Option<&String>, ide: Ide) -> Result<HeaderMap, jsonrpc::Error> {
    build_tgi_headers(api_token, ide)
}

fn parse_api_text(text: reqwest::Result<String>) -> Result<Vec<Generation>, jsonrpc::Error> {
    let generations =
        match serde_json::from_str(&text.unwrap_or_default()).map_err(internal_error)? {
            APIResponse::Generation(gen) => vec![gen],
            APIResponse::Generations(gens) => gens,
            APIResponse::Error(err) => return Err(internal_error(err)),
        };
    Ok(generations)
}

fn build_ollama_body(prompt: String, params: CompletionParams) -> Value {
    let request_body = params.request_body.unwrap_or_default();
    let body = serde_json::json!({
        "prompt": prompt,
        "model": request_body.get("model"),
    });
    body
}
fn build_ollama_headers() -> Result<HeaderMap, jsonrpc::Error> {
    let headers = HeaderMap::new();
    Ok(headers)
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaGeneration {
    response: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OllamaAPIResponse {
    Generation(OllamaGeneration),
    Error(APIError),
}

fn parse_ollama_text(
    text: Result<String, reqwest::Error>,
) -> Result<Vec<Generation>, jsonrpc::Error> {
    match text {
        Ok(text) => {
            let mut gen: Vec<Generation> = Vec::new();
            for row in text.split('\n') {
                if row.is_empty() {
                    continue;
                }
                let chunk = match serde_json::from_str(row) {
                    Ok(OllamaAPIResponse::Generation(ollama_gen)) => ollama_gen.response,
                    Ok(OllamaAPIResponse::Error(err)) => return Err(internal_error(err)),
                    Err(err) => return Err(internal_error(err)),
                };
                gen.push(Generation {
                    generated_text: chunk,
                })
            }
            Ok(gen)
        }
        Err(err) => Err(internal_error(err)),
    }
}

const TGI: &str = "tgi";
const HUGGING_FACE: &str = "huggingface";
const OLLAMA: &str = "ollama";
const DEFAULT_ADAPTOR: &str = HUGGING_FACE;

fn unknown_adaptor_error(adaptor: String) -> jsonrpc::Error {
    internal_error(format!("Unknown adaptor {}", adaptor))
}

pub fn adapt_body(prompt: String, params: CompletionParams) -> Result<Value, jsonrpc::Error> {
    let adaptor = params
        .adaptor
        .clone()
        .unwrap_or(DEFAULT_ADAPTOR.to_string());
    match adaptor.as_str() {
        TGI => Ok(build_tgi_body(prompt, params)),
        HUGGING_FACE => Ok(build_api_body(prompt, params)),
        OLLAMA => Ok(build_ollama_body(prompt, params)),
        _ => Err(unknown_adaptor_error(adaptor)),
    }
}

pub fn adapt_headers(
    adaptor: Option<String>,
    api_token: Option<&String>,
    ide: Ide,
) -> Result<HeaderMap, jsonrpc::Error> {
    let adaptor = adaptor.clone().unwrap_or(DEFAULT_ADAPTOR.to_string());
    match adaptor.as_str() {
        TGI => build_tgi_headers(api_token, ide),
        HUGGING_FACE => build_api_headers(api_token, ide),
        OLLAMA => build_ollama_headers(),
        _ => Err(internal_error(adaptor)),
    }
}

pub fn adapt_text(
    adaptor: Option<String>,
    text: Result<String, reqwest::Error>,
) -> jsonrpc::Result<Vec<Generation>> {
    let adaptor = adaptor.clone().unwrap_or(DEFAULT_ADAPTOR.to_string());
    match adaptor.as_str() {
        TGI => parse_tgi_text(text),
        HUGGING_FACE => parse_api_text(text),
        OLLAMA => parse_ollama_text(text),
        _ => Err(unknown_adaptor_error(adaptor)),
    }
}
