use crate::RequestParams;

use super::{
    internal_error, APIError, APIResponse, CompletionParams, Generation, Ide, NAME, VERSION,
};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, USER_AGENT};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Display;
use tower_lsp::jsonrpc;

fn build_tgi_body(prompt: String, params: &RequestParams) -> Value {
    serde_json::json!({
        "inputs": prompt,
        "parameters": params,
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

fn parse_tgi_text(text: &str) -> Result<Vec<Generation>, jsonrpc::Error> {
    let generations =
        match serde_json::from_str(text).map_err(internal_error)? {
            APIResponse::Generation(gen) => vec![gen],
            APIResponse::Generations(_) => {
                return Err(internal_error(
                    "You are attempting to parse a result in the API inference format when using the `tgi` adaptor",
                ))
            }
            APIResponse::Error(err) => return Err(internal_error(err)),
        };
    Ok(generations)
}

fn build_api_body(prompt: String, params: &RequestParams) -> Value {
    build_tgi_body(prompt, params)
}

fn build_api_headers(api_token: Option<&String>, ide: Ide) -> Result<HeaderMap, jsonrpc::Error> {
    build_tgi_headers(api_token, ide)
}

fn parse_api_text(text: &str) -> Result<Vec<Generation>, jsonrpc::Error> {
    let generations = match serde_json::from_str(text).map_err(internal_error)? {
        APIResponse::Generation(gen) => vec![gen],
        APIResponse::Generations(gens) => gens,
        APIResponse::Error(err) => return Err(internal_error(err)),
    };
    Ok(generations)
}

fn build_ollama_body(prompt: String, params: &CompletionParams) -> Value {
    serde_json::json!({
        "prompt": prompt,
        "model": params.request_body.as_ref().unwrap().get("model"),
        "stream": false,
    })
}
fn build_ollama_headers() -> Result<HeaderMap, jsonrpc::Error> {
    Ok(HeaderMap::new())
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaGeneration {
    response: String,
}

impl From<OllamaGeneration> for Generation {
    fn from(value: OllamaGeneration) -> Self {
        Generation {
            generated_text: value.response,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OllamaAPIResponse {
    Generation(OllamaGeneration),
    Error(APIError),
}

fn parse_ollama_text(text: &str) -> Result<Vec<Generation>, jsonrpc::Error> {
    let generations = match serde_json::from_str(text).map_err(internal_error)? {
        OllamaAPIResponse::Generation(gen) => vec![gen.into()],
        OllamaAPIResponse::Error(err) => return Err(internal_error(err)),
    };
    Ok(generations)
}

fn build_openai_body(prompt: String, params: &CompletionParams) -> Value {
    serde_json::json!({
        "prompt": prompt,
        "model": params.model,
        "max_tokens": params.request_params.max_new_tokens,
        "temperature": params.request_params.temperature,
        "top_p": params.request_params.top_p,
        "stop": params.request_params.stop_tokens.clone(),
    })
}

fn build_openai_headers(api_token: Option<&String>, ide: Ide) -> Result<HeaderMap, jsonrpc::Error> {
    build_api_headers(api_token, ide)
}

#[derive(Debug, Deserialize)]
struct OpenAIGenerationChoice {
    text: String,
}

impl From<OpenAIGenerationChoice> for Generation {
    fn from(value: OpenAIGenerationChoice) -> Self {
        Generation {
            generated_text: value.text,
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenAIGeneration {
    choices: Vec<OpenAIGenerationChoice>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenAIErrorLoc {
    String(String),
    Int(u32),
}

impl Display for OpenAIErrorLoc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenAIErrorLoc::String(s) => s.fmt(f),
            OpenAIErrorLoc::Int(i) => i.fmt(f),
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenAIErrorDetail {
    loc: OpenAIErrorLoc,
    msg: String,
    r#type: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIError {
    detail: Vec<OpenAIErrorDetail>,
}

impl Display for OpenAIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, item) in self.detail.iter().enumerate() {
            if i != 0 {
                writeln!(f)?;
            }
            write!(f, "{}: {} ({})", item.loc, item.msg, item.r#type)?;
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenAIAPIResponse {
    Generation(OpenAIGeneration),
    Error(OpenAIError),
}

fn parse_openai_text(text: &str) -> Result<Vec<Generation>, jsonrpc::Error> {
    match serde_json::from_str(text).map_err(internal_error) {
        Ok(OpenAIAPIResponse::Generation(completion)) => {
            Ok(completion.choices.into_iter().map(|x| x.into()).collect())
        }
        Ok(OpenAIAPIResponse::Error(err)) => Err(internal_error(err)),
        Err(err) => Err(internal_error(err)),
    }
}

const TGI: &str = "tgi";
const HUGGING_FACE: &str = "huggingface";
const OLLAMA: &str = "ollama";
const OPENAI: &str = "openai";
const DEFAULT_ADAPTOR: &str = HUGGING_FACE;

fn unknown_adaptor_error(adaptor: Option<&String>) -> jsonrpc::Error {
    internal_error(format!("Unknown adaptor {:?}", adaptor))
}

pub fn adapt_body(prompt: String, params: &CompletionParams) -> Result<Value, jsonrpc::Error> {
    match params
        .adaptor
        .as_ref()
        .unwrap_or(&DEFAULT_ADAPTOR.to_string())
        .as_str()
    {
        TGI => Ok(build_tgi_body(prompt, &params.request_params)),
        HUGGING_FACE => Ok(build_api_body(prompt, &params.request_params)),
        OLLAMA => Ok(build_ollama_body(prompt, params)),
        OPENAI => Ok(build_openai_body(prompt, params)),
        _ => Err(unknown_adaptor_error(params.adaptor.as_ref())),
    }
}

pub fn adapt_headers(
    adaptor: Option<&String>,
    api_token: Option<&String>,
    ide: Ide,
) -> Result<HeaderMap, jsonrpc::Error> {
    match adaptor.unwrap_or(&DEFAULT_ADAPTOR.to_string()).as_str() {
        TGI => build_tgi_headers(api_token, ide),
        HUGGING_FACE => build_api_headers(api_token, ide),
        OLLAMA => build_ollama_headers(),
        OPENAI => build_openai_headers(api_token, ide),
        _ => Err(unknown_adaptor_error(adaptor)),
    }
}

pub fn parse_generations(adaptor: Option<&String>, text: &str) -> jsonrpc::Result<Vec<Generation>> {
    match adaptor.unwrap_or(&DEFAULT_ADAPTOR.to_string()).as_str() {
        TGI => parse_tgi_text(text),
        HUGGING_FACE => parse_api_text(text),
        OLLAMA => parse_ollama_text(text),
        OPENAI => parse_openai_text(text),
        _ => Err(unknown_adaptor_error(adaptor)),
    }
}
