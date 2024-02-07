use super::{APIError, APIResponse, CompletionParams, Generation, Ide, NAME, VERSION};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, USER_AGENT};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fmt::Display;

use crate::error::{Error, Result};

fn build_tgi_headers(api_token: Option<&String>, ide: Ide) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    let user_agent = format!("{NAME}/{VERSION}; rust/unknown; ide/{ide:?}");
    headers.insert(USER_AGENT, HeaderValue::from_str(&user_agent)?);

    if let Some(api_token) = api_token {
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_token}"))?,
        );
    }

    Ok(headers)
}

fn parse_tgi_text(text: &str) -> Result<Vec<Generation>> {
    match serde_json::from_str(text)? {
        APIResponse::Generation(gen) => Ok(vec![gen]),
        APIResponse::Generations(_) => Err(Error::InvalidBackend),
        APIResponse::Error(err) => Err(Error::Tgi(err)),
    }
}

fn build_api_headers(api_token: Option<&String>, ide: Ide) -> Result<HeaderMap> {
    build_tgi_headers(api_token, ide)
}

fn parse_api_text(text: &str) -> Result<Vec<Generation>> {
    match serde_json::from_str(text)? {
        APIResponse::Generation(gen) => Ok(vec![gen]),
        APIResponse::Generations(gens) => Ok(gens),
        APIResponse::Error(err) => Err(Error::InferenceApi(err)),
    }
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

fn build_ollama_headers() -> HeaderMap {
    HeaderMap::new()
}

fn parse_ollama_text(text: &str) -> Result<Vec<Generation>> {
    match serde_json::from_str(text)? {
        OllamaAPIResponse::Generation(gen) => Ok(vec![gen.into()]),
        OllamaAPIResponse::Error(err) => Err(Error::Ollama(err)),
    }
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
pub struct OpenAIError {
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

fn build_openai_headers(api_token: Option<&String>, ide: Ide) -> Result<HeaderMap> {
    build_api_headers(api_token, ide)
}

fn parse_openai_text(text: &str) -> Result<Vec<Generation>> {
    match serde_json::from_str(text)? {
        OpenAIAPIResponse::Generation(completion) => {
            Ok(completion.choices.into_iter().map(|x| x.into()).collect())
        }
        OpenAIAPIResponse::Error(err) => Err(Error::OpenAI(err)),
    }
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Backend {
    #[default]
    HuggingFace,
    Ollama,
    OpenAi,
    Tgi,
}

pub fn build_body(prompt: String, params: &CompletionParams) -> Map<String, Value> {
    let mut body = params.request_body.clone();
    match params.backend {
        Backend::HuggingFace | Backend::Tgi => {
            body.insert("inputs".to_string(), Value::String(prompt))
        }
        Backend::Ollama | Backend::OpenAi => {
            body.insert("prompt".to_string(), Value::String(prompt))
        }
    };
    body
}

pub fn build_headers(backend: &Backend, api_token: Option<&String>, ide: Ide) -> Result<HeaderMap> {
    match backend {
        Backend::HuggingFace => build_api_headers(api_token, ide),
        Backend::Ollama => Ok(build_ollama_headers()),
        Backend::OpenAi => build_openai_headers(api_token, ide),
        Backend::Tgi => build_tgi_headers(api_token, ide),
    }
}

pub fn parse_generations(backend: &Backend, text: &str) -> Result<Vec<Generation>> {
    match backend {
        Backend::HuggingFace => parse_api_text(text),
        Backend::Ollama => parse_ollama_text(text),
        Backend::OpenAi => parse_openai_text(text),
        Backend::Tgi => parse_tgi_text(text),
    }
}
