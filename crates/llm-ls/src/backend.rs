use super::{Generation, NAME, VERSION};
use custom_types::llm_ls::{Backend, Ide};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, USER_AGENT};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::fmt::Display;

use crate::error::{Error, Result};

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
pub(crate) enum APIResponse {
    Generation(Generation),
    Generations(Vec<Generation>),
    Error(APIError),
}

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
struct CohereGeneration {
    text: String,
}

impl From<CohereGeneration> for Generation {
    fn from(value: CohereGeneration) -> Self {
        Generation {
            generated_text: value.text,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct CohereError {
    message: String,
}

impl std::error::Error for CohereError {
    fn description(&self) -> &str {
        &self.message
    }
}

impl Display for CohereError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CohereAPIResponse {
    Generation(CohereGeneration),
    Error(CohereError),
}

fn build_cohere_headers(api_token: Option<&String>, ide: Ide) -> Result<HeaderMap> {
    build_api_headers(api_token, ide)
}

fn parse_cohere_text(text: &str) -> Result<Vec<Generation>> {
    match serde_json::from_str(text)? {
        CohereAPIResponse::Generation(gen) => Ok(vec![gen.into()]),
        CohereAPIResponse::Error(err) => Err(Error::Cohere(err)),
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct LlamaCppGeneration {
    content: String,
}

impl From<LlamaCppGeneration> for Generation {
    fn from(value: LlamaCppGeneration) -> Self {
        Generation {
            generated_text: value.content,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum LlamaCppAPIResponse {
    Generation(LlamaCppGeneration),
    Error(APIError),
}

fn build_llamacpp_headers() -> HeaderMap {
    HeaderMap::new()
}

fn parse_llamacpp_text(text: &str) -> Result<Vec<Generation>> {
    match serde_json::from_str(text)? {
        LlamaCppAPIResponse::Generation(gen) => Ok(vec![gen.into()]),
        LlamaCppAPIResponse::Error(err) => Err(Error::LlamaCpp(err)),
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

pub(crate) fn build_body(
    backend: &Backend,
    model: String,
    prompt: String,
    mut request_body: Map<String, Value>,
) -> Map<String, Value> {
    match backend {
        Backend::Cohere { .. } => {
            request_body.insert("message".to_owned(), Value::String(prompt));
            request_body.insert("model".to_owned(), Value::String(model));
            request_body.insert("stream".to_owned(), Value::Bool(false));
        }
        Backend::HuggingFace { .. } | Backend::Tgi { .. } => {
            request_body.insert("inputs".to_owned(), Value::String(prompt));
            if let Some(Value::Object(params)) = request_body.get_mut("parameters") {
                params.insert("return_full_text".to_owned(), Value::Bool(false));
            } else {
                let params = json!({ "parameters": { "return_full_text": false } });
                request_body.insert("parameters".to_owned(), params);
            }
        }
        Backend::LlamaCpp { .. } => {
            request_body.insert("prompt".to_owned(), Value::String(prompt));
        }
        Backend::Ollama { .. } | Backend::OpenAi { .. } => {
            request_body.insert("prompt".to_owned(), Value::String(prompt));
            request_body.insert("model".to_owned(), Value::String(model));
            request_body.insert("stream".to_owned(), Value::Bool(false));
        }
    };
    request_body
}

pub(crate) fn build_headers(
    backend: &Backend,
    api_token: Option<&String>,
    ide: Ide,
) -> Result<HeaderMap> {
    match backend {
        Backend::Cohere { .. } => build_cohere_headers(api_token, ide),
        Backend::HuggingFace { .. } => build_api_headers(api_token, ide),
        Backend::LlamaCpp { .. } => Ok(build_llamacpp_headers()),
        Backend::Ollama { .. } => Ok(build_ollama_headers()),
        Backend::OpenAi { .. } => build_openai_headers(api_token, ide),
        Backend::Tgi { .. } => build_tgi_headers(api_token, ide),
    }
}

pub(crate) fn parse_generations(backend: &Backend, text: &str) -> Result<Vec<Generation>> {
    match backend {
        Backend::Cohere { .. } => parse_cohere_text(text),
        Backend::HuggingFace { .. } => parse_api_text(text),
        Backend::LlamaCpp { .. } => parse_llamacpp_text(text),
        Backend::Ollama { .. } => parse_ollama_text(text),
        Backend::OpenAi { .. } => parse_openai_text(text),
        Backend::Tgi { .. } => parse_tgi_text(text),
    }
}
