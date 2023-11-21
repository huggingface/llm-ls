use super::{
    internal_error, APIError, APIResponse, CompletionParams, Generation, Ide, NAME, VERSION,
};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, USER_AGENT};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tower_lsp::jsonrpc;

struct AdaptHuggingFaceRequest;
impl AdaptHuggingFaceRequest {
    fn adapt_body(&self, prompt: String, params: CompletionParams) -> Value {
        return serde_json::json!({
            "inputs": prompt,
            "parameters": params.request_params,
        });
    }
    fn adapt_headers(
        &self,
        api_token: Option<&String>,
        ide: Ide,
    ) -> Result<HeaderMap, jsonrpc::Error> {
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
}

struct AdaptHuggingFaceResponse;
impl AdaptHuggingFaceResponse {
    fn adapt_blob(&self, text: reqwest::Result<String>) -> Result<Vec<Generation>, jsonrpc::Error> {
        let generations =
            match serde_json::from_str(&text.unwrap_or_default()).map_err(internal_error)? {
                APIResponse::Generation(gen) => vec![gen],
                APIResponse::Generations(gens) => gens,
                APIResponse::Error(err) => return Err(internal_error(err)),
            };
        Ok(generations)
    }
}

struct AdaptOllamaRequest;
impl AdaptOllamaRequest {
    fn adapt_body(&self, prompt: String, params: CompletionParams) -> Value {
        let request_body = params.request_body.unwrap_or_default();
        let body = serde_json::json!({
            "prompt": prompt,
            "model": request_body.get("model"),
        });
        body
    }
    fn adapt_headers(&self) -> Result<HeaderMap, jsonrpc::Error> {
        let headers = HeaderMap::new();
        Ok(headers)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OllamaGeneration {
    response: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum OllamaAPIResponse {
    Generation(OllamaGeneration),
    Error(APIError),
}

struct AdaptOllamaResponse;
impl AdaptOllamaResponse {
    fn adapt_blob(
        &self,
        text: Result<String, reqwest::Error>,
    ) -> Result<Vec<Generation>, jsonrpc::Error> {
        match text {
            Ok(text) => {
                let mut gen: Vec<Generation> = Vec::new();
                for row in text.split("\n") {
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
}

const HUGGING_FACE_ADAPTOR: &str = "huggingface";

pub struct Adaptors;
impl Adaptors {
    pub fn adapt_body(&self, prompt: String, params: CompletionParams) -> Value {
        let adaptor = params.adaptor.clone();
        match adaptor.unwrap_or(HUGGING_FACE_ADAPTOR.to_string()).as_str() {
            "ollama" => AdaptOllamaRequest.adapt_body(prompt, params),
            _ => AdaptHuggingFaceRequest.adapt_body(prompt, params),
        }
    }
    pub fn adapt_headers(
        &self,
        adaptor: Option<String>,
        api_token: Option<&String>,
        ide: Ide,
    ) -> Result<HeaderMap, jsonrpc::Error> {
        match adaptor.unwrap_or(HUGGING_FACE_ADAPTOR.to_string()).as_str() {
            "ollama" => AdaptOllamaRequest.adapt_headers(),
            _ => AdaptHuggingFaceRequest.adapt_headers(api_token, ide),
        }
    }
    pub fn adapt_blob(
        &self,
        adaptor: Option<String>,
        text: Result<String, reqwest::Error>,
    ) -> Result<Vec<Generation>, jsonrpc::Error> {
        match adaptor.unwrap_or(HUGGING_FACE_ADAPTOR.to_string()).as_str() {
            "ollama" => AdaptOllamaResponse.adapt_blob(text),
            _ => AdaptHuggingFaceResponse.adapt_blob(text),
        }
    }
}
