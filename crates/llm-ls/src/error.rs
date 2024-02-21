use std::fmt::Display;

use tower_lsp::jsonrpc::Error as LspError;
use tracing::error;

pub(crate) fn internal_error<E: Display>(err: E) -> LspError {
    let err_msg = err.to_string();
    error!(err_msg);
    LspError {
        code: tower_lsp::jsonrpc::ErrorCode::InternalError,
        message: err_msg.into(),
        data: None,
    }
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("candle error: {0}")]
    Candle(#[from] candle::Error),
    #[error("gitignore error: {0}")]
    Gitignore(#[from] gitignore::Error),
    #[error("huggingface api error: {0}")]
    HfApi(#[from] hf_hub::api::tokio::ApiError),
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("inference api error: {0}")]
    InferenceApi(crate::backend::APIError),
    #[error("You are attempting to parse a result in the API inference format when using the `tgi` backend")]
    InvalidBackend,
    #[error("invalid header value: {0}")]
    InvalidHeaderValue(#[from] reqwest::header::InvalidHeaderValue),
    #[error("invalid repository id")]
    InvalidRepositoryId,
    #[error("invalid tokenizer path")]
    InvalidTokenizerPath,
    #[error("malformatted embedding metadata, missing {0} field")]
    MalformattedEmbeddingMetadata(String),
    #[error("embedding has no metadata")]
    MissingMetadata,
    #[error("ollama error: {0}")]
    Ollama(crate::backend::APIError),
    #[error("openai error: {0}")]
    OpenAI(crate::backend::OpenAIError),
    #[error("index out of bounds: {0}")]
    OutOfBoundIndexing(usize),
    #[error("line out of bounds: {0}")]
    OutOfBoundLine(usize),
    #[error("slice out of bounds: {0}..{1}")]
    OutOfBoundSlice(usize, usize),
    #[error("rope error: {0}")]
    Rope(#[from] ropey::Error),
    #[error("serde json error: {0}")]
    SerdeJson(#[from] serde_json::Error),
    #[error("snippet is too larger to be converted to an embedding: {0} > {1}")]
    SnippetTooLarge(usize, usize),
    #[error("strip prefix error: {0}")]
    StripPrefix(#[from] std::path::StripPrefixError),
    #[error("tgi error: {0}")]
    Tgi(crate::backend::APIError),
    #[error("tinyvec-embed error: {0}")]
    TinyVecEmbed(#[from] tinyvec_embed::error::Error),
    #[error("tree-sitter language error: {0}")]
    TreeSitterLanguage(#[from] tree_sitter::LanguageError),
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("tokio join error: {0}")]
    TokioJoin(#[from] tokio::task::JoinError),
    #[error("unknown backend: {0}")]
    UnknownBackend(String),
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

impl From<Error> for LspError {
    fn from(err: Error) -> Self {
        internal_error(err)
    }
}
