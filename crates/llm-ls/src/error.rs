use std::fmt::Display;

use hf_hub::api::tokio::ApiError;
use tower_lsp::jsonrpc::Error as LspError;
use tracing::error;

use crate::APIError;

pub fn internal_error<E: Display>(err: E) -> LspError {
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
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),
    #[error("candle error: {0}")]
    Candle(#[from] candle::Error),
    #[error("gitignore error: {0}")]
    Gitignore(#[from] gitignore::Error),
    #[error("hugging face api error: {0}")]
    HfApi(#[from] ApiError),
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("inference api error: {0}")]
    InferenceApi(APIError),
    #[error("You are attempting to parse a result in the API inference format when using the `tgi` adaptor")]
    InvalidAdaptor,
    #[error("invalid header value: {0}")]
    InvalidHeaderValue(#[from] reqwest::header::InvalidHeaderValue),
    #[error("invalid repository id")]
    InvalidRepositoryId,
    #[error("invalid tokenizer path")]
    InvalidTokenizerPath,
    #[error("ollama error: {0}")]
    Ollama(APIError),
    #[error("openai error: {0}")]
    OpenAI(crate::adaptors::OpenAIError),
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
    #[error("tgi error: {0}")]
    Tgi(APIError),
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("tokio join error: {0}")]
    TokioJoin(#[from] tokio::task::JoinError),
    #[error("unknown adaptor: {0}")]
    UnknownAdaptor(String),
    #[error("vector db error: {0}")]
    VectorDb(#[from] vectordb::error::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<Error> for LspError {
    fn from(err: Error) -> Self {
        internal_error(err)
    }
}
