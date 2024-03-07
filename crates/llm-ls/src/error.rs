use std::fmt::Display;

use tower_lsp::{jsonrpc::Error as LspError, lsp_types::Range};
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
    #[error("no encoding kind provided by the client")]
    EncodingKindMissing,
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
    #[error("range out of bounds: {0:?}")]
    InvalidRange(Range),
    #[error("invalid repository id")]
    InvalidRepositoryId,
    #[error("invalid tokenizer path")]
    InvalidTokenizerPath,
    #[error("ollama error: {0}")]
    Ollama(crate::backend::APIError),
    #[error("openai error: {0}")]
    OpenAI(crate::backend::OpenAIError),
    #[error("index out of bounds: {0}")]
    OutOfBoundIndexing(usize),
    #[error("line out of bounds: {0} >= {1}")]
    OutOfBoundLine(usize, usize),
    #[error("slice out of bounds: {0}..{1}")]
    OutOfBoundSlice(usize, usize),
    #[error("rope error: {0}")]
    Rope(#[from] ropey::Error),
    #[error("serde json error: {0}")]
    SerdeJson(#[from] serde_json::Error),
    #[error("tgi error: {0}")]
    Tgi(crate::backend::APIError),
    #[error("tree-sitter parse error: timeout possibly exceeded")]
    TreeSitterParseError,
    #[error("tree-sitter language error: {0}")]
    TreeSitterLanguage(#[from] tree_sitter::LanguageError),
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("tokio join error: {0}")]
    TokioJoin(#[from] tokio::task::JoinError),
    #[error("unknown backend: {0}")]
    UnknownBackend(String),
    #[error("unknown encoding kind: {0}")]
    UnknownEncodingKind(String)
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

impl From<Error> for LspError {
    fn from(err: Error) -> Self {
        internal_error(err)
    }
}
