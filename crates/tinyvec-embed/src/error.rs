use std::path::PathBuf;

use crate::db::Value;

#[derive(Debug, thiserror::Error)]
pub enum Collection {
    #[error("The dimension of the vector doesn't match the dimension of the collection")]
    DimensionMismatch,
    #[error("attempt to peek an empty binary heap")]
    EmptyBinaryHeap,
    #[error("invalid path: {0}")]
    InvalidPath(PathBuf),
    #[error("join error: {0}")]
    Join(#[from] tokio::task::JoinError),
    #[error("Collection doesn't exist")]
    NotFound,
    #[error("error sending message in channel")]
    Send,
    #[error("Collection already exists")]
    UniqueViolation,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("acquire error: {0}")]
    Acquire(#[from] tokio::sync::AcquireError),
    #[error("bincode error: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("collection error: {0}")]
    Collection(#[from] Collection),
    #[error("a file with an invalid name was found in the database directory")]
    InvalidFileName,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("expected value to be a valid number, got: {0}")]
    ValueNotNumber(Value),
    #[error("expected value to be string, got: {0}")]
    ValueNotString(Value),
}

pub type Result<T> = std::result::Result<T, Error>;
