use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Collection {
    #[error("bincode error: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("The dimension of the vector doesn't match the dimension of the collection")]
    DimensionMismatch,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
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
    #[error("collection error: {0}")]
    Collection(#[from] Collection),
}

pub type Result<T> = std::result::Result<T, Error>;
