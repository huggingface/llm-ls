use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Collection {
    #[error("The dimension of the vector doesn't match the dimension of the collection")]
    DimensionMismatch,
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
    #[error("bincode error: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("collection error: {0}")]
    Collection(#[from] Collection),
    #[error("a file with an invalid name was found in the database directory")]
    InvalidFileName,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
