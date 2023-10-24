use std::fmt;
use std::io;

use crate::msg::ResponseError;

#[derive(Debug, Clone, PartialEq)]
pub struct ProtocolError(pub(crate) String);

impl std::error::Error for ProtocolError {}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

#[derive(Debug)]
pub enum ExtractError {
    /// The extracted message was of a different method than expected.
    MethodMismatch(String, String),
    /// Failed to deserialize the message.
    JsonError {
        msg_type: String,
        method: Option<String>,
        error: serde_json::Error,
    },
    /// Server responded with an Error
    ResponseError { error: ResponseError },
}

impl std::error::Error for ExtractError {}

impl fmt::Display for ExtractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExtractError::MethodMismatch(asked, req_method) => {
                write!(
                    f,
                    "Method mismatch for request, extract for '{}' != request method '{}'",
                    asked, req_method
                )
            }
            ExtractError::JsonError {
                msg_type,
                method,
                error,
            } => {
                let method = if let Some(method) = method {
                    method.clone()
                } else {
                    "None".to_owned()
                };
                write!(f, "Invalid message body\nMessage type: {msg_type}\nMethod: {method}\n error: {error}",)
            }
            ExtractError::ResponseError { error } => {
                write!(f, "Server answered with an error message\n error: {error}",)
            }
        }
    }
}

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    MissingBinaryPath,
}

impl std::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "IO error: {}", e),
            Error::MissingBinaryPath => write!(f, "Missing binary path"),
        }
    }
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
