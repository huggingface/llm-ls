use std::{
    fmt::{self, Display},
    io,
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tracing::debug;

use crate::error::ExtractError;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum Message {
    Request(Request),
    Response(Response),
    Notification(Notification),
}

impl From<Request> for Message {
    fn from(request: Request) -> Message {
        Message::Request(request)
    }
}

impl From<Response> for Message {
    fn from(response: Response) -> Message {
        Message::Response(response)
    }
}

impl From<Notification> for Message {
    fn from(notification: Notification) -> Message {
        Message::Notification(notification)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(transparent)]
pub struct RequestId(IdRepr);

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(untagged)]
enum IdRepr {
    I32(i32),
    String(String),
}

impl From<i32> for RequestId {
    fn from(id: i32) -> RequestId {
        RequestId(IdRepr::I32(id))
    }
}

impl From<String> for RequestId {
    fn from(id: String) -> RequestId {
        RequestId(IdRepr::String(id))
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            IdRepr::I32(it) => fmt::Display::fmt(it, f),
            // Use debug here, to make it clear that `92` and `"92"` are
            // different, and to reduce WTF factor if the sever uses `" "` as an
            // ID.
            IdRepr::String(it) => fmt::Debug::fmt(it, f),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Request {
    pub id: RequestId,
    pub method: String,
    #[serde(default = "serde_json::Value::default")]
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    pub params: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ResponseContent {
    Result(serde_json::Value),
    Error(ResponseError),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Response {
    // JSON RPC allows this to be null if it was impossible
    // to decode the request's id. Ignore this special case
    // and just die horribly.
    pub id: RequestId,
    #[serde(flatten)]
    pub content: ResponseContent,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResponseError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl Display for ResponseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match ErrorCode::try_from_i32(self.code) {
            Ok(code) => write!(f, "{:?} [{}]: {}", code, self.code, self.message),
            Err(_) => write!(f, "Unknown error code [{}]: {}", self.code, self.message),
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum ErrorCode {
    // Defined by JSON RPC:
    ParseError = -32700,
    InvalidRequest = -32600,
    MethodNotFound = -32601,
    InvalidParams = -32602,
    InternalError = -32603,
    ServerErrorStart = -32099,
    ServerErrorEnd = -32000,

    /// Error code indicating that a server received a notification or
    /// request before the server has received the `initialize` request.
    ServerNotInitialized = -32002,
    UnknownErrorCode = -32001,

    // Defined by the protocol:
    /// The client has canceled a request and a server has detected
    /// the cancel.
    RequestCanceled = -32800,

    /// The server detected that the content of a document got
    /// modified outside normal conditions. A server should
    /// NOT send this error code if it detects a content change
    /// in it unprocessed messages. The result even computed
    /// on an older state might still be useful for the client.
    ///
    /// If a client decides that a result is not of any use anymore
    /// the client should cancel the request.
    ContentModified = -32801,

    /// The server cancelled the request. This error code should
    /// only be used for requests that explicitly support being
    /// server cancellable.
    ///
    /// @since 3.17.0
    ServerCancelled = -32802,

    /// A request failed but it was syntactically correct, e.g the
    /// method name was known and the parameters were valid. The error
    /// message should contain human readable information about why
    /// the request failed.
    ///
    /// @since 3.17.0
    RequestFailed = -32803,
}

impl ErrorCode {
    fn try_from_i32(code: i32) -> Result<ErrorCode, ()> {
        match code {
            -32700 => Ok(ErrorCode::ParseError),
            -32600 => Ok(ErrorCode::InvalidRequest),
            -32601 => Ok(ErrorCode::MethodNotFound),
            -32602 => Ok(ErrorCode::InvalidParams),
            -32603 => Ok(ErrorCode::InternalError),
            -32099 => Ok(ErrorCode::ServerErrorStart),
            -32000 => Ok(ErrorCode::ServerErrorEnd),
            -32002 => Ok(ErrorCode::ServerNotInitialized),
            -32001 => Ok(ErrorCode::UnknownErrorCode),
            -32800 => Ok(ErrorCode::RequestCanceled),
            -32801 => Ok(ErrorCode::ContentModified),
            -32802 => Ok(ErrorCode::ServerCancelled),
            -32803 => Ok(ErrorCode::RequestFailed),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Notification {
    pub method: String,
    #[serde(default = "serde_json::Value::default")]
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    pub params: serde_json::Value,
}

impl Message {
    pub async fn read<R: AsyncBufRead + Unpin + ?Sized>(r: &mut R) -> io::Result<Option<Message>> {
        Message::_read(r).await
    }

    async fn _read<R: AsyncBufRead + Unpin + ?Sized>(r: &mut R) -> io::Result<Option<Message>> {
        let text = match read_msg_text(r).await? {
            None => return Ok(None),
            Some(text) => text,
        };
        let msg = serde_json::from_str(&text)?;
        Ok(Some(msg))
    }

    pub async fn write<W: AsyncWrite + Unpin>(self, w: &mut W) -> io::Result<()> {
        self._write(w).await
    }

    async fn _write<W: AsyncWrite + Unpin>(self, w: &mut W) -> io::Result<()> {
        #[derive(Serialize)]
        struct JsonRpc {
            jsonrpc: &'static str,
            #[serde(flatten)]
            msg: Message,
        }
        let text = serde_json::to_string(&JsonRpc {
            jsonrpc: "2.0",
            msg: self,
        })?;
        write_msg_text(w, &text).await
    }
}

impl Response {
    pub fn new_ok<R: Serialize>(id: RequestId, result: R) -> Response {
        Response {
            id,
            content: ResponseContent::Result(serde_json::to_value(result).unwrap()),
        }
    }
    pub fn new_err(id: RequestId, code: i32, message: String) -> Response {
        let error = ResponseError {
            code,
            message,
            data: None,
        };
        Response {
            id,
            content: ResponseContent::Error(error),
        }
    }

    pub fn extract<P: DeserializeOwned>(self) -> Result<(RequestId, P), ExtractError> {
        match self.content {
            ResponseContent::Result(result) => match serde_json::from_value(result) {
                Ok(params) => Ok((self.id, params)),
                Err(error) => Err(ExtractError::JsonError {
                    msg_type: "response".to_owned(),
                    method: None,
                    error,
                }),
            },
            ResponseContent::Error(error) => Err(ExtractError::ResponseError { error }),
        }
    }
}

impl Request {
    pub fn new<P: Serialize>(id: RequestId, method: String, params: P) -> Request {
        Request {
            id,
            method,
            params: serde_json::to_value(params).unwrap(),
        }
    }

    pub fn extract<P: DeserializeOwned>(
        self,
        method: &str,
    ) -> Result<(RequestId, P), ExtractError> {
        if self.method != method {
            return Err(ExtractError::MethodMismatch(self.method, method.to_owned()));
        }
        match serde_json::from_value(self.params) {
            Ok(params) => Ok((self.id, params)),
            Err(error) => Err(ExtractError::JsonError {
                msg_type: "request".to_owned(),
                method: Some(self.method),
                error,
            }),
        }
    }
}

impl Notification {
    pub fn new(method: String, params: impl Serialize) -> Notification {
        Notification {
            method,
            params: serde_json::to_value(params).unwrap(),
        }
    }

    pub fn extract<P: DeserializeOwned>(self, method: &str) -> Result<P, ExtractError> {
        if self.method != method {
            return Err(ExtractError::MethodMismatch(self.method, method.to_owned()));
        }
        match serde_json::from_value(self.params) {
            Ok(params) => Ok(params),
            Err(error) => Err(ExtractError::JsonError {
                msg_type: "notification".to_owned(),
                method: Some(self.method),
                error,
            }),
        }
    }

    pub(crate) fn is_exit(&self) -> bool {
        self.method == "exit"
    }
}

async fn read_msg_text<R: AsyncBufRead + Unpin + ?Sized>(
    inp: &mut R,
) -> io::Result<Option<String>> {
    fn invalid_data(error: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, error)
    }
    macro_rules! invalid_data {
        ($($tt:tt)*) => (invalid_data(format!($($tt)*)))
    }

    let mut size = None;
    let mut buf = String::new();
    loop {
        buf.clear();
        if inp.read_line(&mut buf).await? == 0 {
            return Ok(None);
        }
        if !buf.ends_with("\r\n") {
            return Err(invalid_data!("malformed header: {:?}", buf));
        }
        let buf = &buf[..buf.len() - 2];
        if buf.is_empty() {
            break;
        }
        let mut parts = buf.splitn(2, ": ");
        let header_name = parts.next().unwrap();
        let header_value = parts
            .next()
            .ok_or_else(|| invalid_data!("malformed header: {:?}", buf))?;
        if header_name.eq_ignore_ascii_case("Content-Length") {
            size = Some(header_value.parse::<usize>().map_err(invalid_data)?);
        }
    }
    let size: usize = size.ok_or_else(|| invalid_data!("no Content-Length"))?;
    let mut buf = buf.into_bytes();
    buf.resize(size, 0);
    inp.read_exact(&mut buf).await?;
    let buf = String::from_utf8(buf).map_err(invalid_data)?;
    debug!("< {}", buf);
    Ok(Some(buf))
}

async fn write_msg_text<W: AsyncWrite + Unpin>(out: &mut W, msg: &str) -> io::Result<()> {
    debug!("> {}", msg);
    out.write_all(format!("Content-Length: {}\r\n\r\n", msg.len()).as_bytes())
        .await?;
    out.write_all(msg.as_bytes()).await?;
    out.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::msg::{ResponseContent, ResponseError};

    use super::{Message, Notification, Request, RequestId, Response};

    #[test]
    fn shutdown_with_explicit_null() {
        let text = "{\"jsonrpc\": \"2.0\",\"id\": 3,\"method\": \"shutdown\", \"params\": null }";
        let msg: Message = serde_json::from_str(text).unwrap();

        assert!(
            matches!(msg, Message::Request(req) if req.id == 3.into() && req.method == "shutdown")
        );
    }

    #[test]
    fn shutdown_with_no_params() {
        let text = "{\"jsonrpc\": \"2.0\",\"id\": 3,\"method\": \"shutdown\"}";
        let msg: Message = serde_json::from_str(text).unwrap();

        assert!(
            matches!(msg, Message::Request(req) if req.id == 3.into() && req.method == "shutdown")
        );
    }

    #[test]
    fn notification_with_explicit_null() {
        let text = "{\"jsonrpc\": \"2.0\",\"method\": \"exit\", \"params\": null }";
        let msg: Message = serde_json::from_str(text).unwrap();

        assert!(matches!(msg, Message::Notification(not) if not.method == "exit"));
    }

    #[test]
    fn notification_with_no_params() {
        let text = "{\"jsonrpc\": \"2.0\",\"method\": \"exit\"}";
        let msg: Message = serde_json::from_str(text).unwrap();

        assert!(matches!(msg, Message::Notification(not) if not.method == "exit"));
    }

    #[test]
    fn serialize_request_with_null_params() {
        let msg = Message::Request(Request {
            id: RequestId::from(3),
            method: "shutdown".into(),
            params: serde_json::Value::Null,
        });
        let serialized = serde_json::to_string(&msg).unwrap();

        assert_eq!("{\"id\":3,\"method\":\"shutdown\"}", serialized);
    }

    #[test]
    fn serialize_notification_with_null_params() {
        let msg = Message::Notification(Notification {
            method: "exit".into(),
            params: serde_json::Value::Null,
        });
        let serialized = serde_json::to_string(&msg).unwrap();

        assert_eq!("{\"method\":\"exit\"}", serialized);
    }

    #[test]
    fn serialize_response_with_null_result() {
        let text = "{\"id\":1,\"result\":null}";
        let msg = Message::Response(Response {
            id: RequestId::from(1),
            content: ResponseContent::Result(serde_json::Value::Null),
        });
        let serialized = serde_json::to_string(&msg).unwrap();

        assert_eq!(text, serialized);
    }

    #[test]
    fn serialize_response_with_error() {
        let text = "{\"id\":1,\"error\":{\"code\":-32603,\"message\":\"some error message\"}}";
        let msg = Message::Response(Response {
            id: RequestId::from(1),
            content: ResponseContent::Error(ResponseError {
                code: -32603,
                message: "some error message".to_owned(),
                data: None,
            }),
        });
        let serialized = serde_json::to_string(&msg).unwrap();

        assert_eq!(text, serialized);
    }
}
