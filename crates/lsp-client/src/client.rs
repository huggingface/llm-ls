use std::sync::Arc;

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::{oneshot, Mutex};
use tokio::task::JoinHandle;
use tracing::{debug, error};

use crate::error::Result;
use crate::msg::{Message, Notification, Request, Response};
use crate::res_queue::ResQueue;
use crate::server::Server;

pub struct Connection {
    pub(crate) sender: UnboundedSender<Message>,
    pub(crate) receiver: UnboundedReceiver<Message>,
}

#[derive(Clone)]
pub struct LspClient {
    reader_thread: Arc<JoinHandle<()>>,
    res_queue: Arc<Mutex<ResQueue<oneshot::Sender<Response>>>>,
    server: Arc<Server>,
    server_sender: UnboundedSender<Message>,
}

impl LspClient {
    pub async fn new(conn: Connection, server: Server) -> Self {
        let res_queue = Arc::new(Mutex::new(ResQueue::default()));
        let res_queue_clone = res_queue.clone();
        let mut rx = conn.receiver;
        let reader_thread = tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                match msg {
                    Message::Request(req) => Self::on_request(req),
                    Message::Notification(not) => Self::on_notification(not),
                    Message::Response(res) => {
                        Self::complete_request(res_queue_clone.clone(), res).await
                    }
                }
            }
        });
        Self {
            reader_thread: Arc::new(reader_thread),
            res_queue,
            server: Arc::new(server),
            server_sender: conn.sender,
        }
    }

    fn on_request(_req: Request) {
        todo!("requests are not handled by client");
    }

    fn on_notification(not: Notification) {
        debug!("received notification: {not:?}");
    }

    pub async fn send_request<R: lsp_types::request::Request>(
        &self,
        params: R::Params,
    ) -> Result<Response> {
        let (sender, receiver) = oneshot::channel::<Response>();
        let request =
            self.res_queue
                .lock()
                .await
                .outgoing
                .register(R::METHOD.to_string(), params, sender);

        self.send(request.into());
        Ok(receiver.await?)
    }

    async fn complete_request(
        res_queue: Arc<Mutex<ResQueue<oneshot::Sender<Response>>>>,
        response: Response,
    ) {
        let sender = res_queue
            .lock()
            .await
            .outgoing
            .complete(response.id.clone())
            .expect("received response for unknown request");

        sender.send(response).unwrap();
    }

    pub fn send_notification<N: lsp_types::notification::Notification>(&self, params: N::Params) {
        let not = Notification::new(N::METHOD.to_string(), params);
        self.send(not.into());
    }

    pub async fn shutdown(&self) -> Result<()> {
        self.send_request::<lsp_types::request::Shutdown>(())
            .await?;
        Ok(())
    }

    /// Exit will join on server threads waiting for exit.
    ///
    /// This will fail if there are other strong references to the [`Server`] instance
    pub async fn exit(self) {
        self.send_notification::<lsp_types::notification::Exit>(());

        match Arc::into_inner(self.reader_thread) {
            Some(reader) => match reader.await {
                Ok(r) => r,
                Err(err) => {
                    error!("client reader panicked!");
                    std::panic::panic_any(err)
                }
            },
            None => error!("error joining client thread, resources may have been leaked"),
        };

        match Arc::into_inner(self.server) {
            Some(server) => {
                match server.join().await {
                    Ok(_) => (),
                    Err(err) => error!("thread exited with error: {}", err),
                };
            }
            None => error!("error joining server threads, resources may have been leaked"),
        };
    }

    fn send(&self, message: Message) {
        self.server_sender.send(message).unwrap();
    }
}
