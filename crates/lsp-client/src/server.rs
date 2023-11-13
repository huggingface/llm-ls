use std::{io, path::PathBuf, process::Stdio};

use tokio::{
    io::{BufReader, BufWriter},
    process::{Child, Command},
    sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender},
    task::JoinHandle,
};
use tracing::{debug, error};

use crate::msg::Message;
use crate::{
    client::Connection,
    error::{Error, Result},
};

pub struct Server {
    threads: IoThreads,
}

impl Server {
    pub fn build() -> ServerBuilder {
        ServerBuilder {
            binary_path: None,
            command: None,
            transport: Transport::default(),
        }
    }

    /// join server's threads to the current thread
    pub async fn join(self) -> Result<()> {
        self.threads.join().await?;
        Ok(())
    }
}

#[derive(Default)]
pub enum Transport {
    #[default]
    Stdio,
    Socket,
}

pub struct ServerBuilder {
    binary_path: Option<PathBuf>,
    command: Option<Command>,
    transport: Transport,
}

impl ServerBuilder {
    pub fn binary_path(mut self, binary_path: PathBuf) -> Self {
        self.binary_path = Some(binary_path);
        self
    }

    pub fn command(mut self, command: Command) -> Self {
        self.command = Some(command);
        self
    }

    pub fn transport(mut self, transport: Transport) -> Self {
        self.transport = transport;
        self
    }

    pub async fn start(self) -> Result<(Connection, Server)> {
        let mut command = if let Some(command) = self.command {
            command
        } else if let Some(path) = self.binary_path {
            Command::new(path)
        } else {
            return Err(Error::MissingBinaryPath);
        };
        match self.transport {
            Transport::Stdio => {
                let child = command
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .spawn()?;
                let (sender, receiver, threads) = stdio(child);
                Ok((Connection { sender, receiver }, Server { threads }))
            }
            Transport::Socket => {
                todo!("socket transport not implemented");
            }
        }
    }
}

fn stdio(
    mut child: Child,
) -> (
    UnboundedSender<Message>,
    UnboundedReceiver<Message>,
    IoThreads,
) {
    let (writer_sender, mut writer_receiver) = unbounded_channel::<Message>();
    let writer = tokio::spawn(async move {
        let stdin = child.stdin.take().unwrap();
        let mut bufr = BufWriter::new(stdin);
        while let Some(it) = writer_receiver.recv().await {
            let is_exit = matches!(&it, Message::Notification(n) if n.is_exit());
            debug!("sending message {:#?}", it);
            it.write(&mut bufr).await?;
            if is_exit {
                break;
            }
        }
        Ok(())
    });
    let (reader_sender, reader_receiver) = unbounded_channel::<Message>();
    let reader = tokio::spawn(async move {
        let stdout = child.stdout.take().unwrap();
        let mut reader = BufReader::new(stdout);
        while let Some(msg) = Message::read(&mut reader).await? {
            debug!("received message {:#?}", msg);
            reader_sender
                .send(msg)
                .expect("receiver was dropped, failed to send a message");
        }
        Ok(())
    });
    let threads = IoThreads { reader, writer };
    (writer_sender, reader_receiver, threads)
}

pub struct IoThreads {
    reader: JoinHandle<io::Result<()>>,
    writer: JoinHandle<io::Result<()>>,
}

impl IoThreads {
    pub async fn join(self) -> io::Result<()> {
        match self.reader.await? {
            Ok(_) => (),
            Err(err) => {
                error!("reader err: {err}");
            }
        }
        match self.writer.await? {
            Ok(_) => (),
            Err(err) => {
                error!("writer err: {err}");
            }
        }
        Ok(())
    }
}
