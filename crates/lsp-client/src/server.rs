use std::{
    io::{self, BufReader},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::mpsc::{sync_channel, Receiver, SyncSender},
    thread,
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
    pub fn join(self) -> Result<()> {
        self.threads.join()?;
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

    pub fn start(self) -> Result<(Connection, Server)> {
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

fn stdio(mut child: Child) -> (SyncSender<Message>, Receiver<Message>, IoThreads) {
    let (writer_sender, writer_receiver) = sync_channel::<Message>(0);
    let writer = thread::spawn(move || {
        let mut stdin = child.stdin.take().unwrap();
        for it in writer_receiver.into_iter() {
            let is_exit = matches!(&it, Message::Notification(n) if n.is_exit());
            debug!("sending message {:#?}", it);
            it.write(&mut stdin)?;
            if is_exit {
                break;
            }
        }
        Ok(())
    });
    let (reader_sender, reader_receiver) = sync_channel::<Message>(0);
    let reader = thread::spawn(move || {
        let stdout = child.stdout.take().unwrap();
        let mut reader = BufReader::new(stdout);
        while let Some(msg) = Message::read(&mut reader)? {
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
    reader: thread::JoinHandle<io::Result<()>>,
    writer: thread::JoinHandle<io::Result<()>>,
}

impl IoThreads {
    pub fn join(self) -> io::Result<()> {
        match self.reader.join() {
            Ok(r) => r?,
            Err(err) => {
                error!("reader panicked!");
                std::panic::panic_any(err)
            }
        }
        match self.writer.join() {
            Ok(r) => r,
            Err(err) => {
                error!("writer panicked!");
                std::panic::panic_any(err);
            }
        }
    }
}
