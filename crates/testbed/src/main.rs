use std::{
    collections::{HashMap, VecDeque},
    fmt::Display,
    io::BufReader,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Instant,
};

use anyhow::anyhow;
use clap::Parser;
use futures_util::{stream::FuturesUnordered, StreamExt, TryStreamExt};
use lang::Language;
use lsp_client::{client::LspClient, msg::RequestId, server::Server};
use lsp_types::{
    DidOpenTextDocumentParams, InitializeParams, TextDocumentIdentifier, TextDocumentItem,
    TextDocumentPositionParams,
};
use ropey::Rope;
use runner::Runner;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;
use tokio::{
    fs::{self, read_to_string, File, OpenOptions},
    io::{self, AsyncReadExt, AsyncWriteExt},
    process::Command,
    sync::RwLock,
};
use tokio_util::compat::FuturesAsyncReadCompatExt;
use tracing::{debug, info, info_span, warn, Instrument};
use tracing_subscriber::EnvFilter;
use url::Url;

use crate::{
    runner::run_test,
    types::{
        FimParams, GetCompletions, GetCompletionsParams, GetCompletionsResult, Ide, RequestParams,
        TokenizerConfig,
    },
};

mod lang;
mod runner;
mod types;

/// Testbed runs llm-ls' code completion to measure its performance
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Hugging Face Inference API Token
    #[arg(short, long)]
    api_token: Option<String>,

    /// Path to llm-ls' binary
    #[arg(short, long)]
    llm_ls_bin_path: Option<String>,

    /// Path to the repositories.yaml file
    #[arg(long)]
    repos_file_path: Option<String>,

    /// Path to the local repositories/ directory
    #[arg(short, long)]
    repos_path: Option<String>,
}

#[derive(Clone, Deserialize, Serialize)]
struct LocalRepo {
    path: PathBuf,
}

#[derive(Clone, Deserialize, Serialize)]
struct GithubRepo {
    owner: String,
    name: String,
    revision: String,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
enum RepoSource {
    Local(LocalRepo),
    Github(GithubRepo),
}

#[derive(Clone, Deserialize, Serialize)]
struct Repository {
    build_command: String,
    build_args: Vec<String>,
    holes: Vec<Hole>,
    language: Language,
    runner: Runner,
    runner_command: Option<String>,
    setup_commands: Option<Vec<(String, Vec<String>)>>,
    source: RepoSource,
}

impl Repository {
    /// can panic if local path is not utf8
    fn name(&self) -> String {
        match &self.source {
            RepoSource::Local(local) => local.path.to_str().unwrap().to_owned(),
            RepoSource::Github(github) => format!("{}/{}", github.owner, github.name),
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
struct Hole {
    cursor: lsp_types::Position,
    /// relative path of a file in the repository
    file: String,
}

impl Display for Hole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} [{}, {}]",
            self.file, self.cursor.line, self.cursor.character
        )
    }
}

// unused for now, consider all holes as lines
// enum HoleType {
//     Line,
//     Multiline
// }

#[derive(Deserialize, Serialize)]
struct RepositoriesConfig {
    context_window: usize,
    fim: FimParams,
    model: String,
    request_params: RequestParams,
    repositories: Vec<Repository>,
    tls_skip_verify_insecure: bool,
    tokenizer_config: Option<TokenizerConfig>,
    tokens_to_clear: Vec<String>,
}

async fn get_api_token(args_token: Option<String>) -> anyhow::Result<Option<String>> {
    if args_token.is_some() {
        Ok(args_token)
    } else {
        let home_dir = home::home_dir().ok_or(anyhow!("failed to find home dir"))?;
        let cached_token = home_dir.join(".cache/huggingface/token");
        if cached_token.try_exists()? {
            let mut token = String::new();
            File::open(cached_token)
                .await?
                .read_to_string(&mut token)
                .await?;
            Ok(Some(token.trim().to_owned()))
        } else {
            Ok(None)
        }
    }
}

async fn download_repo_from_github(
    temp_dir: &TempDir,
    repo: &GithubRepo,
) -> anyhow::Result<PathBuf> {
    let repo_dir_name = format!("{}-{}", repo.name, repo.revision);
    let archive_path = temp_dir.path().join(format!("{}.zip", repo_dir_name));
    let mut archive = File::create(&archive_path).await?;
    let stream = reqwest::get(&format!(
        "https://github.com/{}/{}/archive/{}.zip",
        repo.owner, repo.name, repo.revision,
    ))
    .await?
    .error_for_status()?
    .bytes_stream();
    let stream = stream
        .map_err(|e| futures::io::Error::new(futures::io::ErrorKind::Other, e))
        .into_async_read();
    let mut stream = stream.compat();
    io::copy(&mut stream, &mut archive).await?;
    let archive = BufReader::new(std::fs::File::open(archive_path)?);
    zip::ZipArchive::new(archive)?.extract(temp_dir.path())?;
    Ok(temp_dir.path().join(repo_dir_name))
}

async fn copy_dir_contents(source: &Path, dest: &Path) -> anyhow::Result<()> {
    let mut stack = VecDeque::new();
    stack.push_back((source.to_path_buf(), dest.to_path_buf()));
    while let Some((src, dst)) = stack.pop_back() {
        let mut entries = fs::read_dir(&src).await?;
        while let Some(entry) = entries.next_entry().await? {
            let entry_type = entry.file_type().await?;

            let src_path = entry.path();
            let dst_path = fs::canonicalize(&dst).await?.join(entry.file_name());

            if entry_type.is_dir() {
                fs::create_dir(&dst_path).await?;
                stack.push_back((src_path, dst_path));
            } else if entry_type.is_file() {
                fs::copy(&src_path, &dst_path).await?;
            }
        }
    }

    Ok(())
}

async fn setup_repo_dir(
    repos_path_base: &Path,
    source: &RepoSource,
) -> anyhow::Result<(TempDir, PathBuf)> {
    match source {
        RepoSource::Local(local) => {
            debug!("setting up local repo: {}", local.path.to_str().unwrap());
            let temp_dir = TempDir::new()?;
            copy_dir_contents(&repos_path_base.join(&local.path), temp_dir.path()).await?;
            let repo_path = temp_dir.path().to_path_buf();
            Ok((temp_dir, repo_path))
        }
        RepoSource::Github(github) => {
            debug!("setting repo from github: {}/{}", github.owner, github.name);
            let temp_dir = TempDir::new()?;
            let repo_path = download_repo_from_github(&temp_dir, github).await?;
            Ok((temp_dir, repo_path))
        }
    }
}

async fn run_setup(
    commands: &Vec<(String, Vec<String>)>,
    repo_path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    for command in commands {
        let status = Command::new(&command.0)
            .args(&command.1)
            .current_dir(&repo_path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?
            .wait()
            .await?;
        if !status.success() {
            return Err(anyhow!(
                "error running: \"{} {}\"",
                command.0,
                command.1.join(" ")
            ));
        }
    }
    Ok(())
}

async fn build(
    command: &str,
    args: &Vec<String>,
    repo_path: impl AsRef<Path>,
) -> anyhow::Result<bool> {
    let status = Command::new(command)
        .args(args)
        .current_dir(repo_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?
        .wait()
        .await?;
    Ok(status.success())
}

#[allow(clippy::too_many_arguments)]
async fn complete_hole(
    idx: usize,
    hole: Hole,
    repo: Repository,
    client: Arc<LspClient>,
    file_cache: Arc<RwLock<HashMap<PathBuf, Rope>>>,
    repos_path_base: PathBuf,
    api_token: Option<String>,
    context_window: usize,
    fim: FimParams,
    model: String,
    request_params: RequestParams,
    tls_skip_verify_insecure: bool,
    tokens_to_clear: Vec<String>,
    tokenizer_config: Option<TokenizerConfig>,
) -> anyhow::Result<f32> {
    let span = info_span!(
        "complete_hole",
        repo_name = repo.name(),
        hole = hole.to_string()
    );
    async move {
        let hole_instant = Instant::now();
        let (_temp_dir, repo_path) = setup_repo_dir(&repos_path_base, &repo.source).await?;
        let file_path = repo_path.join(&hole.file);
        let file_path_str = file_path
            .to_str()
            .ok_or(anyhow!("failed to convert file to str"))?;
        let mut file_content = if file_cache.read().await.contains_key(&file_path) {
            file_cache
                .read()
                .await
                .get(&file_path)
                .ok_or(anyhow!("failed to find {} in file cache", file_path_str))?
                .to_owned()
        } else {
            let file_content = Rope::from_str(&read_to_string(&file_path).await?);
            file_cache
                .write()
                .await
                .insert(file_path.clone(), file_content.clone());
            file_content
        };
        let hole_start =
            file_content.line_to_char(hole.cursor.line as usize) + hole.cursor.character as usize;
        let hole_end = hole_start
            + file_content
                .line(hole.cursor.line as usize)
                .slice(hole.cursor.character as usize..)
                .len_chars()
            - 1;
        file_content.remove(hole_start..hole_end);

        let uri = Url::parse(&format!("file:/{file_path_str}"))?;
        client.send_notification::<lsp_types::notification::DidOpenTextDocument>(
            DidOpenTextDocumentParams {
                text_document: TextDocumentItem {
                    uri: uri.clone(),
                    language_id: repo.language.to_string(),
                    version: 0,
                    text: file_content.to_string(),
                },
            },
        );
        let response = client
            .send_request::<GetCompletions>(GetCompletionsParams {
                api_token,
                context_window,
                fim,
                ide: Ide::default(),
                model,
                request_params,
                text_document_position: TextDocumentPositionParams {
                    position: hole.cursor,
                    text_document: TextDocumentIdentifier { uri },
                },
                tls_skip_verify_insecure,
                tokens_to_clear,
                tokenizer_config,
            })
            .await?;
        let (_, result): (RequestId, GetCompletionsResult) = response.extract()?;
        file_content.insert(hole_start, &result.completions[0].generated_text);
        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&file_path)
            .await?;
        file.write_all(file_content.to_string().as_bytes()).await?;
        let test_percentage = if build(&repo.build_command, &repo.build_args, &repo_path).await? {
            run_test(repo.runner, &repo.runner_command, &repo_path).await?
        } else {
            0f32
        };
        info!("hole {idx} passed {}%", test_percentage * 100f32);
        info!(
            "checked completion in {} ms",
            hole_instant.elapsed().as_millis()
        );
        Ok(test_percentage)
    }
    .instrument(span)
    .await
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_target(true)
        .with_line_number(true)
        .with_env_filter(
            EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    let current_dir = std::env::current_dir()?;
    let llm_ls_path = if let Some(bin_path) = args.llm_ls_bin_path {
        bin_path.into()
    } else {
        current_dir.join("target/debug/llm-ls")
    };
    info!(
        "initializing language server at path: {}",
        llm_ls_path.to_str().unwrap()
    );
    let (conn, server) = Server::build().binary_path(llm_ls_path).start().await?;
    let client = Arc::new(LspClient::new(conn, server).await);
    client
        .send_request::<lsp_types::request::Initialize>(InitializeParams::default())
        .await?;

    let api_token = get_api_token(args.api_token).await?;
    let file_cache = Arc::new(RwLock::new(HashMap::new()));
    let repos_path_base = if let Some(path) = args.repos_path {
        path.into()
    } else {
        current_dir.join("crates/testbed/repositories")
    };
    let mut passing_tests_percentage = vec![];

    let repos_file_path = if let Some(path) = args.repos_file_path {
        path.into()
    } else {
        current_dir.join("crates/testbed/repositories.yaml")
    };
    let mut repos_file = String::new();
    File::open(&repos_file_path)
        .await?
        .read_to_string(&mut repos_file)
        .await?;
    let repos_config: RepositoriesConfig = serde_yaml::from_str(&repos_file)?;
    let RepositoriesConfig {
        context_window,
        fim,
        model,
        request_params,
        repositories,
        tls_skip_verify_insecure,
        tokenizer_config,
        tokens_to_clear,
    } = repos_config;
    let mut handles = FuturesUnordered::new();
    for repo in repositories {
        let (_temp_dir, setup_temp_dir) = setup_repo_dir(&repos_path_base, &repo.source).await?;
        if let Some(commands) = &repo.setup_commands {
            run_setup(commands, &setup_temp_dir).await?;
        }
        for (idx, hole) in repo.holes.iter().enumerate() {
            let hole = hole.clone();
            let repo = repo.clone();
            let client = client.clone();
            let file_cache = file_cache.clone();
            let repos_path_base = repos_path_base.clone();
            let api_token = api_token.clone();
            let fim = fim.clone();
            let model = model.clone();
            let request_params = request_params.clone();
            let tokens_to_clear = tokens_to_clear.clone();
            let tokenizer_config = tokenizer_config.clone();
            handles.push(tokio::spawn(async move {
                complete_hole(
                    idx,
                    hole,
                    repo,
                    client,
                    file_cache,
                    repos_path_base,
                    api_token,
                    context_window,
                    fim,
                    model,
                    request_params,
                    tls_skip_verify_insecure,
                    tokens_to_clear,
                    tokenizer_config,
                )
                .await
            }));
        }
    }

    while let Some(res) = handles.next().await {
        match res {
            Ok(Ok(percentage)) => passing_tests_percentage.push(percentage),
            Ok(Err(err)) => return Err(err),
            Err(err) => return Err(err.into()),
        }
    }
    info!(
        "llm-ls average completion success percentage: {:.2}%",
        passing_tests_percentage.iter().sum::<f32>() / passing_tests_percentage.len() as f32
            * 100f32
    );

    client.shutdown().await?;
    match Arc::into_inner(client) {
        Some(client) => client.exit().await,
        None => warn!("could not exit because client is referenced elsewhere"),
    }
    Ok(())
}
