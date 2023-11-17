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
use lsp_client::{client::LspClient, error::ExtractError, msg::RequestId, server::Server};
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
    sync::{OnceCell, RwLock, Semaphore},
};
use tokio_util::compat::FuturesAsyncReadCompatExt;
use tracing::{debug, error, info, info_span, warn, Instrument};
use tracing_subscriber::EnvFilter;
use url::Url;

use crate::{
    holes_generator::generate_holes,
    runner::run_test,
    types::{
        FimParams, GetCompletions, GetCompletionsParams, GetCompletionsResult, Ide, RequestParams,
        TokenizerConfig,
    },
};

mod holes_generator;
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

    /// Comma separated list of repos in the repositories file to run completions or holes generation for;
    /// matches on path for local repos and `owner/name` for github repos
    #[arg(short, long)]
    filter: Option<String>,

    /// When this is specified, holes files will be generated based on the repositories.yaml file
    #[arg(short, long, action)]
    generate_holes: bool,

    /// Path to the directory containing the holes files
    #[arg(short = 'H', long)]
    holes_dir_path: Option<String>,

    /// Number of holes to create per repository
    #[arg(short = 'n', long, default_value_t = 100)]
    holes_per_repo: usize,

    /// Path to llm-ls' binary
    #[arg(short, long)]
    llm_ls_bin_path: Option<String>,

    /// Concurrent hole completions number
    #[arg(short, long, default_value_t = 8)]
    parallel_hole_completions: usize,

    /// Path to the local repositories/ directory
    #[arg(short = 'R', long)]
    repos_dir_path: Option<String>,

    /// Path to the repositories.yaml file
    #[arg(short, long)]
    repos_file_path: Option<String>,
}

#[derive(Clone, Deserialize, Serialize)]
struct LocalRepo {
    path: PathBuf,
    src_path: String,
    #[serde(default)]
    exclude_paths: Vec<String>,
}

#[derive(Clone, Deserialize, Serialize)]
struct GithubRepo {
    owner: String,
    name: String,
    revision: String,
    #[serde(default)]
    src_path: String,
    #[serde(default)]
    exclude_paths: Vec<String>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
enum RepoSource {
    Local(LocalRepo),
    Github(GithubRepo),
}

impl RepoSource {
    fn source_type(&self) -> String {
        match self {
            Self::Local { .. } => "local".to_owned(),
            Self::Github { .. } => "github".to_owned(),
        }
    }

    fn src_path(&self) -> String {
        match self {
            Self::Local(local) => local.src_path.clone(),
            Self::Github(github) => github.src_path.clone(),
        }
    }

    fn exclude_paths(&self) -> Vec<String> {
        match self {
            Self::Local(local) => local.exclude_paths.clone(),
            Self::Github(github) => github.exclude_paths.clone(),
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
struct Repository {
    build_command: String,
    build_args: Vec<String>,
    env: Option<Vec<String>>,
    holes_file: String,
    language: Language,
    runner: Runner,
    runner_command: Option<String>,
    runner_args: Option<Vec<String>>,
    #[serde(default)]
    runner_extra_args: Vec<String>,
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

impl Hole {
    fn new(line: u32, character: u32, file: String) -> Self {
        Self {
            cursor: lsp_types::Position::new(line, character),
            file,
        }
    }
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

#[derive(Clone, Deserialize, Serialize)]
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

struct HoleCompletionResult {
    repo_name: String,
    repo_source_type: String,
    pass_percentage: f32,
    completion_time_ms: u128,
}

impl HoleCompletionResult {
    fn new(
        repo_name: String,
        repo_source_type: String,
        pass_percentage: f32,
        completion_time_ms: u128,
    ) -> Self {
        Self {
            repo_name,
            repo_source_type,
            pass_percentage,
            completion_time_ms,
        }
    }
}

struct SetupCache {
    cache: HashMap<String, OnceCell<(TempDir, PathBuf)>>,
}

impl SetupCache {
    fn new(repositories: &Vec<Repository>) -> Self {
        let mut cache = HashMap::new();
        for repo in repositories {
            cache.insert(repo.name(), OnceCell::new());
        }
        Self { cache }
    }

    async fn get_setup_cache(
        &self,
        repos_dir_path: PathBuf,
        repo: Repository,
    ) -> anyhow::Result<&(TempDir, PathBuf)> {
        self.cache
            .get(&repo.name())
            .ok_or(anyhow!(
                "failed to find setup cache for repo {}",
                repo.name()
            ))?
            .get_or_try_init(|| async move {
                let (temp_dir, repo_path) = setup_repo_dir(&repos_dir_path, &repo.source).await?;
                if let Some(commands) = &repo.setup_commands {
                    run_setup(commands, &repo.env, &repo_path).await?;
                }
                Ok((temp_dir, repo_path))
            })
            .await
    }

    async fn create_cache_copy(
        &self,
        repos_dir_path: PathBuf,
        repo: Repository,
    ) -> anyhow::Result<TempDir> {
        let (_cached_dir, path_in_dir) = self.get_setup_cache(repos_dir_path, repo).await?;
        let temp_dir = TempDir::new()?;
        copy_dir_contents(path_in_dir, temp_dir.path()).await?;
        Ok(temp_dir)
    }
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
    debug!(
        "copying files from {} to {}",
        source.to_str().unwrap(),
        dest.to_str().unwrap()
    );
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
            } else if entry_type.is_symlink() {
                let link_target = fs::read_link(&src_path).await?;
                fs::symlink(link_target, dst_path.clone()).await?;
            }
        }
    }

    Ok(())
}

async fn setup_repo_dir(
    repos_dir_path: &Path,
    source: &RepoSource,
) -> anyhow::Result<(TempDir, PathBuf)> {
    match source {
        RepoSource::Local(local) => {
            debug!("setting up local repo: {}", local.path.to_str().unwrap());
            let temp_dir = TempDir::new()?;
            copy_dir_contents(&repos_dir_path.join(&local.path), temp_dir.path()).await?;
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

fn parse_env(env: &Option<Vec<String>>) -> anyhow::Result<Vec<(String, String)>> {
    let mut env_vars = vec![];
    if let Some(env) = env {
        for var in env {
            env_vars.push(
                var.split_once('=')
                    .map(|(n, v)| (n.to_owned(), v.to_owned()))
                    .ok_or(anyhow!("failed to split env var {var}"))?,
            );
        }
    }
    Ok(env_vars)
}

async fn run_setup(
    commands: &Vec<(String, Vec<String>)>,
    env: &Option<Vec<String>>,
    repo_path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let parsed_env = parse_env(env)?;
    for command in commands {
        let mut status_cmd = Command::new(&command.0);
        for (name, value) in &parsed_env {
            status_cmd.env(name, value);
        }
        debug!(
            "running setup command: {} {}",
            command.0,
            command.1.join(" ")
        );
        let status = status_cmd
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
    env: &Option<Vec<String>>,
    repo_path: impl AsRef<Path>,
) -> anyhow::Result<bool> {
    let parsed_env = parse_env(env)?;
    let mut status_cmd = Command::new(command);
    for (name, value) in parsed_env {
        status_cmd.env(name, value);
    }
    debug!("building repo: {command} {args:?}");
    let status = status_cmd
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
async fn complete_holes(
    hole: Hole,
    repo: Repository,
    client: Arc<LspClient>,
    file_cache: Arc<RwLock<HashMap<PathBuf, Rope>>>,
    repos_dir_path: PathBuf,
    repos_config: RepositoriesConfig,
    api_token: Option<String>,
    semaphore: Arc<Semaphore>,
    setup_cache: Arc<SetupCache>,
) -> anyhow::Result<HoleCompletionResult> {
    let permit = semaphore.acquire_owned().await?;
    let span = info_span!("complete_hole", repo_name = repo.name());
    let RepositoriesConfig {
        context_window,
        fim,
        model,
        request_params,
        tls_skip_verify_insecure,
        tokenizer_config,
        tokens_to_clear,
        ..
    } = repos_config;
    async move {
        let tmp_dir = setup_cache
            .create_cache_copy(repos_dir_path, repo.clone())
            .await?;
        let repo_path = tmp_dir.path();
        let hole_instant = Instant::now();
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
        let original_content = file_content.clone();
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
                api_token: api_token.clone(),
                context_window,
                fim: fim.clone(),
                ide: Ide::default(),
                model: model.clone(),
                request_params: request_params.clone(),
                text_document_position: TextDocumentPositionParams {
                    position: hole.cursor,
                    text_document: TextDocumentIdentifier { uri },
                },
                tls_skip_verify_insecure,
                tokens_to_clear: tokens_to_clear.clone(),
                tokenizer_config: tokenizer_config.clone(),
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
        let test_percentage =
            if build(&repo.build_command, &repo.build_args, &repo.env, &repo_path).await? {
                run_test(
                    repo.runner,
                    &repo.runner_command,
                    &repo.runner_args,
                    &mut repo.runner_extra_args.clone(),
                    &repo.env,
                    repo_path,
                )
                .await?
            } else {
                0f32
            };
        debug!("{} passed {}%", hole.to_string(), test_percentage * 100f32);
        let hole_completions_result = HoleCompletionResult::new(
            repo.name(),
            repo.source.source_type(),
            test_percentage,
            hole_instant.elapsed().as_millis(),
        );
        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&file_path)
            .await?;
        file.write_all(original_content.to_string().as_bytes())
            .await?;
        drop(permit);
        Ok(hole_completions_result)
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

    let api_token = get_api_token(args.api_token).await?;
    let current_dir = std::env::current_dir()?;
    let llm_ls_path = if let Some(bin_path) = args.llm_ls_bin_path {
        bin_path.into()
    } else {
        current_dir.join("target/release/llm-ls")
    };

    let repos_dir_path = if let Some(path) = args.repos_dir_path {
        path.into()
    } else {
        current_dir.join("crates/testbed/repositories")
    };

    let repos_file_path = if let Some(path) = args.repos_file_path {
        path.into()
    } else {
        current_dir.join("crates/testbed/repositories.yaml")
    };

    let holes_dir_path = if let Some(path) = args.holes_dir_path {
        path.into()
    } else {
        current_dir.join("crates/testbed/holes")
    };

    let (filter_repos, filter_list) = if let Some(filter) = args.filter {
        (true, filter.split(',').map(|s| s.to_owned()).collect())
    } else {
        (false, vec![])
    };

    let mut repos_file = String::new();
    File::open(&repos_file_path)
        .await?
        .read_to_string(&mut repos_file)
        .await?;
    let repos_config: RepositoriesConfig = serde_yaml::from_str(&repos_file)?;
    if args.generate_holes {
        return generate_holes(
            repos_config,
            &repos_dir_path,
            &holes_dir_path,
            args.holes_per_repo,
            filter_repos,
            filter_list,
        )
        .await;
    }

    debug!(
        "initializing language server at path: {}",
        llm_ls_path.to_str().unwrap()
    );
    let (conn, server) = Server::build().binary_path(llm_ls_path).start().await?;
    let client = Arc::new(LspClient::new(conn, server).await);
    client
        .send_request::<lsp_types::request::Initialize>(InitializeParams::default())
        .await?;

    let file_cache = Arc::new(RwLock::new(HashMap::new()));
    let mut passing_tests_percentage = vec![];

    let repositories = repos_config.repositories.clone();
    let setup_cache = Arc::new(SetupCache::new(&repositories));
    let mut handles = FuturesUnordered::new();
    let semaphore = Arc::new(Semaphore::new(args.parallel_hole_completions));
    for repo in repositories {
        if filter_repos && !filter_list.contains(&repo.name()) {
            continue;
        }
        let holes_file_path = holes_dir_path.join(&repo.holes_file);
        let mut holes = String::new();
        File::open(holes_file_path)
            .await?
            .read_to_string(&mut holes)
            .await?;
        let holes: Vec<Hole> = serde_json::from_str(&holes)?;
        info!("running {} hole completions", holes.len());
        for hole in holes {
            let repo = repo.clone();
            let client = client.clone();
            let file_cache = file_cache.clone();
            let repos_dir_path = repos_dir_path.clone();
            let repos_config = repos_config.clone();
            let api_token = api_token.clone();
            let semaphore = semaphore.clone();
            let setup_cache = setup_cache.clone();
            handles.push(tokio::spawn(async move {
                complete_holes(
                    hole,
                    repo,
                    client,
                    file_cache,
                    repos_dir_path,
                    repos_config,
                    api_token,
                    semaphore,
                    setup_cache,
                )
                .await
            }));
        }
    }

    while let Some(res) = handles.next().await {
        match res {
            Ok(Ok(res)) => passing_tests_percentage.push(res),
            Ok(Err(err)) => {
                if let Some(extract_err) = err.downcast_ref::<ExtractError>() {
                    error!("llm-ls response error: {extract_err}");
                } else {
                    return Err(err);
                }
            }
            Err(err) => return Err(err.into()),
        }
    }
    let mut results_map: HashMap<(String, String), (u128, f32, f32)> = HashMap::new();
    for res in passing_tests_percentage {
        results_map
            .entry((res.repo_name, res.repo_source_type))
            .and_modify(|p| {
                p.0 += res.completion_time_ms;
                p.1 += res.pass_percentage;
                p.2 += 1f32;
            })
            .or_insert((res.completion_time_ms, res.pass_percentage, 1f32));
    }
    let json_result = results_map
        .iter()
        .map(|(k, v)| {
            let avg_hole_completion_time_ms = v.0 as f32 / v.2 / 1_000f32;
            let pass_percentage = v.1 / v.2 * 100f32;
            info!(
                "{} from {} obtained {:.2}% in {:.3}s",
                k.0, k.1, pass_percentage, avg_hole_completion_time_ms
            );
            serde_json::json!({
                "repo_name": k.0,
                "source_type": k.1,
                "avg_hole_completion_time_ms": format!("{:.3}", avg_hole_completion_time_ms),
                "pass_percentage": format!("{:.2}", pass_percentage),
            })
        })
        .collect::<Vec<serde_json::Value>>();
    OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("results.json")
        .await?
        .write_all(serde_json::to_string(&json_result)?.as_bytes())
        .await?;

    info!("all tests were run, exiting");
    client.shutdown().await?;
    match Arc::into_inner(client) {
        Some(client) => client.exit().await,
        None => warn!("could not send exit notification because client is referenced elsewhere"),
    }
    Ok(())
}
