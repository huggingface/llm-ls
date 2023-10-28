use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{self, BufReader, BufWriter, Read},
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::anyhow;
use clap::Parser;
use lang::Language;
use lsp_client::{client::LspClient, msg::RequestId, server::Server};
use lsp_types::{
    DidOpenTextDocumentParams, InitializeParams, TextDocumentIdentifier, TextDocumentItem,
    TextDocumentPositionParams,
};
use ropey::Rope;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;
use url::Url;

use crate::{
    runner::get_runner_map,
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

#[derive(Deserialize, Serialize)]
struct LocalRepo {
    path: PathBuf,
}

#[derive(Deserialize, Serialize)]
struct GithubRepo {
    owner: String,
    name: String,
    revision: String,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
enum RepoSource {
    Local(LocalRepo),
    Github(GithubRepo),
}

#[derive(Deserialize, Serialize)]
struct Repository {
    build_command: String,
    build_args: Vec<String>,
    holes: Vec<Hole>,
    language: Language,
    runner: String,
    runner_cmd: Option<String>,
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

#[derive(Deserialize, Serialize)]
struct Hole {
    cursor: lsp_types::Position,
    /// relative path of a file in the repository
    file: String,
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

fn get_api_token(args_token: Option<String>) -> anyhow::Result<Option<String>> {
    if args_token.is_some() {
        Ok(args_token)
    } else {
        let home_dir = home::home_dir().ok_or(anyhow!("failed to find home dir"))?;
        let cached_token = home_dir.join(".cache/huggingface/token");
        if cached_token.exists() {
            let mut token = String::new();
            File::open(cached_token)?.read_to_string(&mut token)?;
            Ok(Some(token.trim().to_owned()))
        } else {
            Ok(None)
        }
    }
}

fn download_repo_from_github(temp_dir: &TempDir, repo: &GithubRepo) -> anyhow::Result<()> {
    let repo_dir_name = format!("{}-{}", repo.name, repo.revision);
    let archive_path = temp_dir.path().join(format!("{}.zip", repo_dir_name));
    let mut archive = File::create(&archive_path)?;
    let resp = ureq::get(&format!(
        "https://github.com/{}/{}/archive/{}.zip",
        repo.owner, repo.name, repo.revision,
    ))
    .call()?;
    if resp.status() != 200 {
        return Err(anyhow!("error fetching archive: {}", resp.status()));
    }
    io::copy(&mut resp.into_reader(), &mut archive)?;
    let archive = BufReader::new(File::open(archive_path)?);
    zip::ZipArchive::new(archive)?.extract(temp_dir.path())?;
    copy_dir_contents(temp_dir.path().join(repo_dir_name), temp_dir.path())?;
    Ok(())
}

fn copy_dir_contents(src: impl AsRef<Path>, dest: impl AsRef<Path>) -> anyhow::Result<()> {
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let entry_type = entry.file_type()?;

        let src_path = entry.path();
        let dest_path = fs::canonicalize(&dest)?.join(entry.file_name());

        if entry_type.is_dir() {
            fs::create_dir(&dest_path)?;
            copy_dir_contents(&src_path, &dest_path)?;
        } else if entry_type.is_file() {
            fs::copy(&src_path, &dest_path)?;
        }
    }

    Ok(())
}

fn setup_repo_dir(repos_path_base: &Path, source: &RepoSource) -> anyhow::Result<TempDir> {
    match source {
        RepoSource::Local(local) => {
            debug!("setting up local repo: {}", local.path.to_str().unwrap());
            let temp_dir = TempDir::new()?;
            copy_dir_contents(repos_path_base.join(&local.path), temp_dir.path())?;
            Ok(temp_dir)
        }
        RepoSource::Github(github) => {
            debug!("setting repo from github: {}/{}", github.owner, github.name);
            let temp_dir = TempDir::new()?;
            download_repo_from_github(&temp_dir, github)?;
            Ok(temp_dir)
        }
    }
}

fn run_setup(
    commands: &Vec<(String, Vec<String>)>,
    repo_path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    for command in commands {
        let status = Command::new(&command.0)
            .args(&command.1)
            .current_dir(&repo_path)
            .spawn()?
            .wait()?;
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

fn build(command: &str, args: &Vec<String>, repo_path: impl AsRef<Path>) -> anyhow::Result<bool> {
    let status = Command::new(command)
        .args(args)
        .current_dir(repo_path)
        .spawn()?
        .wait()?;
    Ok(status.success())
}

fn main() -> anyhow::Result<()> {
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
    let (conn, server) = Server::build().binary_path(llm_ls_path).start()?;
    let client = LspClient::new(conn, server);
    client.send_request::<lsp_types::request::Initialize>(InitializeParams::default());

    let api_token = get_api_token(args.api_token)?;
    let mut file_cache: HashMap<PathBuf, Rope> = HashMap::new();
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
    let repos_file = std::fs::File::open(&repos_file_path)?;
    let repos_config: RepositoriesConfig = serde_yaml::from_reader(repos_file)?;
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
    let runner_map = get_runner_map();
    for repo in repositories {
        // TODO:
        // - run hole completion in parallel
        info!("testing completions on {}", repo.name());
        let setup_temp_dir = setup_repo_dir(&repos_path_base, &repo.source)?;
        if let Some(commands) = &repo.setup_commands {
            run_setup(&commands, &setup_temp_dir.path())?;
        }
        for hole in repo.holes {
            let temp_dir = setup_repo_dir(&repos_path_base, &repo.source)?;
            let repo_path = temp_dir.path();
            let file_path = repo_path.join(hole.file);
            let file_path_str = file_path
                .to_str()
                .ok_or(anyhow!("failed to convert file to str"))?;
            let mut file_content = if file_cache.contains_key(&file_path) {
                file_cache
                    .get(&file_path)
                    .ok_or(anyhow!("failed to find {} in file cache", file_path_str))?
                    .to_owned()
            } else {
                let file_content = Rope::from_reader(BufReader::new(File::open(&file_path)?))?;
                file_cache.insert(file_path.clone(), file_content.clone());
                file_content
            };
            let hole_start = file_content.line_to_char(hole.cursor.line as usize)
                + hole.cursor.character as usize;
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
            let response = client.send_request::<GetCompletions>(GetCompletionsParams {
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
            });
            let (_, result): (RequestId, GetCompletionsResult) = response.extract()?;
            file_content.insert(hole_start, &result.completions[0].generated_text);
            file_content.write_to(BufWriter::new(
                OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .open(&file_path)?,
            ))?;
            if build(&repo.build_command, &repo.build_args, &repo_path)? {
                let runner = runner_map
                    .get(&repo.runner)
                    .ok_or(anyhow!("could not find runner named {}", repo.runner))?;
                passing_tests_percentage.push((runner)(&repo.runner_cmd, &repo_path)?);
            } else {
                passing_tests_percentage.push(0f32);
            }
        }
    }

    info!(
        "llm-ls average completion success percentage: {:.2}%",
        passing_tests_percentage.iter().sum::<f32>() / passing_tests_percentage.len() as f32
            * 100f32
    );

    client.shutdown();
    client.exit();
    Ok(())
}
