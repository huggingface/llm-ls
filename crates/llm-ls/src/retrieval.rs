use crate::config::LlmLsConfig;
use crate::error::{Error, Result};
use candle::utils::{cuda_is_available, metal_is_available};
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use gitignore::Gitignore;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::{path::PathBuf, sync::Arc};
use tinyvec_embed::db::{Collection, Compare, Db, Embedding, FilterBuilder, SimilarityResult};
use tinyvec_embed::similarity::Distance;
use tokenizers::{Encoding, Tokenizer, TruncationDirection};
use tokio::io::AsyncReadExt;
use tokio::task::spawn_blocking;
use tokio::time::Instant;
use tower_lsp::lsp_types::notification::Progress;
use tower_lsp::lsp_types::{
    NumberOrString, ProgressParams, ProgressParamsValue, Range, WorkDoneProgress,
    WorkDoneProgressReport,
};
use tower_lsp::Client;
use tracing::{debug, error, warn};

// TODO:
// - create sliding window and splitting of files logic
// - handle ipynb
// - handle updates

async fn file_is_empty(file_path: impl AsRef<Path>) -> Result<bool> {
    let mut content = String::new();
    tokio::fs::File::open(&file_path)
        .await?
        .read_to_string(&mut content)
        .await?;
    Ok(content.trim().is_empty())
}

fn is_code_file(file_name: &Path) -> bool {
    let code_extensions = [
        "ada",
        "adb",
        "ads",
        "c",
        "h",
        "cpp",
        "hpp",
        "cc",
        "cxx",
        "hxx",
        "cs",
        "css",
        "scss",
        "sass",
        "less",
        "java",
        "js",
        "jsx",
        "ts",
        "tsx",
        "php",
        "phtml",
        "html",
        "xml",
        "json",
        "yaml",
        "yml",
        "toml",
        "cfg",
        "conf",
        "sh",
        "bash",
        "zsh",
        "ps1",
        "psm1",
        "bat",
        "cmd",
        "py",
        "rb",
        "swift",
        "pl",
        "pm",
        "t",
        "r",
        "rs",
        "go",
        "kt",
        "kts",
        "sql",
        "md",
        "markdown",
        "txt",
        "lua",
        "ex",
        "exs",
        "erl",
        "rb",
        "scala",
        "sc",
        "ml",
        "mli",
        "zig",
        "clj",
        "cljs",
        "cljc",
        "cljx",
        "cr",
        "Dockerfile",
        "fs",
        "fsi",
        "fsx",
        "hs",
        "lhs",
        "groovy",
        "jsonnet",
        "jl",
        "nim",
        "rkt",
        "scm",
        "tf",
        "nix",
        "vue",
        "svelte",
        "lisp",
        "lsp",
        "el",
        "elc",
        "eln",
    ];

    let extension = file_name.extension().and_then(|ext| ext.to_str());

    if let Some(ext) = extension {
        code_extensions.contains(&ext.to_lowercase().as_str())
    } else {
        false
    }
}

async fn build_model_and_tokenizer() -> Result<(BertModel, Tokenizer)> {
    let start = Instant::now();
    let device = device(false)?;
    let model_id = "intfloat/multilingual-e5-small".to_string();
    let revision = "main".to_string();
    let repo = Repo::with_revision(model_id, RepoType::Model, revision);
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json").await?;
        let tokenizer = api.get("tokenizer.json").await?;
        let weights = api.get("pytorch_model.bin").await?;
        (config, tokenizer, weights)
    };
    let config = tokio::fs::read_to_string(config_filename).await?;
    let config: Config = serde_json::from_str(&config)?;
    let mut tokenizer: Tokenizer = Tokenizer::from_file(tokenizer_filename)?;
    tokenizer.with_padding(None);
    tokenizer.with_truncation(None)?;

    let vb = VarBuilder::from_pth(&weights_filename, DTYPE, &device)?;
    let model = BertModel::load(vb, &config)?;
    debug!(
        "loaded model and tokenizer in {} ms",
        start.elapsed().as_millis()
    );
    Ok((model, tokenizer))
}

fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            warn!("Running on CPU, to run on GPU(metal), use the `-metal` binary");
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            warn!("Running on CPU, to run on GPU, use the `-cuda` binary");
        }
        Ok(Device::Cpu)
    }
}

async fn initialse_database(cache_path: PathBuf) -> Db {
    let uri = cache_path.join("database");
    let mut db = Db::open(uri).await.expect("failed to open database");
    match db
        .create_collection("code-slices".to_owned(), 384, Distance::Cosine)
        .await
    {
        Ok(_)
        | Err(tinyvec_embed::error::Error::Collection(
            tinyvec_embed::error::Collection::UniqueViolation,
        )) => (),
        Err(err) => panic!("failed to create collection: {err}"),
    }
    db
}

pub(crate) struct Snippet {
    pub(crate) file_url: String,
    pub(crate) code: String,
}

impl TryFrom<&SimilarityResult> for Snippet {
    type Error = Error;

    fn try_from(value: &SimilarityResult) -> Result<Self> {
        let meta = value
            .embedding
            .metadata
            .as_ref()
            .ok_or(Error::MissingMetadata)?;
        let file_url = meta
            .get("file_url")
            .ok_or_else(|| Error::MalformattedEmbeddingMetadata("file_url".to_owned()))?
            .inner_string()?;
        let code = meta
            .get("snippet")
            .ok_or_else(|| Error::MalformattedEmbeddingMetadata("snippet".to_owned()))?
            .inner_string()?;
        Ok(Snippet { file_url, code })
    }
}

pub(crate) struct SnippetRetriever {
    db: Db,
    model: Arc<BertModel>,
    tokenizer: Tokenizer,
    window_size: usize,
    window_step: usize,
}

impl SnippetRetriever {
    /// # Panics
    ///
    /// Panics if the database cannot be initialised.
    pub(crate) async fn new(
        cache_path: PathBuf,
        window_size: usize,
        window_step: usize,
    ) -> Result<Self> {
        let (model, tokenizer) = build_model_and_tokenizer().await?;
        Ok(Self {
            db: initialse_database(cache_path).await,
            model: Arc::new(model),
            tokenizer,
            window_size,
            window_step,
        })
    }

    pub(crate) async fn build_workspace_snippets(
        &mut self,
        client: Client,
        config: Arc<LlmLsConfig>,
        token: NumberOrString,
        workspace_root: &str,
    ) -> Result<()> {
        debug!("building workspace snippets");
        let workspace_root = PathBuf::from(workspace_root);
        let mut files = Vec::new();
        let mut gitignore = Gitignore::parse(&workspace_root).ok();
        for pattern in config.ignored_paths.iter() {
            if let Some(gitignore) = gitignore.as_mut() {
                if let Err(err) = gitignore.add_rule(pattern.clone()) {
                    error!("failed to parse pattern: {err}");
                }
            };
        }

        client
            .send_notification::<Progress>(ProgressParams {
                token: token.clone(),
                value: ProgressParamsValue::WorkDone(WorkDoneProgress::Report(
                    WorkDoneProgressReport {
                        message: Some("listing workspace files".to_owned()),
                        ..Default::default()
                    },
                )),
            })
            .await;
        let mut stack = VecDeque::new();
        stack.push_back(workspace_root.clone());
        while let Some(src) = stack.pop_back() {
            let mut entries = tokio::fs::read_dir(&src).await?;
            while let Some(entry) = entries.next_entry().await? {
                let entry_type = entry.file_type().await?;

                let src_path = entry.path();

                if let Some(gitignore) = &gitignore {
                    if gitignore.ignored(&src_path)? {
                        continue;
                    }
                }

                if entry_type.is_dir() {
                    stack.push_back(src_path);
                } else if entry_type.is_file()
                    && is_code_file(&src_path)
                    && !file_is_empty(&src_path).await?
                {
                    files.push(src_path);
                }
            }
        }
        for (i, file) in files.iter().enumerate() {
            let file_url = file.to_str().expect("file path should be utf8").to_owned();
            self.add_document(file_url).await?;
            client
                .send_notification::<Progress>(ProgressParams {
                    token: token.clone(),
                    value: ProgressParamsValue::WorkDone(WorkDoneProgress::Report(
                        WorkDoneProgressReport {
                            message: Some(format!(
                                "{i}/{} ({})",
                                files.len(),
                                file.strip_prefix(workspace_root.as_path())?
                                    .to_str()
                                    .expect("expect file name to be valid unicode")
                            )),
                            ..Default::default()
                        },
                    )),
                })
                .await;
        }

        Ok(())
    }

    pub(crate) async fn add_document(&self, file_url: String) -> Result<()> {
        self.build_and_add_snippets(file_url, 0, None).await?;
        Ok(())
    }

    pub(crate) async fn update_document(&mut self, file_url: String, range: Range) -> Result<()> {
        self.build_and_add_snippets(
            file_url,
            range.start.line as usize,
            Some(range.end.line as usize),
        )
        .await?;
        Ok(())
    }

    pub(crate) async fn search(
        &self,
        snippet: String,
        filter: Option<FilterBuilder>,
    ) -> Result<Vec<Snippet>> {
        let col = self.db.get_collection("code-slices").await?;
        let mut encoding = self.tokenizer.encode(snippet.clone(), true)?;
        encoding.truncate(512, 1, TruncationDirection::Right);
        let query = self
            .generate_embedding(encoding, self.model.clone())
            .await?;
        let result = col
            .read()
            .await
            .get(&query, 5, filter)
            .await?
            .iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<_>>>()?;
        Ok(result)
    }

    pub(crate) async fn stop(&self) -> Result<()> {
        self.db.save().await?;
        Ok(())
    }

    pub(crate) async fn remove(&self, file_url: String, range: Range) -> Result<()> {
        let col = self.db.get_collection("code-slices").await?;
        col.write().await.remove(Some(
            Collection::filter()
                .comparison(
                    "start_line_no".to_owned(),
                    Compare::GtEq,
                    range.start.line.into(),
                )
                .and()
                .comparison(
                    "end_line_no".to_owned(),
                    Compare::LtEq,
                    range.end.line.into(),
                )
                .and()
                .comparison("file_url".to_owned(), Compare::Eq, file_url.into()),
        ))?;
        Ok(())
    }
}

impl SnippetRetriever {
    // TODO: handle overflowing in Encoding
    async fn generate_embedding(
        &self,
        encoding: Encoding,
        model: Arc<BertModel>,
    ) -> Result<Vec<f32>> {
        let start = Instant::now();
        let embedding = spawn_blocking(move || -> Result<Vec<f32>> {
            let tokens = encoding.get_ids().to_vec();
            let token_ids = Tensor::new(&tokens[..], &model.device)?.unsqueeze(0)?;
            let token_type_ids = token_ids.zeros_like()?;
            let embedding = model.forward(&token_ids, &token_type_ids)?;
            let (_n_sentence, n_tokens, _hidden_size) = embedding.dims3()?;
            let embedding = (embedding.sum(1)? / (n_tokens as f64))?;
            let embedding = embedding.get(0)?.to_vec1::<f32>()?;
            Ok(embedding)
        })
        .await?;
        debug!("embedding generated in {} ms", start.elapsed().as_millis());
        embedding
    }

    async fn build_and_add_snippets(
        &self,
        file_url: String,
        start: usize,
        end: Option<usize>,
    ) -> Result<()> {
        let col = self.db.get_collection("code-slices").await?;
        let file = tokio::fs::read_to_string(&file_url).await?;
        let lines = file.split('\n').collect::<Vec<_>>();
        let end = end.unwrap_or(lines.len()).min(lines.len());
        for start_line in (start..end).step_by(self.window_step) {
            let end_line = (start_line + self.window_size - 1).min(lines.len());
            if !col
                .read()
                .await
                .get(
                    &[],
                    1,
                    Some(
                        Collection::filter()
                            .comparison("file_url".to_owned(), Compare::Eq, file_url.clone().into())
                            .and()
                            .comparison("start_line_no".to_owned(), Compare::Eq, start_line.into())
                            .and()
                            .comparison("end_line_no".to_owned(), Compare::Eq, end_line.into()),
                    ),
                )
                .await?
                .is_empty()
            {
                debug!("snippet {file_url}[{start_line}, {end_line}] already indexed");
                continue;
            }
            let window = lines[start_line..end_line].to_vec();
            let snippet = window.join("\n");
            if snippet.is_empty() {
                continue;
            }

            let mut encoding = self.tokenizer.encode(snippet.clone(), true)?;
            encoding.truncate(512, 1, TruncationDirection::Right);
            let result = self.generate_embedding(encoding, self.model.clone()).await;
            let embedding = match result {
                Ok(e) => e,
                Err(err) => {
                    error!(
                        "error generating embedding for {file_url}[{start_line}, {end_line}]: {err}",
                    );
                    continue;
                }
            };
            col.write().await.insert(Embedding::new(
                embedding,
                Some(HashMap::from([
                    ("file_url".to_owned(), file_url.clone().into()),
                    ("start_line_no".to_owned(), start_line.into()),
                    ("end_line_no".to_owned(), end_line.into()),
                    ("snippet".to_owned(), snippet.clone().into()),
                ])),
            ))?;
        }
        Ok(())
    }
}
