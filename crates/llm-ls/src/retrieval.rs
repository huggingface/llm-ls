use crate::error::Result;
use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use candle::utils::{cuda_is_available, metal_is_available};
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use futures_util::StreamExt;
use gitignore::Gitignore;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use lance_linalg::distance::MetricType;
use std::collections::VecDeque;
use std::path::Path;
use std::{path::PathBuf, sync::Arc};
use tokenizers::Tokenizer;
use tokio::io::AsyncReadExt;
use tokio::task::spawn_blocking;
use tower_lsp::lsp_types::notification::Progress;
use tower_lsp::lsp_types::{
    NumberOrString, ProgressParams, ProgressParamsValue, Range, WorkDoneProgress,
    WorkDoneProgressReport,
};
use tower_lsp::Client;
use tracing::{error, info, warn};
use vectordb::error::Error;
use vectordb::table::ReadParams;
use vectordb::{Connection, Database, Table};

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
        "ini",
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
    let device = device(false)?;
    let model_id = "bigcode/starencoder".to_string();
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
    let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

    let vb = VarBuilder::from_pth(&weights_filename, DTYPE, &device)?;
    let model = BertModel::load(vb, &config)?;
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

async fn initialse_database(cache_path: PathBuf) -> Arc<dyn Table> {
    let uri = cache_path.join("database");
    let db = Database::connect(uri.to_str().expect("path should be utf8"))
        .await
        .expect("failed to open database");
    match db
        .open_table_with_params("code-slices", ReadParams::default())
        .await
    {
        Ok(table) => table,
        Err(vectordb::error::Error::TableNotFound { .. }) => {
            let schema = Arc::new(Schema::new(vec![
                Field::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        768,
                    ),
                    false,
                ),
                Field::new("content", DataType::Utf8, false),
                Field::new("file_url", DataType::Utf8, false),
                Field::new("start_line_no", DataType::UInt32, false),
                Field::new("end_line_no", DataType::UInt32, false),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(FixedSizeListBuilder::new(Float32Builder::new(), 768).finish()),
                    Arc::new(StringArray::from(Vec::<&str>::new())),
                    Arc::new(StringArray::from(Vec::<&str>::new())),
                    Arc::new(UInt32Array::from(Vec::<u32>::new())),
                    Arc::new(UInt32Array::from(Vec::<u32>::new())),
                ],
            )
            .expect("failure while defining schema");
            db.create_table(
                "code-slices",
                Box::new(RecordBatchIterator::new(
                    vec![batch].into_iter().map(Ok),
                    schema,
                )),
                None,
            )
            .await
            .expect("failed to create table")
        }
        Err(err) => panic!("error while opening table: {}", err),
    }
}

pub(crate) struct SnippetRetriever {
    db: Arc<dyn Table>,
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
        token: NumberOrString,
        workspace_root: &str,
    ) -> Result<()> {
        info!("building workspace snippets");
        let workspace_root = PathBuf::from(workspace_root);
        let mut files = Vec::new();
        let gitignore = Gitignore::parse(&workspace_root).ok();

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
        stack.push_back(workspace_root);
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
            let file_url = file.to_str().expect("file path should be utf8").to_string();
            self.add_document(file_url).await?;
            client
                .send_notification::<Progress>(ProgressParams {
                    token: token.clone(),
                    value: ProgressParamsValue::WorkDone(WorkDoneProgress::Report(
                        WorkDoneProgressReport {
                            message: Some(format!("({i}/{}) done", files.len())),
                            ..Default::default()
                        },
                    )),
                })
                .await;
        }

        Ok(())
    }

    pub(crate) async fn add_document(&mut self, file_url: String) -> Result<()> {
        let file = tokio::fs::read_to_string(&file_url).await?;
        let lines = file.split('\n').collect::<Vec<_>>();
        let mut embeddings = FixedSizeListBuilder::new(Float32Builder::new(), 768);
        let mut snippets = Vec::new();
        let mut file_urls = Vec::new();
        let mut start_line_no = Vec::new();
        let mut end_line_no = Vec::new();
        for start_line in (0..lines.len()).step_by(self.window_step) {
            let end_line = (start_line + self.window_size - 1).min(lines.len());
            if self
                .exists(format!(
                    "file_url = '{file_url}' AND start_line_no = {start_line} AND end_line_no = {end_line}"
                ))
                .await?
            {
                info!("snippet {file_url}[{start_line}, {end_line}] already indexed");
                continue;
            }
            let window = lines[start_line..end_line].to_vec();
            let snippet = window.join("\n");
            if snippet.is_empty() {
                continue;
            }
            if snippet.len() > 1024 {
                warn!("snippet {file_url}[{start_line}, {end_line}] is too big to be indexed");
                continue;
            }
            let tokenizer = self
                .tokenizer
                .with_padding(None)
                .with_truncation(None)?
                .clone();
            let model = self.model.clone();
            let snippet_clone = snippet.clone();
            let result = spawn_blocking(move || -> Result<Vec<f32>> {
                let tokens = tokenizer.encode(snippet_clone, true)?.get_ids().to_vec();
                let token_ids = Tensor::new(&tokens[..], &model.device)?.unsqueeze(0)?;
                let token_type_ids = token_ids.zeros_like()?;
                let embedding = model.forward(&token_ids, &token_type_ids)?;
                let (_n_sentence, n_tokens, _hidden_size) = embedding.dims3()?;
                let embedding = (embedding.sum(1)? / (n_tokens as f64))?;
                let embedding = embedding.get(0)?.to_vec1::<f32>()?;
                Ok(embedding)
            })
            .await?;
            let embedding = match result {
                Ok(e) => e,
                Err(err) => {
                    error!(
                        "error generating embedding for {file_url}[{start_line}, {end_line}]: {err}",
                    );
                    continue;
                }
            };
            embeddings.values().append_slice(&embedding);
            embeddings.append(true);
            snippets.push(snippet.clone());
            file_urls.push(file_url.clone());
            start_line_no.push(start_line as u32);
            end_line_no.push(end_line as u32);
        }

        let batch = RecordBatch::try_new(
            self.db.schema(),
            vec![
                Arc::new(embeddings.finish()),
                Arc::new(StringArray::from(snippets)),
                Arc::new(StringArray::from(file_urls)),
                Arc::new(UInt32Array::from(start_line_no)),
                Arc::new(UInt32Array::from(end_line_no)),
            ],
        )?;
        self.db
            .add(
                Box::new(RecordBatchIterator::new(
                    vec![batch].into_iter().map(Ok),
                    self.db.schema(),
                )),
                None,
            )
            .await?;
        Ok(())
    }

    pub(crate) async fn update_document(&mut self, file_url: String, range: Range) {
        // TODO:
        // - delete elements matching Range
        //   - keep the smallest start line to create new windows from
        // - build new windows based on range
        // - insert them into table
    }

    pub(crate) async fn exists(&self, filter: String) -> Result<bool> {
        let mut results = self
            .db
            .search(&[0.])
            .metric_type(MetricType::Cosine)
            .filter(&filter)
            .execute_stream()
            .await?;
        let exists = if let Some(record_batch) = results.next().await {
            let record_batch = record_batch.map_err(Into::<Error>::into)?;
            if record_batch.num_rows() > 0 {
                true
            } else {
                info!("record batch: {record_batch:?}");
                false
            }
        } else {
            false
        };
        if !exists {
            info!("filter: {filter}");
        }
        Ok(exists)
    }

    pub(crate) async fn search(&self, snippet: String, filter: &str) -> Result<String> {
        Ok("toto".to_string())
    }
}
