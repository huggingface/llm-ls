use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
};

use anyhow::anyhow;
use rand::{seq::SliceRandom, Rng};
use ropey::Rope;
use tokio::{
    fs::{self, OpenOptions},
    io::{AsyncReadExt, AsyncWriteExt},
};
use tracing::info;

use crate::{setup_repo_dir, Hole, RepositoriesConfig};

async fn file_is_empty(file_path: impl AsRef<Path>) -> anyhow::Result<bool> {
    let mut content = String::new();
    fs::File::open(&file_path)
        .await?
        .read_to_string(&mut content)
        .await?;
    Ok(content.trim().is_empty())
}

pub(crate) async fn generate_holes(
    repositories_config: RepositoriesConfig,
    repos_dir_path: &Path,
    holes_dir_path: &Path,
    holes_per_repo: usize,
    filter_repos: bool,
    filter_list: Vec<String>,
) -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();
    for repo in repositories_config.repositories {
        if filter_repos && !filter_list.contains(&repo.name()) {
            continue;
        }
        let repo_name = repo.name();
        info!("creating {} holes for {}", holes_per_repo, repo_name);
        let (_tmp_dir, path) = setup_repo_dir(repos_dir_path, &repo.source).await?;
        let mut files = vec![];

        let mut stack = VecDeque::new();
        let exclude_paths = repo
            .source
            .exclude_paths()
            .iter()
            .map(|p| path.join(p))
            .collect::<Vec<PathBuf>>();
        stack.push_back(path.join(repo.source.src_path()));
        while let Some(src) = stack.pop_back() {
            let mut entries = fs::read_dir(&src).await?;
            while let Some(entry) = entries.next_entry().await? {
                let entry_type = entry.file_type().await?;

                let src_path = entry.path();
                if exclude_paths.iter().any(|p| src_path.starts_with(p)) {
                    continue;
                }

                if entry_type.is_dir() {
                    stack.push_back(src_path);
                } else if entry_type.is_file()
                    && repo
                        .language
                        .is_code_file(src_path.file_name().unwrap().to_str().unwrap())
                    && !file_is_empty(&src_path).await?
                {
                    files.push(src_path);
                }
            }
        }

        let mut holes = vec![];
        let mut i = 0;
        while i < holes_per_repo {
            let file_path = files
                .choose(&mut rng)
                .ok_or(anyhow!("files vec is empty"))?;
            let mut content = String::new();
            fs::File::open(&file_path)
                .await?
                .read_to_string(&mut content)
                .await?;
            let rope = Rope::from_str(&content);
            let line_nb = rng.gen_range(0..rope.len_lines());
            let line = rope.line(line_nb);
            let line_string = line.to_string();
            let trimmed = line_string.trim();
            if trimmed.starts_with(repo.language.comment_token()) || trimmed.is_empty() {
                continue;
            }
            let column_nb = rng.gen_range(0..15.min(line.len_chars()));
            holes.push(Hole::new(
                line_nb as u32,
                column_nb as u32,
                file_path.strip_prefix(&path)?.to_str().unwrap().to_owned(),
            ));
            i += 1;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&holes_dir_path.join(repo.holes_file))
            .await?;
        file.write_all(serde_json::to_string(&holes)?.as_bytes())
            .await?;
    }

    Ok(())
}
