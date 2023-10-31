use std::{collections::VecDeque, path::Path};

use rand::Rng;
use ropey::Rope;
use tokio::{
    fs::{self, OpenOptions},
    io::{AsyncReadExt, AsyncWriteExt},
};

use crate::{setup_repo_dir, Hole, RepositoriesConfig};

const HOLES_PER_REPO: usize = 1_000;

pub(crate) async fn generate_holes(
    repositories_config: RepositoriesConfig,
    repos_dir_path: &Path,
    holes_dir_path: &Path,
) -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();
    for repo in repositories_config.repositories {
        let (_tmp_dir, path) = setup_repo_dir(repos_dir_path, &repo.source).await?;
        let mut files = vec![];

        let mut stack = VecDeque::new();
        stack.push_back(path.join(repo.source.src_path()));
        while let Some(src) = stack.pop_back() {
            let mut entries = fs::read_dir(&src).await?;
            while let Some(entry) = entries.next_entry().await? {
                let entry_type = entry.file_type().await?;

                let src_path = entry.path();

                if entry_type.is_dir() {
                    stack.push_back(src_path);
                } else if entry_type.is_file()
                    && repo
                        .language
                        .is_code_file(src_path.file_name().unwrap().to_str().unwrap())
                {
                    files.push(src_path);
                }
            }
        }

        let file_count = files.len();
        let mut holes = vec![];
        for file_path in files {
            let mut holes_per_file = HOLES_PER_REPO / file_count;
            let mut content = String::new();
            fs::File::open(&file_path)
                .await?
                .read_to_string(&mut content)
                .await?;
            let rope = Rope::from_str(&content);
            for _ in 0..holes_per_file {
                let line_nb = rng.gen_range(0..rope.len_lines());
                let line = rope.line(line_nb);
                let line_string = line.to_string();
                let trimmed = line_string.trim();
                if trimmed.starts_with(repo.language.comment_token()) || trimmed.is_empty() {
                    holes_per_file += 1;
                    continue;
                }
                let column_nb = rng.gen_range(0..15.min(line.len_chars()));
                holes.push(Hole::new(
                    line_nb as u32,
                    column_nb as u32,
                    file_path.strip_prefix(&path)?.to_str().unwrap().to_owned(),
                ));
            }
        }
        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&holes_dir_path.join(repo.holes_file))
            .await?;
        file.write_all(serde_json::to_string(&holes)?.as_bytes())
            .await?;
    }

    Ok(())
}
