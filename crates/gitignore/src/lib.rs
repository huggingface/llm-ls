use std::{
    fmt::Debug,
    fs::{canonicalize, File},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use glob::{MatchOptions, Pattern};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("non utf8 path")]
    NonUtf8Path,
    #[error("path has no parent folder")]
    NoParent,
    #[error("glob pattern error: {0}")]
    Pattern(#[from] glob::PatternError),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Rule {
    negate: bool,
    patterns: Vec<Pattern>,
    _source_line: usize,
}

impl Rule {
    pub fn parse(
        mut pattern: String,
        base_path: impl AsRef<Path>,
        _source_line: usize,
    ) -> Result<Option<Self>> {
        let mut patterns = vec![];
        if pattern.trim().is_empty() || pattern.starts_with('#') {
            return Ok(None);
        }
        pattern = pattern.trim_start().to_owned();
        let negate = if pattern.starts_with('!') {
            pattern.remove(0);
            true
        } else {
            false
        };
        let directory = if pattern.ends_with('/') {
            pattern.pop();
            true
        } else {
            false
        };
        let anchored = pattern.contains('/');
        if pattern.starts_with('/') {
            pattern.remove(0);
        }
        let base_path_str = base_path.as_ref().to_str().ok_or(Error::NonUtf8Path)?;
        let base_pattern = if anchored || pattern.starts_with("**") {
            format!("{base_path_str}/{pattern}")
        } else {
            format!("{base_path_str}/**/{pattern}")
        };
        patterns.push(Pattern::new(&format!("{base_pattern}/**"))?);
        if !directory {
            patterns.push(Pattern::new(&base_pattern)?);
        }
        Ok(Some(Self {
            negate,
            patterns,
            _source_line,
        }))
    }
}

#[derive(Debug)]
pub struct Gitignore {
    base_path: PathBuf,
    rules: Vec<Rule>,
    _source_file: PathBuf,
}

impl Gitignore {
    /// Parses a `.gitignore` file at `path`.
    ///
    /// If `path` is a directory, attempts to read `{dir}/.gitignore`.
    pub fn parse(path: impl AsRef<Path>) -> Result<Self> {
        let mut path = canonicalize(path)?;
        if path.is_dir() {
            path = path.join(".gitignore");
        }
        let reader = BufReader::new(File::open(&path)?);
        let mut rules = Vec::new();
        for (line_nb, line) in reader.lines().enumerate() {
            let line = line?;
            if let Some(rule) =
                Rule::parse(line, path.parent().ok_or(Error::NoParent)?, line_nb + 1)?
            {
                rules.push(rule);
            }
        }
        Ok(Self {
            base_path: path.parent().ok_or(Error::NoParent)?.to_path_buf(),
            rules,
            _source_file: path,
        })
    }

    pub fn ignored(&self, path: impl AsRef<Path>) -> Result<bool> {
        let path = if path.as_ref().starts_with(&self.base_path) {
            path.as_ref().to_path_buf()
        } else {
            canonicalize(self.base_path.join(path))?
        };
        let match_opts = MatchOptions {
            case_sensitive: true,
            require_literal_separator: true,
            require_literal_leading_dot: false,
        };
        let path_str = path.to_str().ok_or(Error::NonUtf8Path)?;
        let to_match = if path.is_dir() {
            format!("{path_str}/")
        } else {
            path_str.to_owned()
        };
        for rule in &self.rules {
            for pattern in rule.patterns.iter() {
                // TODO: handle negation properly
                // negation should include
                if rule.negate {
                    continue;
                }
                if pattern.matches_with(&to_match, match_opts) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Add ad hoc rule from a pattern
    pub fn add_rule(&mut self, pattern: String) -> Result<()> {
        if let Some(rule) = Rule::parse(pattern, &self.base_path, usize::MAX)? {
            self.rules.push(rule);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempdir::TempDir;

    use super::*;

    fn create_gitignore(rules: &str, name: &str) -> (TempDir, Gitignore) {
        let temp_dir = TempDir::new(name).unwrap();
        std::fs::File::create(temp_dir.path().join("LICENSE")).unwrap();
        std::fs::create_dir_all(temp_dir.path().join("config")).unwrap();
        std::fs::File::create(temp_dir.path().join("config.yaml")).unwrap();
        std::fs::File::create(temp_dir.path().join("Cargo.toml")).unwrap();
        std::fs::File::create(temp_dir.path().join("README.md")).unwrap();
        std::fs::create_dir_all(temp_dir.path().join("xtask")).unwrap();
        std::fs::create_dir_all(temp_dir.path().join("crates/gitignore")).unwrap();
        std::fs::File::create(temp_dir.path().join("crates/gitignore/Cargo.toml")).unwrap();
        std::fs::create_dir_all(temp_dir.path().join("crates/llm-ls/src")).unwrap();
        std::fs::create_dir_all(temp_dir.path().join("crates/llm-ls/config")).unwrap();
        std::fs::File::create(temp_dir.path().join("crates/llm-ls/config.yaml")).unwrap();
        std::fs::File::create(temp_dir.path().join("crates/llm-ls/Cargo.toml")).unwrap();
        std::fs::File::create(temp_dir.path().join("crates/llm-ls/src/main.rs")).unwrap();
        std::fs::create_dir_all(temp_dir.path().join("crates/lsp-client/src")).unwrap();
        std::fs::File::create(temp_dir.path().join("crates/lsp-client/Cargo.toml")).unwrap();
        std::fs::File::create(temp_dir.path().join("crates/lsp-client/src/lib.rs")).unwrap();
        std::fs::create_dir_all(temp_dir.path().join("crates/mock_server")).unwrap();
        std::fs::File::create(temp_dir.path().join("crates/mock_server/Cargo.toml")).unwrap();
        std::fs::create_dir_all(temp_dir.path().join("crates/testbed/src")).unwrap();
        std::fs::File::create(temp_dir.path().join("crates/testbed/Cargo.toml")).unwrap();
        std::fs::File::create(temp_dir.path().join("crates/testbed/src/main.rs")).unwrap();
        std::fs::create_dir_all(
            temp_dir
                .path()
                .join("crates/testbed/repositories/simple/src"),
        )
        .unwrap();
        std::fs::File::create(
            temp_dir
                .path()
                .join("crates/testbed/repositories/simple/src/main.rs"),
        )
        .unwrap();
        let gitignore_path = temp_dir.path().join(name);
        std::fs::File::create(&gitignore_path)
            .unwrap()
            .write_all(rules.as_bytes())
            .unwrap();
        let gitignore = Gitignore::parse(gitignore_path).unwrap();
        (temp_dir, gitignore)
    }

    #[test]
    fn test_regular_relative_pattern() {
        let (_temp_dir, gitignore) = create_gitignore("Cargo.toml", "regular_relative_pattern");
        assert!(gitignore.ignored("Cargo.toml").unwrap());
        assert!(!gitignore.ignored("LICENSE").unwrap());
    }

    #[test]
    fn test_glob_pattern() {
        let (_temp_dir, gitignore) = create_gitignore("crates/**/Cargo.toml", "glob_pattern");
        assert!(gitignore.ignored("crates/gitignore/Cargo.toml").unwrap());
        assert!(gitignore.ignored("crates/llm-ls/Cargo.toml").unwrap());
        assert!(gitignore.ignored("crates/lsp-client/Cargo.toml").unwrap());
        assert!(gitignore.ignored("crates/mock_server/Cargo.toml").unwrap());
        assert!(gitignore.ignored("crates/testbed/Cargo.toml").unwrap());
        assert!(!gitignore.ignored("crates/llm-ls/src/main.rs").unwrap());
        assert!(!gitignore.ignored("crates/lsp-client/src/lib.rs").unwrap());
        assert!(!gitignore.ignored("crates/testbed/src/main.rs").unwrap());
    }

    #[test]
    fn test_dir_start_glob_pattern() {
        let (_temp_dir, gitignore) = create_gitignore("**/crates/", "start_glob_pattern");
        assert!(gitignore.ignored("crates/").unwrap());
        assert!(gitignore.ignored("crates/llm-ls/Cargo.toml").unwrap());
        assert!(gitignore
            .ignored("crates/testbed/repositories/simple/src/main.rs")
            .unwrap());
        assert!(!gitignore.ignored("xtask/").unwrap());
        assert!(!gitignore.ignored("README.md").unwrap());
    }

    #[test]
    fn test_dir_relative_path() {
        let (_temp_dir, gitignore) = create_gitignore("crates/", "relative_path");
        assert!(gitignore.ignored("crates/").unwrap());
        assert!(gitignore.ignored("crates/llm-ls/Cargo.toml").unwrap());
        assert!(gitignore
            .ignored("crates/testbed/repositories/simple/src/main.rs")
            .unwrap());
        assert!(!gitignore.ignored("xtask/").unwrap());
        assert!(!gitignore.ignored("README.md").unwrap());
    }

    // TODO:
    // #[test]
    // fn test_negate_pattern() {
    //     let (_temp_dir, gitignore) = create_gitignore(
    //         "aaa/*\n\
    //         !aaa/Cargo.toml",
    //         "negate_pattern",
    //     );
    //     assert!(!gitignore.ignored("aaa/Cargo.toml").unwrap());
    //     assert!(gitignore.ignored("aaa/config.yaml").unwrap());
    // }

    #[test]
    fn test_ad_hoc_rule_add() {
        let (_temp_dir, mut gitignore) = create_gitignore("!Cargo.toml", "adhoc_add");
        assert!(!gitignore.ignored("config.yaml").unwrap());
        assert!(!gitignore.ignored("Cargo.toml").unwrap());
        gitignore.add_rule("config.yaml".to_owned()).unwrap();
        assert!(gitignore.ignored("config.yaml").unwrap());
    }

    #[test]
    fn test_anchored_file_or_dir() {
        let (_temp_dir, gitignore) = create_gitignore("/config*", "anchored_file_or_dir");
        assert!(gitignore.ignored("config.yaml").unwrap());
        assert!(gitignore.ignored("config").unwrap());
        assert!(!gitignore.ignored("crates/llm-ls/config.yaml").unwrap());
        assert!(!gitignore.ignored("crates/llm-ls/config").unwrap());
    }
}
