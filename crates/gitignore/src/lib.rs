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
    #[error("glob pattern error: {0}")]
    Pattern(#[from] glob::PatternError),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Rule {
    negate: bool,
    pattern: Pattern,
    _source_line: usize,
}

impl Rule {
    pub fn parse(
        mut pattern: String,
        base_path: impl AsRef<Path>,
        _source_line: usize,
    ) -> Result<Option<Self>> {
        if pattern.trim().is_empty() || pattern.starts_with('#') {
            return Ok(None);
        }
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
        let pattern = if anchored {
            let base = format!("{}/{pattern}", base_path.as_ref().to_str().unwrap());
            if directory {
                format!("{base}/**")
            } else {
                base
            }
        } else if !pattern.starts_with("**") {
            let base = format!("**/{pattern}");
            if directory {
                format!("{base}/**")
            } else {
                base
            }
        } else {
            pattern
        };
        Ok(Some(Self {
            negate,
            pattern: Pattern::new(&pattern)?,
            _source_line,
        }))
    }
}

#[derive(Debug)]
pub struct Gitignore {
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
            if let Some(rule) = Rule::parse(line, path.parent().unwrap(), line_nb + 1)? {
                rules.push(rule);
            }
        }
        Ok(Self {
            rules,
            _source_file: path,
        })
    }

    pub fn ignored(&self, path: impl AsRef<Path>) -> Result<bool> {
        let path = canonicalize(path)?;
        let match_opts = MatchOptions {
            case_sensitive: true,
            require_literal_separator: true,
            require_literal_leading_dot: false,
        };
        for rule in &self.rules {
            let path_str = path.to_str().ok_or(Error::NonUtf8Path)?;
            let to_match = if path.is_dir() {
                format!("{path_str}/")
            } else {
                path_str.to_owned()
            };
            if rule.pattern.matches_with(&to_match, match_opts) {
                return Ok(!rule.negate);
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Once;

    use super::*;

    static INIT: Once = Once::new();

    fn create_gitignore(rules: &str, name: &str) -> Gitignore {
        INIT.call_once(|| {
            std::env::set_current_dir(canonicalize("../..").unwrap()).unwrap();
        });
        std::fs::write(name, rules).unwrap();
        let gitignore = Gitignore::parse(name).unwrap();
        std::fs::remove_file(name).unwrap();
        gitignore
    }

    #[test]
    fn test_regular_pattern() {
        let gitignore = create_gitignore("Cargo.toml", "regular_pattern");
        assert!(gitignore.ignored("Cargo.toml").unwrap());
        assert!(!gitignore.ignored("LICENSE").unwrap());
    }

    #[test]
    fn test_glob_pattern() {
        let gitignore = create_gitignore("crates/**/Cargo.toml", "glob_pattern");
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
    fn test_negate_glob_pattern() {
        let gitignore = create_gitignore("!crates/**/Cargo.toml", "negate_glob_pattern");
        assert!(!gitignore.ignored("crates/gitignore/Cargo.toml").unwrap());
        assert!(!gitignore.ignored("crates/llm-ls/Cargo.toml").unwrap());
        assert!(!gitignore.ignored("crates/lsp-client/Cargo.toml").unwrap());
        assert!(!gitignore.ignored("crates/mock_server/Cargo.toml").unwrap());
        assert!(!gitignore.ignored("crates/testbed/Cargo.toml").unwrap());
        assert!(!gitignore.ignored("crates/llm-ls/src/main.rs").unwrap());
        assert!(!gitignore.ignored("crates/lsp-client/src/lib.rs").unwrap());
        assert!(!gitignore.ignored("crates/testbed/src/main.rs").unwrap());
    }

    #[test]
    fn test_start_glob_pattern() {
        let gitignore = create_gitignore("**/crates/", "start_glob_pattern");
        assert!(gitignore.ignored("crates/").unwrap());
        assert!(gitignore.ignored("crates/llm-ls/Cargo.toml").unwrap());
        assert!(gitignore
            .ignored("crates/testbed/repositories/simple/src/main.rs")
            .unwrap());
        assert!(!gitignore.ignored("xtask/").unwrap());
        assert!(!gitignore.ignored("README.md").unwrap());
    }

    #[test]
    fn test_relative_path() {
        let gitignore = create_gitignore("crates/", "relative_path");
        assert!(gitignore.ignored("crates/").unwrap());
        assert!(gitignore.ignored("crates/llm-ls/Cargo.toml").unwrap());
        assert!(gitignore
            .ignored("crates/testbed/repositories/simple/src/main.rs")
            .unwrap());
        assert!(!gitignore.ignored("xtask/").unwrap());
        assert!(!gitignore.ignored("README.md").unwrap());
    }

    #[test]
    fn test_negate_pattern() {
        let gitignore = create_gitignore(
            "!Cargo.toml\n\
            Cargo.toml",
            "negate_pattern",
        );
        assert!(!gitignore.ignored("Cargo.toml").unwrap());
    }
}
