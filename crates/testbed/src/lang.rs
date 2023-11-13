use std::fmt;

use serde::{Deserialize, Serialize};

// const JS_EXT: [&str; 2] = [".js", ".jsx"];
const PY_EXT: [&str; 1] = [".py"];
const RS_EXT: [&str; 1] = [".rs"];
const TS_EXT: [&str; 3] = [".ts", ".tsx", ".d.ts"];

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Language {
    Python,
    Rust,
    Typescript,
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Python => write!(f, "python"),
            Self::Rust => write!(f, "rust"),
            Self::Typescript => write!(f, "typescript"),
        }
    }
}

impl Language {
    pub(crate) fn is_code_file(&self, file_name: &str) -> bool {
        match self {
            Self::Python => PY_EXT.iter().any(|ext| file_name.ends_with(ext)),
            Self::Rust => RS_EXT.iter().any(|ext| file_name.ends_with(ext)),
            Self::Typescript => TS_EXT.iter().any(|ext| file_name.ends_with(ext)),
        }
    }

    pub(crate) fn comment_token(&self) -> &str {
        match self {
            Self::Python => "#",
            Self::Rust => "//",
            Self::Typescript => "//",
        }
    }
}
