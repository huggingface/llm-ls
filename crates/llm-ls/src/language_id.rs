use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub(crate) enum LanguageId {
    Bash,
    C,
    Cpp,
    CSharp,
    Elixir,
    Erlang,
    Go,
    Html,
    Java,
    JavaScript,
    JavaScriptReact,
    Json,
    Kotlin,
    Lua,
    Markdown,
    ObjectiveC,
    Python,
    R,
    Ruby,
    Rust,
    Scala,
    Swift,
    TypeScript,
    TypeScriptReact,
    Unknown,
}

impl fmt::Display for LanguageId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bash => write!(f, "shellscript"),
            Self::C => write!(f, "c"),
            Self::Cpp => write!(f, "cpp"),
            Self::CSharp => write!(f, "csharp"),
            Self::Elixir => write!(f, "elixir"),
            Self::Erlang => write!(f, "erlang"),
            Self::Go => write!(f, "go"),
            Self::Html => write!(f, "html"),
            Self::Java => write!(f, "java"),
            Self::JavaScript => write!(f, "javascript"),
            Self::JavaScriptReact => write!(f, "javascriptreact"),
            Self::Json => write!(f, "json"),
            Self::Kotlin => write!(f, "kotlin"),
            Self::Lua => write!(f, "lua"),
            Self::Markdown => write!(f, "markdown"),
            Self::ObjectiveC => write!(f, "objective-c"),
            Self::Python => write!(f, "python"),
            Self::R => write!(f, "r"),
            Self::Ruby => write!(f, "ruby"),
            Self::Rust => write!(f, "rust"),
            Self::Scala => write!(f, "scala"),
            Self::Swift => write!(f, "swift"),
            Self::TypeScript => write!(f, "typescript"),
            Self::TypeScriptReact => write!(f, "typescriptreact"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

impl From<&str> for LanguageId {
    fn from(value: &str) -> Self {
        match value {
            "c" => Self::C,
            "cpp" => Self::Cpp,
            "csharp" => Self::CSharp,
            "elixir" => Self::Elixir,
            "erlang" => Self::Erlang,
            "go" => Self::Go,
            "html" => Self::Html,
            "java" => Self::Java,
            "javascript" => Self::JavaScript,
            "javascriptreact" => Self::JavaScriptReact,
            "json" => Self::Json,
            "kotlin" => Self::Kotlin,
            "lua" => Self::Lua,
            "markdown" => Self::Markdown,
            "objective-c" => Self::ObjectiveC,
            "python" => Self::Python,
            "r" => Self::R,
            "ruby" => Self::Ruby,
            "rust" => Self::Rust,
            "scala" => Self::Scala,
            "shellscript" => Self::Bash,
            "swift" => Self::Swift,
            "typescript" => Self::TypeScript,
            "typescriptreact" => Self::TypeScriptReact,
            _ => Self::Unknown,
        }
    }
}

impl From<String> for LanguageId {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}
