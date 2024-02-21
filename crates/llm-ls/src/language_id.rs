use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub(crate) enum LanguageId {
    Abap,
    Bat,
    BibTeX,
    Bash,
    Clojure,
    CoffeeScript,
    C,
    Cpp,
    CSharp,
    Css,
    Diff,
    Dart,
    Dockerfile,
    Elixir,
    Erlang,
    FSharp,
    GitCommit,
    GitRebase,
    Go,
    Groovy,
    Handlebars,
    Html,
    Ini,
    Java,
    JavaScript,
    JavaScriptReact,
    Json,
    Kotlin,
    LaTeX,
    Less,
    Lua,
    Makefile,
    Markdown,
    ObjectiveC,
    ObjectiveCpp,
    Perl,
    Perl6,
    Php,
    Powershell,
    Pug,
    Python,
    R,
    Razor,
    Ruby,
    Rust,
    Scss,
    Scala,
    ShaderLab,
    Sql,
    Swift,
    Toml,
    TypeScript,
    TypeScriptReact,
    TeX,
    VisualBasic,
    Xml,
    Xsl,
    Yaml,
    Unknown,
}

impl fmt::Display for LanguageId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Abap => write!(f, "abap"),
            Self::Bash => write!(f, "shellscript"),
            Self::Bat => write!(f, "bat"),
            Self::BibTeX => write!(f, "bibtex"),
            Self::Clojure => write!(f, "clojure"),
            Self::CoffeeScript => write!(f, "coffeescript"),
            Self::C => write!(f, "c"),
            Self::Cpp => write!(f, "cpp"),
            Self::CSharp => write!(f, "csharp"),
            Self::Css => write!(f, "css"),
            Self::Diff => write!(f, "diff"),
            Self::Dart => write!(f, "dart"),
            Self::Dockerfile => write!(f, "dockerfile"),
            Self::Elixir => write!(f, "elixir"),
            Self::Erlang => write!(f, "erlang"),
            Self::FSharp => write!(f, "fsharp"),
            Self::GitCommit => write!(f, "git-commit"),
            Self::GitRebase => write!(f, "git-rebase"),
            Self::Go => write!(f, "go"),
            Self::Groovy => write!(f, "groovy"),
            Self::Handlebars => write!(f, "handlebars"),
            Self::Html => write!(f, "html"),
            Self::Ini => write!(f, "ini"),
            Self::Java => write!(f, "java"),
            Self::JavaScript => write!(f, "javascript"),
            Self::JavaScriptReact => write!(f, "javascriptreact"),
            Self::Json => write!(f, "json"),
            Self::Kotlin => write!(f, "kotlin"),
            Self::LaTeX => write!(f, "latex"),
            Self::Less => write!(f, "less"),
            Self::Lua => write!(f, "lua"),
            Self::Makefile => write!(f, "makefile"),
            Self::Markdown => write!(f, "markdown"),
            Self::ObjectiveC => write!(f, "objective-c"),
            Self::ObjectiveCpp => write!(f, "objective-cpp"),
            Self::Perl => write!(f, "perl"),
            Self::Perl6 => write!(f, "perl6"),
            Self::Php => write!(f, "php"),
            Self::Powershell => write!(f, "powershell"),
            Self::Pug => write!(f, "jade"),
            Self::Python => write!(f, "python"),
            Self::R => write!(f, "r"),
            Self::Razor => write!(f, "razor"),
            Self::Ruby => write!(f, "ruby"),
            Self::Rust => write!(f, "rust"),
            Self::ShaderLab => write!(f, "shaderlab"),
            Self::Scss => write!(f, "scss"),
            Self::Scala => write!(f, "scala"),
            Self::Sql => write!(f, "sql"),
            Self::Swift => write!(f, "swift"),
            Self::Toml => write!(f, "toml"),
            Self::TypeScript => write!(f, "typescript"),
            Self::TypeScriptReact => write!(f, "typescriptreact"),
            Self::TeX => write!(f, "tex"),
            Self::VisualBasic => write!(f, "vb"),
            Self::Xml => write!(f, "xml"),
            Self::Xsl => write!(f, "xsl"),
            Self::Yaml => write!(f, "Yaml"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

pub(crate) struct LanguageIdError {
    language_id: String,
}

impl fmt::Display for LanguageIdError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Invalid language id: {}", self.language_id)
    }
}

impl From<&str> for LanguageId {
    fn from(value: &str) -> Self {
        match value {
            "abap" => Self::Abap,
            "bat" => Self::Bat,
            "bibtex" => Self::BibTeX,
            "clojure" => Self::Clojure,
            "coffeescript" => Self::CoffeeScript,
            "c" => Self::C,
            "cpp" => Self::Cpp,
            "csharp" => Self::CSharp,
            "css" => Self::Css,
            "diff" => Self::Diff,
            "dart" => Self::Dart,
            "dockerfile" => Self::Dockerfile,
            "elixir" => Self::Elixir,
            "erlang" => Self::Erlang,
            "fsharp" => Self::FSharp,
            "git-commit" => Self::GitCommit,
            "git-rebase" => Self::GitRebase,
            "go" => Self::Go,
            "groovy" => Self::Groovy,
            "handlebars" => Self::Handlebars,
            "html" => Self::Html,
            "ini" => Self::Ini,
            "jade" => Self::Pug,
            "java" => Self::Java,
            "javascript" => Self::JavaScript,
            "javascriptreact" => Self::JavaScriptReact,
            "json" => Self::Json,
            "kotlin" => Self::Kotlin,
            "latex" => Self::LaTeX,
            "less" => Self::Less,
            "lua" => Self::Lua,
            "makefile" => Self::Makefile,
            "markdown" => Self::Markdown,
            "objective-c" => Self::ObjectiveC,
            "objective-cpp" => Self::ObjectiveCpp,
            "perl" => Self::Perl,
            "perl6" => Self::Perl6,
            "php" => Self::Php,
            "powershell" => Self::Powershell,
            "python" => Self::Python,
            "r" => Self::R,
            "razor" => Self::Razor,
            "ruby" => Self::Ruby,
            "rust" => Self::Rust,
            "sass" | "scss" => Self::Scss,
            "scala" => Self::Scala,
            "shaderlab" => Self::ShaderLab,
            "shellscript" => Self::Bash,
            "sql" => Self::Sql,
            "swift" => Self::Swift,
            "typescript" => Self::TypeScript,
            "typescriptreact" => Self::TypeScriptReact,
            "tex" => Self::TeX,
            "vb" => Self::VisualBasic,
            "xml" => Self::Xml,
            "xsl" => Self::Xsl,
            "yaml" => Self::Yaml,
            _ => Self::Unknown,
        }
    }
}

impl From<String> for LanguageId {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LanguageComment {
    open: String,
    close: Option<String>,
}

impl LanguageComment {
    pub(crate) fn comment_string(&self, s: String) -> String {
        let close = if let Some(close) = self.close.as_ref() {
            close.clone()
        } else {
            String::new()
        };
        format!("{} {s} {close}", self.open)
    }
}

impl LanguageId {
    pub(crate) fn get_language_comment(&self) -> LanguageComment {
        match self {
            Self::Abap => LanguageComment {
                open: "*".to_owned(),
                close: None,
            },
            Self::Bash => LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
            Self::Bat => LanguageComment {
                open: "REM".to_owned(),
                close: None,
            },
            Self::BibTeX => LanguageComment {
                open: "%".to_owned(),
                close: None,
            },
            Self::Clojure => LanguageComment {
                open: ";;".to_owned(),
                close: None,
            },
            Self::CoffeeScript => LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
            Self::C => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::Cpp => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::CSharp => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::Css => LanguageComment {
                open: "/*".to_owned(),
                close: Some("*/".to_owned()),
            },
            Self::Diff => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::Dart => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::Dockerfile => LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
            Self::Elixir => LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
            Self::Erlang => LanguageComment {
                open: "%".to_owned(),
                close: None,
            },
            Self::FSharp => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::GitCommit => LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
            Self::GitRebase => LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
            Self::Go => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::Groovy => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::Handlebars => LanguageComment {
                open: "{{!--".to_owned(),
                close: Some("--}}".to_owned()),
            },
            Self::Html => LanguageComment {
                open: "<!--".to_owned(),
                close: Some("-->".to_owned()),
            },
            Self::Ini => LanguageComment {
                open: ";".to_owned(),
                close: None,
            },
            Self::Java => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::JavaScript => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::JavaScriptReact => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::Json => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::Kotlin => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::LaTeX => LanguageComment {
                open: "%".to_owned(),
                close: None,
            },
            Self::Less => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::Lua => LanguageComment {
                open: "--".to_owned(),
                close: None,
            },
            Self::Makefile => LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
            Self::Markdown => LanguageComment {
                open: "<!--".to_owned(),
                close: Some("-->".to_owned()),
            },
            Self::ObjectiveC => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::ObjectiveCpp => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::Perl => LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
            Self::Perl6 => LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
            Self::Php => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::Powershell => LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
            Self::Pug => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::Python => LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
            Self::R => LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
            Self::Razor => LanguageComment {
                open: "@*".to_owned(),
                close: Some("*@".to_owned()),
            },
            Self::Ruby => LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
            Self::Rust => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::Scss => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::Scala => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::ShaderLab => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::Sql => LanguageComment {
                open: "--".to_owned(),
                close: None,
            },
            Self::Swift => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::Toml => LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
            Self::TypeScript => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::TypeScriptReact => LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
            Self::TeX => LanguageComment {
                open: "%".to_owned(),
                close: None,
            },
            Self::VisualBasic => LanguageComment {
                open: "'".to_owned(),
                close: None,
            },
            Self::Xml => LanguageComment {
                open: "<!--".to_owned(),
                close: Some("-->".to_owned()),
            },
            Self::Xsl => LanguageComment {
                open: "<!--".to_owned(),
                close: Some("-->".to_owned()),
            },
            Self::Yaml => LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
            Self::Unknown => LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
        }
    }
}
