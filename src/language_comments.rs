use crate::LanguageComment;
use std::collections::HashMap;

pub fn build_language_comments() -> HashMap<String, LanguageComment> {
    HashMap::from([
        (
            "abap".to_owned(),
            LanguageComment {
                open: "*".to_owned(),
                close: None,
            },
        ),
        (
            "bat".to_owned(),
            LanguageComment {
                open: "REM".to_owned(),
                close: None,
            },
        ),
        (
            "bibtex".to_owned(),
            LanguageComment {
                open: "%".to_owned(),
                close: None,
            },
        ),
        (
            "clojure".to_owned(),
            LanguageComment {
                open: ";;".to_owned(),
                close: None,
            },
        ),
        (
            "coffeescript".to_owned(),
            LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
        ),
        (
            "c".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "cpp".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "csharp".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "css".to_owned(),
            LanguageComment {
                open: "/*".to_owned(),
                close: Some("*/".to_owned()),
            },
        ),
        (
            "diff".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "dart".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "dockerfile".to_owned(),
            LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
        ),
        (
            "elixir".to_owned(),
            LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
        ),
        (
            "erlang".to_owned(),
            LanguageComment {
                open: "%".to_owned(),
                close: None,
            },
        ),
        (
            "fsharp".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "git-commit".to_owned(),
            LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
        ),
        (
            "git-rebase".to_owned(),
            LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
        ),
        (
            "go".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "groovy".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "handlebars".to_owned(),
            LanguageComment {
                open: "{{!--".to_owned(),
                close: Some("--}}".to_owned()),
            },
        ),
        (
            "html".to_owned(),
            LanguageComment {
                open: "<!--".to_owned(),
                close: Some("-->".to_owned()),
            },
        ),
        (
            "ini".to_owned(),
            LanguageComment {
                open: ";".to_owned(),
                close: None,
            },
        ),
        (
            "java".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "javascript".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "javascriptreact".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "json".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "latex".to_owned(),
            LanguageComment {
                open: "%".to_owned(),
                close: None,
            },
        ),
        (
            "less".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "lua".to_owned(),
            LanguageComment {
                open: "--".to_owned(),
                close: None,
            },
        ),
        (
            "makefile".to_owned(),
            LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
        ),
        (
            "markdown".to_owned(),
            LanguageComment {
                open: "<!--".to_owned(),
                close: Some("-->".to_owned()),
            },
        ),
        (
            "objective-c".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "objective-cpp".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "perl".to_owned(),
            LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
        ),
        (
            "perl6".to_owned(),
            LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
        ),
        (
            "php".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "powershell".to_owned(),
            LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
        ),
        (
            "jade".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "python".to_owned(),
            LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
        ),
        (
            "r".to_owned(),
            LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
        ),
        (
            "razor".to_owned(),
            LanguageComment {
                open: "@*".to_owned(),
                close: Some("*@".to_owned()),
            },
        ),
        (
            "ruby".to_owned(),
            LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
        ),
        (
            "rust".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "scss".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "sass".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "scala".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "shaderlab".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "shellscript".to_owned(),
            LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
        ),
        (
            "sql".to_owned(),
            LanguageComment {
                open: "--".to_owned(),
                close: None,
            },
        ),
        (
            "swift".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "toml".to_owned(),
            LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
        ),
        (
            "typescript".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "typescriptreact".to_owned(),
            LanguageComment {
                open: "//".to_owned(),
                close: None,
            },
        ),
        (
            "tex".to_owned(),
            LanguageComment {
                open: "%".to_owned(),
                close: None,
            },
        ),
        (
            "vb".to_owned(),
            LanguageComment {
                open: "'".to_owned(),
                close: None,
            },
        ),
        (
            "xml".to_owned(),
            LanguageComment {
                open: "<!--".to_owned(),
                close: Some("-->".to_owned()),
            },
        ),
        (
            "xsl".to_owned(),
            LanguageComment {
                open: "<!--".to_owned(),
                close: Some("-->".to_owned()),
            },
        ),
        (
            "yaml".to_owned(),
            LanguageComment {
                open: "#".to_owned(),
                close: None,
            },
        ),
    ])
}
