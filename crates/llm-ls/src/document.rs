use ropey::Rope;
use tower_lsp::jsonrpc::Result;
use tree_sitter::{Parser, Tree};

use crate::internal_error;
use crate::language_id::LanguageId;

fn get_parser(language_id: LanguageId) -> Result<Parser> {
    match language_id {
        LanguageId::Bash => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_bash::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::C => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_c::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::Cpp => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_cpp::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::CSharp => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_c_sharp::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::Elixir => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_elixir::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::Erlang => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_erlang::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::Go => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_go::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::Html => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_html::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::Java => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_java::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::JavaScript | LanguageId::JavaScriptReact => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_javascript::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::Json => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_json::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::Lua => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_lua::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::Markdown => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_md::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::ObjectiveC => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_objc::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::Python => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_python::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::R => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_r::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::Ruby => Ok(Parser::new()),
        LanguageId::Rust => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_rust::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::Scala => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_scala::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::Swift => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_swift::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::TypeScript => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_typescript::language_typescript())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::TypeScriptReact => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_typescript::language_tsx())
                .map_err(internal_error)?;
            Ok(parser)
        }
        LanguageId::Unknown => Ok(Parser::new()),
    }
}

pub(crate) struct Document {
    #[allow(dead_code)]
    language_id: LanguageId,
    pub(crate) text: Rope,
    parser: Parser,
    pub(crate) tree: Option<Tree>,
}

impl Document {
    pub(crate) async fn open(language_id: &str, text: &str) -> Result<Self> {
        let language_id = language_id.into();
        let rope = Rope::from_str(text);
        let mut parser = get_parser(language_id)?;
        let tree = parser.parse(text, None);
        Ok(Document {
            language_id,
            text: rope,
            parser,
            tree,
        })
    }

    pub(crate) async fn change(&mut self, text: &str) -> Result<()> {
        let rope = Rope::from_str(text);
        self.tree = self.parser.parse(text, None);
        self.text = rope;
        Ok(())
    }
}
