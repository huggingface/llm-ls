use ropey::Rope;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::Range;
use tree_sitter::{InputEdit, Parser, Point, Tree};

use crate::language_id::LanguageId;
use crate::{get_position_idx, internal_error};

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
        LanguageId::Kotlin => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_kotlin::language())
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
        LanguageId::Ruby => {
            let mut parser = Parser::new();
            parser
                .set_language(tree_sitter_ruby::language())
                .map_err(internal_error)?;
            Ok(parser)
        }
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
    pub(crate) language_id: LanguageId,
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

    pub(crate) async fn change(&mut self, range: Range, text: &str) -> Result<()> {
        let start_idx = get_position_idx(
            &self.text,
            range.start.line as usize,
            range.start.character as usize,
        )?;
        let start_byte = self
            .text
            .try_char_to_byte(start_idx)
            .map_err(internal_error)?;
        let old_end_idx = get_position_idx(
            &self.text,
            range.end.line as usize,
            range.end.character as usize,
        )?;
        let old_end_byte = self
            .text
            .try_char_to_byte(old_end_idx)
            .map_err(internal_error)?;
        let start_position = Point {
            row: range.start.line as usize,
            column: range.start.character as usize,
        };
        let old_end_position = Point {
            row: range.end.line as usize,
            column: range.end.character as usize,
        };
        let (new_end_idx, new_end_position) = if range.start == range.end {
            let row = range.start.line as usize;
            let column = range.start.character as usize;
            let idx = self.text.try_line_to_char(row).map_err(internal_error)? + column;
            let rope = Rope::from_str(text);
            let text_len = rope.len_chars();
            let end_idx = idx + text_len;
            self.text.insert(idx, text);
            (
                end_idx,
                Point {
                    row,
                    column: column + text_len,
                },
            )
        } else {
            let removal_idx = self
                .text
                .try_line_to_char(range.end.line as usize)
                .map_err(internal_error)?
                + (range.end.character as usize);
            let slice_size = removal_idx - start_idx;
            self.text
                .try_remove(start_idx..removal_idx)
                .map_err(internal_error)?;
            self.text.insert(start_idx, text);
            let rope = Rope::from_str(text);
            let text_len = rope.len_chars();
            let character_difference = text_len as isize - slice_size as isize;
            let new_end_idx = if character_difference.is_negative() {
                removal_idx - character_difference.wrapping_abs() as usize
            } else {
                removal_idx + character_difference as usize
            };
            let row = self
                .text
                .try_char_to_line(new_end_idx)
                .map_err(internal_error)?;
            let line_start = self.text.try_line_to_char(row).map_err(internal_error)?;
            let column = new_end_idx - line_start;
            (new_end_idx, Point { row, column })
        };
        if let Some(tree) = self.tree.as_mut() {
            let edit = InputEdit {
                start_byte,
                old_end_byte,
                new_end_byte: self
                    .text
                    .try_char_to_byte(new_end_idx)
                    .map_err(internal_error)?,
                start_position,
                old_end_position,
                new_end_position,
            };
            tree.edit(&edit);
        }
        self.tree = self.parser.parse(self.text.to_string(), self.tree.as_ref());
        Ok(())
    }
}
