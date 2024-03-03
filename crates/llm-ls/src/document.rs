use ropey::{Error as RopeyError, Rope, RopeSlice};
use tower_lsp::lsp_types::{Position, TextDocumentContentChangeEvent};
use tree_sitter::{InputEdit, Parser, Point, Tree};

use crate::error::{Error as LspError, Result};
use crate::language_id::LanguageId;

fn get_parser(language_id: LanguageId) -> Result<Parser> {
    match language_id {
        LanguageId::Bash => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_bash::language())?;
            Ok(parser)
        }
        LanguageId::C => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_c::language())?;
            Ok(parser)
        }
        LanguageId::Cpp => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_cpp::language())?;
            Ok(parser)
        }
        LanguageId::CSharp => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_c_sharp::language())?;
            Ok(parser)
        }
        LanguageId::Elixir => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_elixir::language())?;
            Ok(parser)
        }
        LanguageId::Erlang => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_erlang::language())?;
            Ok(parser)
        }
        LanguageId::Go => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_go::language())?;
            Ok(parser)
        }
        LanguageId::Html => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_html::language())?;
            Ok(parser)
        }
        LanguageId::Java => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_java::language())?;
            Ok(parser)
        }
        LanguageId::JavaScript | LanguageId::JavaScriptReact => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_javascript::language())?;
            Ok(parser)
        }
        LanguageId::Json => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_json::language())?;
            Ok(parser)
        }
        LanguageId::Kotlin => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_kotlin::language())?;
            Ok(parser)
        }
        LanguageId::Lua => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_lua::language())?;
            Ok(parser)
        }
        LanguageId::Markdown => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_md::language())?;
            Ok(parser)
        }
        LanguageId::ObjectiveC => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_objc::language())?;
            Ok(parser)
        }
        LanguageId::Python => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_python::language())?;
            Ok(parser)
        }
        LanguageId::R => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_r::language())?;
            Ok(parser)
        }
        LanguageId::Ruby => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_ruby::language())?;
            Ok(parser)
        }
        LanguageId::Rust => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_rust::language())?;
            Ok(parser)
        }
        LanguageId::Scala => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_scala::language())?;
            Ok(parser)
        }
        LanguageId::Swift => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_swift::language())?;
            Ok(parser)
        }
        LanguageId::TypeScript => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_typescript::language_typescript())?;
            Ok(parser)
        }
        LanguageId::TypeScriptReact => {
            let mut parser = Parser::new();
            parser.set_language(tree_sitter_typescript::language_tsx())?;
            Ok(parser)
        }
        LanguageId::Unknown => Ok(Parser::new()),
    }
}

#[derive(Clone, Debug, Copy)]
/// We redeclare this enum here because the `lsp_types` crate exports a Cow
/// type that is unconvenient to deal with.
pub enum PositionEncodingKind {
    #[allow(dead_code)]
    UTF8,
    UTF16,
    #[allow(dead_code)]
    UTF32,
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

    pub(crate) fn apply_content_change(
        &mut self,
        change: &TextDocumentContentChangeEvent,
        position_encoding: PositionEncodingKind,
    ) -> Result<()> {
        match change.range {
            Some(range) => {
                assert!(
                    range.start.line < range.end.line
                        || (range.start.line == range.end.line
                            && range.start.character <= range.end.character)
                );

                let same_line = range.start.line == range.end.line;
                let same_character = range.start.character == range.end.character;

                let change_start_line_cu_idx = range.start.character as usize;
                let change_end_line_cu_idx = range.end.character as usize;

                // 1. Get the line at which the change starts.
                let change_start_line_idx = range.start.line as usize;
                let change_start_line = match self.text.get_line(change_start_line_idx) {
                    Some(line) => line,
                    None => {
                        return Err(LspError::Rope(RopeyError::LineIndexOutOfBounds(
                            change_start_line_idx,
                            self.text.len_lines(),
                        )));
                    }
                };

                // 2. Get the line at which the change ends. (Small optimization
                // where we first check whether start and end line are the
                // same O(log N) lookup. We repeat this all throughout this
                // function).
                let change_end_line_idx = range.end.line as usize;
                let change_end_line = match same_line {
                    true => change_start_line,
                    false => match self.text.get_line(change_end_line_idx) {
                        Some(line) => line,
                        None => {
                            return Err(LspError::Rope(RopeyError::LineIndexOutOfBounds(
                                change_end_line_idx,
                                self.text.len_lines(),
                            )));
                        }
                    },
                };

                fn compute_char_idx(
                    position_encoding: PositionEncodingKind,
                    position: &Position,
                    slice: &RopeSlice,
                ) -> Result<usize> {
                    match position_encoding {
                        PositionEncodingKind::UTF8 => {
                            slice.try_byte_to_char(position.character as usize)
                        }
                        PositionEncodingKind::UTF16 => {
                            slice.try_utf16_cu_to_char(position.character as usize)
                        }
                        PositionEncodingKind::UTF32 => Ok(position.character as usize),
                    }
                    .map_err(|err| {
                        LspError::Rope(err)
                    })
                }

                // 3. Compute the character offset into the start/end line where
                // the change starts/ends.
                let change_start_line_char_idx =
                    compute_char_idx(position_encoding, &range.start, &change_start_line)?;
                let change_end_line_char_idx = match same_line && same_character {
                    true => change_start_line_char_idx,
                    false => compute_char_idx(position_encoding, &range.end, &change_end_line)?,
                };

                // 4. Compute the character and byte offset into the document
                // where the change starts/ends.
                let change_start_doc_char_idx =
                    self.text.line_to_char(change_start_line_idx) + change_start_line_char_idx;
                let change_end_doc_char_idx = match same_line && same_character {
                    true => change_start_doc_char_idx,
                    false => self.text.line_to_char(change_end_line_idx) + change_end_line_char_idx,
                };
                let change_start_doc_byte_idx = self.text.char_to_byte(change_start_doc_char_idx);
                let change_end_doc_byte_idx = match same_line && same_character {
                    true => change_start_doc_byte_idx,
                    false => self.text.char_to_byte(change_end_doc_char_idx),
                };

                // 5. Compute the byte offset into the start/end line where the
                // change starts/ends. Required for tree-sitter.
                let change_start_line_byte_idx = match position_encoding {
                    PositionEncodingKind::UTF8 => change_start_line_cu_idx,
                    PositionEncodingKind::UTF16 => {
                        change_start_line.char_to_utf16_cu(change_start_line_char_idx)
                    }
                    PositionEncodingKind::UTF32 => change_start_line_char_idx,
                };
                let change_end_line_byte_idx = match same_line && same_character {
                    true => change_start_line_byte_idx,
                    false => match position_encoding {
                        PositionEncodingKind::UTF8 => change_end_line_cu_idx,
                        PositionEncodingKind::UTF16 => {
                            change_end_line.char_to_utf16_cu(change_end_line_char_idx)
                        }
                        PositionEncodingKind::UTF32 => change_end_line_char_idx,
                    },
                };

                self.text
                    .remove(change_start_doc_char_idx..change_end_doc_char_idx);
                self.text.insert(change_start_doc_char_idx, &change.text);

                if let Some(tree) = &mut self.tree {
                    // 6. Compute the byte index into the new end line where the
                    // change ends. Required for tree-sitter.
                    let change_new_end_line_idx = self
                        .text
                        .byte_to_line(change_start_doc_byte_idx + change.text.len());
                    let change_new_end_line_byte_idx =
                        change_start_doc_byte_idx + change.text.len();

                    // 7. Construct the tree-sitter edit. We stay mindful that
                    // tree-sitter Point::column is a byte offset.
                    let edit = InputEdit {
                        start_byte: change_start_doc_byte_idx,
                        old_end_byte: change_end_doc_byte_idx,
                        new_end_byte: change_start_doc_byte_idx + change.text.len(),
                        start_position: Point {
                            row: change_start_line_idx,
                            column: change_start_line_byte_idx,
                        },
                        old_end_position: Point {
                            row: change_end_line_idx,
                            column: change_end_line_byte_idx,
                        },
                        new_end_position: Point {
                            row: change_new_end_line_idx,
                            column: change_new_end_line_byte_idx,
                        },
                    };

                    tree.edit(&edit);

                    self.tree = Some(self
                                .parser
                                .parse(self.text.to_string(), Some(tree))
                                .expect("parse should always return a tree when the language was set and no timeout was specified"));
                }

                Ok(())
            }
            None => {
                self.text = Rope::from_str(&change.text);
                self.tree = self.parser.parse(&change.text, None);

                Ok(())
            }
        }
    }
}
