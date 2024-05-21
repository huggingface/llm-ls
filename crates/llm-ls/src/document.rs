use ropey::{Rope, RopeSlice};
use tower_lsp::lsp_types::{Position, TextDocumentContentChangeEvent};
use tree_sitter::{InputEdit, Parser, Point, Tree};

use crate::error::{Error, Result};
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
    Utf8,
    Utf16,
    Utf32,
}

impl TryFrom<tower_lsp::lsp_types::PositionEncodingKind> for PositionEncodingKind {
    type Error = Error;

    fn try_from(value: tower_lsp::lsp_types::PositionEncodingKind) -> Result<Self> {
        if value == tower_lsp::lsp_types::PositionEncodingKind::UTF8 {
            Ok(PositionEncodingKind::Utf8)
        } else if value == tower_lsp::lsp_types::PositionEncodingKind::UTF16 {
            Ok(PositionEncodingKind::Utf16)
        } else if value == tower_lsp::lsp_types::PositionEncodingKind::UTF32 {
            Ok(PositionEncodingKind::Utf32)
        } else {
            Err(Error::UnknownEncodingKind(value.as_str().to_owned()))
        }
    }
}

impl TryFrom<Vec<tower_lsp::lsp_types::PositionEncodingKind>> for PositionEncodingKind {
    type Error = Error;

    fn try_from(value: Vec<tower_lsp::lsp_types::PositionEncodingKind>) -> Result<Self> {
        if value.contains(&tower_lsp::lsp_types::PositionEncodingKind::UTF8) {
            Ok(PositionEncodingKind::Utf8)
        } else if value.contains(&tower_lsp::lsp_types::PositionEncodingKind::UTF16) {
            Ok(PositionEncodingKind::Utf16)
        } else if value.contains(&tower_lsp::lsp_types::PositionEncodingKind::UTF32) {
            Ok(PositionEncodingKind::Utf32)
        } else {
            Err(Error::EncodingKindMissing)
        }
    }
}

impl PositionEncodingKind {
    pub fn to_lsp_type(self) -> tower_lsp::lsp_types::PositionEncodingKind {
        match self {
            PositionEncodingKind::Utf8 => tower_lsp::lsp_types::PositionEncodingKind::UTF8,
            PositionEncodingKind::Utf16 => tower_lsp::lsp_types::PositionEncodingKind::UTF16,
            PositionEncodingKind::Utf32 => tower_lsp::lsp_types::PositionEncodingKind::UTF32,
        }
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

    pub(crate) fn apply_content_change(
        &mut self,
        change: &TextDocumentContentChangeEvent,
        position_encoding: PositionEncodingKind,
    ) -> Result<()> {
        match change.range {
            Some(range) => {
                if range.start.line > range.end.line
                    || (range.start.line == range.end.line
                        && range.start.character > range.end.character)
                {
                    return Err(Error::InvalidRange(range));
                }

                let same_line = range.start.line == range.end.line;
                let same_character = range.start.character == range.end.character;

                let change_start_line_cu_idx = range.start.character as usize;
                let change_end_line_cu_idx = range.end.character as usize;

                // 1. Get the line at which the change starts.
                let change_start_line_idx = range.start.line as usize;
                let change_start_line =
                    self.text.get_line(change_start_line_idx).ok_or_else(|| {
                        Error::OutOfBoundLine(change_start_line_idx, self.text.len_lines())
                    })?;

                // 2. Get the line at which the change ends. (Small optimization
                // where we first check whether start and end line are the
                // same O(log N) lookup. We repeat this all throughout this
                // function).
                let change_end_line_idx = range.end.line as usize;
                let change_end_line = match same_line {
                    true => change_start_line,
                    false => self.text.get_line(change_end_line_idx).ok_or_else(|| {
                        Error::OutOfBoundLine(change_end_line_idx, self.text.len_lines())
                    })?,
                };

                fn compute_char_idx(
                    position_encoding: PositionEncodingKind,
                    position: &Position,
                    slice: &RopeSlice,
                ) -> Result<usize> {
                    Ok(match position_encoding {
                        PositionEncodingKind::Utf8 => {
                            slice.try_byte_to_char(position.character as usize)?
                        }
                        PositionEncodingKind::Utf16 => {
                            slice.try_utf16_cu_to_char(position.character as usize)?
                        }
                        PositionEncodingKind::Utf32 => position.character as usize,
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
                    PositionEncodingKind::Utf8 => change_start_line_cu_idx,
                    PositionEncodingKind::Utf16 => {
                        change_start_line.char_to_utf16_cu(change_start_line_char_idx)
                    }
                    PositionEncodingKind::Utf32 => change_start_line_char_idx,
                };
                let change_end_line_byte_idx = match same_line && same_character {
                    true => change_start_line_byte_idx,
                    false => match position_encoding {
                        PositionEncodingKind::Utf8 => change_end_line_cu_idx,
                        PositionEncodingKind::Utf16 => {
                            change_end_line.char_to_utf16_cu(change_end_line_char_idx)
                        }
                        PositionEncodingKind::Utf32 => change_end_line_char_idx,
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

                    match self.parser.parse(self.text.to_string(), Some(tree)) {
                        Some(new_tree) => {
                            self.tree = Some(new_tree);
                        }
                        None => {
                            return Err(Error::TreeSitterParsing);
                        }
                    }
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

#[cfg(test)]
mod test {
    use tower_lsp::lsp_types::{Position, Range};
    use tree_sitter::Node;

    use super::*;

    macro_rules! new_change {
        ($start_line:expr, $start_char:expr, $end_line:expr, $end_char:expr, $text:expr) => {
            &TextDocumentContentChangeEvent {
                range: Some(Range::new(
                    Position::new($start_line as u32, $start_char as u32),
                    Position::new($end_line as u32, $end_char as u32),
                )),
                range_length: None,
                text: $text.to_string(),
            }
        };
    }

    #[tokio::test]
    async fn test_text_document_apply_content_change() {
        let mut rope = Rope::from_str("🤗 Hello 🤗\nABC 🇫🇷\n world!");
        let mut doc = Document::open("unknown", &rope.to_string()).await.unwrap();

        doc.apply_content_change(new_change!(0, 0, 0, 3, ""), PositionEncodingKind::Utf16)
            .unwrap();
        rope = Rope::from_str("Hello 🤗\nABC 🇫🇷\n world!");
        assert_eq!(doc.text.to_string(), rope.to_string());

        doc.apply_content_change(
            new_change!(1, 4 + "🇫🇷".len(), 1, 4 + "🇫🇷".len(), " DEF"),
            PositionEncodingKind::Utf8,
        )
        .unwrap();
        rope = Rope::from_str("Hello 🤗\nABC 🇫🇷 DEF\n world!");
        assert_eq!(doc.text.to_string(), rope.to_string());

        doc.apply_content_change(
            new_change!(1, 0, 1, 4 + "🇫🇷".chars().count() + 4, ""),
            PositionEncodingKind::Utf32,
        )
        .unwrap();
        rope = Rope::from_str("Hello 🤗\n\n world!");
        assert_eq!(doc.text.to_string(), rope.to_string());

        doc.apply_content_change(new_change!(1, 0, 1, 1, ""), PositionEncodingKind::Utf16)
            .unwrap();
        rope = Rope::from_str("Hello 🤗\n world!");
        assert_eq!(doc.text.to_string(), rope.to_string());

        doc.apply_content_change(new_change!(0, 5, 1, 1, "，"), PositionEncodingKind::Utf16)
            .unwrap();
        rope = Rope::from_str("Hello，world!");
        assert_eq!(doc.text.to_string(), rope.to_string());

        doc.apply_content_change(
            new_change!(0, 0, 0, rope.len_utf16_cu(), ""),
            PositionEncodingKind::Utf16,
        )
        .unwrap();
        assert_eq!(doc.text.to_string(), "");
    }

    #[tokio::test]
    async fn test_text_document_apply_content_change_no_range() {
        let mut rope = Rope::from_str(
            "let a = '🥸 你好';\rfunction helloWorld() { return '🤲🏿'; }\nlet b = 'Hi, 😊';",
        );
        let mut doc = Document::open(&LanguageId::JavaScript.to_string(), &rope.to_string())
            .await
            .unwrap();
        let mut parser = Parser::new();

        parser
            .set_language(tree_sitter_javascript::language())
            .unwrap();

        assert!(doc.apply_content_change(
            &TextDocumentContentChangeEvent {
                range: None,
                range_length: None,
                text: "let a = '🥸 你好';\rfunction helloWorld() { return '🤲🏿'; }\nlet b = 'Hi, 😊';".to_owned(),
            },
            PositionEncodingKind::Utf16,
        ).is_ok());
        assert_eq!(doc.text.to_string(), rope.to_string());

        let tree = parser.parse(&rope.to_string(), None).unwrap();

        assert!(nodes_are_equal_recursive(
            &doc.tree.as_ref().unwrap().root_node(),
            &tree.root_node()
        ));

        assert!(doc
            .apply_content_change(
                &TextDocumentContentChangeEvent {
                    range: None,
                    range_length: None,
                    text: "let a = '🥸 你好，😊';".to_owned(),
                },
                PositionEncodingKind::Utf16,
            )
            .is_ok());
        rope = Rope::from_str("let a = '🥸 你好，😊';");
        assert_eq!(doc.text.to_string(), rope.to_string());

        let tree = parser.parse(&rope.to_string(), None).unwrap();

        assert!(nodes_are_equal_recursive(
            &doc.tree.as_ref().unwrap().root_node(),
            &tree.root_node()
        ));
    }

    #[tokio::test]
    async fn test_text_document_apply_content_change_bounds() {
        let rope = Rope::from_str("");
        let mut doc = Document::open(&LanguageId::Unknown.to_string(), &rope.to_string())
            .await
            .unwrap();

        assert!(doc
            .apply_content_change(new_change!(0, 0, 0, 1, ""), PositionEncodingKind::Utf16)
            .is_err());

        assert!(doc
            .apply_content_change(new_change!(1, 0, 1, 0, ""), PositionEncodingKind::Utf16)
            .is_err());

        assert!(doc
            .apply_content_change(new_change!(0, 0, 0, 0, "🤗"), PositionEncodingKind::Utf16)
            .is_ok());
        let rope = Rope::from_str("🤗");
        assert_eq!(doc.text.to_string(), rope.to_string());

        assert!(doc
            .apply_content_change(
                new_change!(0, rope.len_utf16_cu(), 0, rope.len_utf16_cu(), "\r\n"),
                PositionEncodingKind::Utf16
            )
            .is_ok());
        let rope = Rope::from_str("🤗\r\n");
        assert_eq!(doc.text.to_string(), rope.to_string());

        assert!(doc
            .apply_content_change(
                new_change!(0, '🤗'.len_utf16(), 0, '🤗'.len_utf16(), "\n"),
                PositionEncodingKind::Utf16
            )
            .is_ok());
        let rope = Rope::from_str("🤗\n\r\n");
        assert_eq!(doc.text.to_string(), rope.to_string());

        assert!(doc
            .apply_content_change(
                new_change!(0, '🤗'.len_utf16(), 2, 0, ""),
                PositionEncodingKind::Utf16
            )
            .is_ok());
        let rope = Rope::from_str("🤗");
        assert_eq!(doc.text.to_string(), rope.to_string());
    }

    #[tokio::test]
    // Ensure that the three stays consistent across updates.
    async fn test_document_update_tree_consistency_easy() {
        let a = "let a = '你好';\rlet b = 'Hi, 😊';";

        let mut document = Document::open(&LanguageId::JavaScript.to_string(), a)
            .await
            .unwrap();

        document
            .apply_content_change(new_change!(0, 9, 0, 11, "𐐀"), PositionEncodingKind::Utf16)
            .unwrap();

        let b = "let a = '𐐀';\rlet b = 'Hi, 😊';";

        assert_eq!(document.text.to_string(), b);

        let mut parser = Parser::new();

        parser
            .set_language(tree_sitter_javascript::language())
            .unwrap();

        let b_tree = parser.parse(b, None).unwrap();

        assert!(nodes_are_equal_recursive(
            &document.tree.unwrap().root_node(),
            &b_tree.root_node()
        ));
    }

    #[tokio::test]
    async fn test_document_update_tree_consistency_medium() {
        let a = "let a = '🥸 你好';\rfunction helloWorld() { return '🤲🏿'; }\nlet b = 'Hi, 😊';";

        let mut document = Document::open(&LanguageId::JavaScript.to_string(), a)
            .await
            .unwrap();

        document
            .apply_content_change(new_change!(0, 14, 2, 13, "，"), PositionEncodingKind::Utf16)
            .unwrap();

        let b = "let a = '🥸 你好，😊';";

        assert_eq!(document.text.to_string(), b);

        let mut parser = Parser::new();

        parser
            .set_language(tree_sitter_javascript::language())
            .unwrap();

        let b_tree = parser.parse(b, None).unwrap();

        assert!(nodes_are_equal_recursive(
            &document.tree.unwrap().root_node(),
            &b_tree.root_node()
        ));
    }

    fn nodes_are_equal_recursive(node1: &Node, node2: &Node) -> bool {
        if node1.kind() != node2.kind() {
            return false;
        }

        if node1.start_byte() != node2.start_byte() {
            return false;
        }

        if node1.end_byte() != node2.end_byte() {
            return false;
        }

        if node1.start_position() != node2.start_position() {
            return false;
        }

        if node1.end_position() != node2.end_position() {
            return false;
        }

        if node1.child_count() != node2.child_count() {
            return false;
        }

        for i in 0..node1.child_count() {
            let child1 = node1.child(i).unwrap();
            let child2 = node2.child(i).unwrap();

            if !nodes_are_equal_recursive(&child1, &child2) {
                return false;
            }
        }

        true
    }
}
