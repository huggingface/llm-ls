use lsp_types::request::Request;

use crate::llm_ls::{
    AcceptCompletionParams, GetCompletionsParams, GetCompletionsResult, RejectCompletionParams,
};

#[derive(Debug)]
pub enum GetCompletions {}

impl Request for GetCompletions {
    type Params = GetCompletionsParams;
    type Result = GetCompletionsResult;
    const METHOD: &'static str = "llm-ls/getCompletions";
}

#[derive(Debug)]
pub enum AcceptCompletion {}

impl Request for AcceptCompletion {
    type Params = AcceptCompletionParams;
    type Result = ();
    const METHOD: &'static str = "llm-ls/acceptCompletion";
}

#[derive(Debug)]
pub enum RejectCompletion {}

impl Request for RejectCompletion {
    type Params = RejectCompletionParams;
    type Result = ();
    const METHOD: &'static str = "llm-ls/rejectCompletion";
}
