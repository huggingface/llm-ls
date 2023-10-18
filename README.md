# llm-ls

> [!IMPORTANT]
> This is currently a work in progress, expect things to be broken!

**llm-ls** is a LSP server leveraging LLMs to make your development experience smoother and more efficient.

The goal of llm-ls is to provide a common platform for IDE extensions to be build on. llm-ls takes care of the heavy lifting with regards to interacting with LLMs so that extension code can be as lightweight as possible.

## Features

### Prompt

Uses the current file as context to generate the prompt. Can use "fill in the middle" or not depending on your needs.

It also makes sure that you are within the context window of the model by tokenizing the prompt.

### Telemetry

Gathers information about requests and completions that can enable retraining.

Note that **llm-ls** does not export any data anywhere (other than setting a user agent when querying the model API), everything is stored in a log file if you set the log level to `info`.

### Completion

**llm-ls** parses the AST of the code to determine if completions should be multi line, single line or empty (no completion).

## Compatible extensions

- [x] [llm.nvim](https://github.com/huggingface/llm.nvim)
- [x] [llm-vscode](https://github.com/huggingface/llm-vscode)
- [x] [llm-intellij](https://github.com/huggingface/llm-intellij)
- [ ] [jupytercoder](https://github.com/bigcode-project/jupytercoder)

## Roadmap

- support getting context from multiple files in the workspace
- add `suffix_percent` setting that determines the ratio of # of tokens for the prefix vs the suffix in the prompt
- add context window fill percent or change context_window to `max_tokens`
- filter bad suggestions (repetitive, same as below, etc)
- support for ollama
- support for llama.cpp
- oltp traces ?
