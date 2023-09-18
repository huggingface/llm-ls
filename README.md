# llm-ls

> [!IMPORTANT]
> This is currently a work in progress, expect things to be broken!

**llm-ls** is a LSP server leveraging LLMs for code completion (and more?).

## Compatible extensions

- [x] [llm.nvim](https://github.com/huggingface/llm.nvim)
- [ ] [huggingface-vscode](https://github.com/huggingface/huggingface-vscode)
- [ ] [jupytercoder](https://github.com/bigcode-project/jupytercoder)

## Installation

### Neovim

`llm-ls` can be installed via [mason.nvim](https://github.com/williamboman/mason.nvim). With `mason.nvim` installed run the following command to install `llm-ls`:

```vim
:MasonInstall llm-ls
```

Then reference `llm-ls` path in your `llm.nvim` configuration:

```lua
{
  "huggingface/llm.nvim",
  opts = {
    lsp = {
      enabled = true,
      bin_path = vim.api.nvim_call_function("stdpath", { "data" }) .. "/mason/bin/llm-ls",
    },
  },
},
```
