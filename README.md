# ccserver

> [!IMPORTANT]
> This is currently a work in progress.

**ccserver** is a LSP server for ML code completion (and more?).

## Developing

Clone/fork this repo and run `cargo build [--release]`.

Then add the following code to your lua config:

```lua
local client_id = vim.lsp.start({
  name = "ccserver",
  cmd = { "/path/to/ccserver/target/{debug|release}/ccserver" },
  root_dir = vim.fs.dirname(vim.fs.find({ ".git" }, { upward = true })[1]),
})

if client_id == nil then
  vim.notify("[ccserver] Error starting server", vim.log.levels.ERROR)
else
  local augroup = "ccserver"

  vim.api.nvim_create_augroup(augroup, { clear = true })

  vim.api.nvim_create_autocmd("BufEnter", {
    pattern = "*",
    callback = function(ev)
      if not vim.lsp.buf_is_attached(ev.buf, client_id) then
        vim.lsp.buf_attach_client(ev.buf, client_id)
      end
    end,
  })
end
```
