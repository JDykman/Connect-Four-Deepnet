---@diagnostic disable: undefined-global
local EXE_NAME = "POOGLE_hot_reload"
local BIN_DIR = "dist/bin"

local function build_and_run()
  vim.cmd("wa")

  -- Mirror build.sh's pgrep check (Linux truncates process names to 15 chars)
  local app_running = vim.fn.system("pgrep -x " .. EXE_NAME:sub(1, 15)):gsub("%s+", "") ~= ""

  -- Build (.so only if running, full build if not)
  local output = vim.fn.system("bash build.sh")
  if vim.v.shell_error ~= 0 then
    vim.notify("Build failed:\n" .. output, vim.log.levels.ERROR)
    return
  end

  -- Launch detached so Neovim stays responsive
  if not app_running then
    local ld_path = BIN_DIR .. ":" .. (os.getenv("LD_LIBRARY_PATH") or "")
    vim.fn.jobstart(
      { "env", "LD_LIBRARY_PATH=" .. ld_path, "./" .. BIN_DIR .. "/" .. EXE_NAME },
      { detach = true, cwd = vim.fn.getcwd() }
    )
  end
end

vim.keymap.set("n", "<leader>r", build_and_run, { desc = "Build & Run (build.sh)" })

-- Re-apply after LspAttach since LazyVim's Snacks.keymap debounces at 100ms and overwrites buffer keymaps
vim.api.nvim_create_autocmd("LspAttach", {
  callback = function(args)
    vim.defer_fn(function()
      vim.keymap.set("n", "<leader>r", build_and_run, { desc = "Build & Run (build.sh)", buffer = args.buf })
    end, 200)
  end,
})
