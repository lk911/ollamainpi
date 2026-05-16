# ollamainpi

An Ollama-style local LLM manager for the PI coding agent, powered by
[`node-llama-cpp`](https://github.com/withcatai/node-llama-cpp).

It gives you all the things you would normally use Ollama for — directly inside PI:

- Download GGUF models from HuggingFace or any URL
- Manage them on disk (`~/.pi/ollamainpi/models/`)
- Run inference (CPU or GPU-accelerated)
- An OpenAI-compatible HTTP server at `http://127.0.0.1:11435/v1`
- Local models registered as a PI **provider**, so they appear in `/model` and `pi --list-models`
- A `local_llm` **tool** so the agent itself can call a local model mid-conversation
- Per-model configuration (context size, GPU layer offload)

## Install

```bash
# Copy this directory into PI's global extensions folder
mkdir -p ~/.pi/agent/extensions
cp -r ./ollamainpi ~/.pi/agent/extensions/

# Install node-llama-cpp + typebox
cd ~/.pi/agent/extensions/ollamainpi
npm install
```

Or, while iterating locally, point PI directly at this directory:

```bash
pi -e ./ollamainpi/src/index.ts
```

## Quick start

```text
# Inside PI:
/llm:pull hf:bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf as llama-3.2-3b
/llm:list
/llm:run llama-3.2-3b Write a haiku about sandboxes.

# Now pick it as the active PI model:
/model
# -> select "Ollama in PI (local)" / llama-3.2-3b
```

## Commands

| Command | Description |
|---|---|
| `/llm:list` | List installed local models |
| `/llm:pull <source> [as <id>]` | Download a GGUF model. Source is `hf:user/repo/file.gguf` or a direct URL. |
| `/llm:rm <id>` | Remove a model (confirms before deleting the file) |
| `/llm:run <id> <prompt>` | One-shot prompt against a local model |
| `/llm:config <id> ctx=<n> gpu=<n\|auto\|max>` | Per-model context size and GPU layer offload |
| `/llm:server start\|stop\|status` | Control the local OpenAI-compatible HTTP server |
| `/llm:port <n>` | Change the server port (default 11435) |
| `/llm:tokens <id> <text>` | Tokenize text with a local model |

## Files

```
ollamainpi/
├── package.json         # Declares node-llama-cpp + typebox deps; entry point
├── src/
│   ├── index.ts         # Extension entry; registers events, commands, tool, provider
│   ├── config.ts        # Storage paths, registry/settings types and I/O
│   ├── model-manager.ts # Pull/list/remove/update GGUF models
│   ├── inference.ts     # node-llama-cpp wrapper (lazy load, GPU auto)
│   └── server.ts        # OpenAI-compatible HTTP server (chat completions, models)
```

## Flags

- `--no-ollamainpi-server` — start PI without auto-launching the local HTTP server.

## Notes

- Default port is **11435** (intentionally different from a host Ollama on 11434).
- Models live under `~/.pi/ollamainpi/models/`. Registry is `~/.pi/ollamainpi/registry.json`.
- GPU acceleration is selected automatically (`getLlama({ gpu: "auto" })`). To force CPU, set `gpu=0` via `/llm:config`.
- The server is bound to `127.0.0.1` only — it is not exposed on the network.
