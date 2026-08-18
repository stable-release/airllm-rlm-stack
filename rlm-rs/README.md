# rlm-rs — Recursive Language Model harness in Rust

A Rust port of the Recursive Language Model (RLM) inference pattern from
[stable-release/rlm](https://github.com/stable-release/rlm) (MIT OASYS lab's `rlms`
Python package), acting as the **context and memory manager** for the local GGUF
models served through this repo's airllm fork + llama.cpp runtime.

Instead of pasting long context into the model's window, the model interacts with
its context **programmatically**: it writes small scripts in a sandboxed environment
where the context lives as named variables, and it can issue **recursive sub-calls**
over slices of that context. The Python original uses a Python REPL; this port uses
[Rhai](https://rhai.rs) (a small, pure-Rust embedded scripting engine), which keeps
the whole harness a single ~3 MB native binary with no Python runtime.

## Mapping from the Python `rlms` package

| Python rlms                     | rlm-rs                                        |
|---------------------------------|-----------------------------------------------|
| OpenAI/Anthropic/vLLM clients   | OpenAI-compatible client to local llama-server |
| Python REPL environment         | Rhai script environment (sandboxed, metered)  |
| context-as-variables            | `ctx_list` / `ctx_len` / `peek` / `grep`      |
| recursive sub-calls             | `llm(prompt)`, `llm_on(prompt, ctx, start, len)` (large slices spawn a nested RLM loop, depth-limited) |
| multi-turn persistence          | `remember` / `recall` + `rlm_memory.json` session log |

## Usage

```powershell
# one-shot question over a long file (auto-starts llama-server if needed)
.\target\release\rlm.exe run -q "Summarize the plot and list every character" -c path\to\novel.txt

# interactive chat; conversation history is itself managed as navigable context
.\target\release\rlm.exe chat

# just bring the model server up (OpenAI-compatible at http://127.0.0.1:8090/v1)
.\target\release\rlm.exe serve

# use a specific GGUF instead of the auto-discovered one
.\target\release\rlm.exe run -q "..." --model models\your-model.gguf
```

Configuration lives in `rlm.config.json` (created from defaults on first run; see
[rlm.config.example.json](rlm.config.example.json)). An empty `model_path`
auto-discovers the smallest GGUF under `models\`; an empty `worker_model_path`
auto-discovers `models\worker\` (no worker model = single-model mode). Relative
paths resolve against the repo root, so the checkout is location-independent.

**Two-tier inference:** leaf sub-calls (`llm`, single-shot `llm_on`) run on the
fast worker model; the root loop and nested recursive loops stay on the main
model. Budget VRAM explicitly via `n_gpu_layers` / `worker_n_gpu_layers` — two
auto-fits overcommit the GPU and paging wrecks both models.

## How a run works

1. `rlm` health-checks `http://127.0.0.1:8090`; if nothing is there it spawns
   `llama-server.exe` from `runtime\llama.cpp` with the configured
   GGUF and waits for the model to load.
2. Context files are loaded into the environment as named variables (never into
   the prompt).
3. The root model iterates: script, REPL output, script ... until it calls
   `finish(answer)` (or answers in plain text / hits the iteration cap).
4. `llm_on` slices larger than `recurse_threshold` chars start a **nested RLM
   loop** over that slice, up to `max_depth`.
5. Facts saved with `remember()` and a rolling session log persist in
   `rlm_memory.json` and are surfaced to future sessions.

## Build

```powershell
cargo build --release
```
