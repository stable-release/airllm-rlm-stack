# rlm-rs — Recursive Language Model harness in Rust

> Canonical standalone home: **[stable-release/rlm-rs](https://github.com/stable-release/rlm-rs)**
> (releases, CI, contributing). This copy is vendored into the stack for the
> one-clone experience.

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
| hosted-API and vLLM clients     | OpenAI-compatible client to local llama-server |
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

### Existing MLX server

On Apple silicon, start the local MLX server separately, then use the bounded
[MLX example config](rlm.config.mlx.example.json):

```bash
curl --fail http://127.0.0.1:8090/health
./target/release/rlm run \
  --config rlm.config.mlx.example.json --no-server --iters 6 \
  -q "..." -c doc.txt
```

The `--no-server` flag prevents the harness from attempting to launch the
Windows llama.cpp executable. Do not use `--model` in this mode; it is a GGUF
auto-start override and does not select the model already loaded by the MLX
endpoint.

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

## Control plane (`rlm daemon`)

A loopback-only run service ported from the fork's `rlmd` design (see
`agent/offline-rlm-control-plane`), keeping its security posture:

- **Binds 127.0.0.1 only** — remote access is your tunnel's job (SSH, Tailscale
  Serve, or a TLS reverse proxy); the daemon refuses to listen on LAN interfaces.
- **Bearer auth**: set `RLM_API_TOKEN` and every `/v1/*` request must send
  `Authorization: Bearer <token>`. `/health` stays open but exposes no run data.
- **Capabilities default-deny** and are intersected with server config (v0
  capability: `memory` — persistent remember/recall; `daemon_allow_memory`
  must also be true server-side).
- **No local file access for clients**: contexts are supplied inline in the
  request body (bounded by `daemon_max_context_chars`); the daemon never opens
  local paths on a client's behalf. The model's script environment remains
  sandboxed Rhai with no filesystem, network, or exec functions registered.
- **Limits only ratchet down**: per-run `max_iterations` / `max_tokens` /
  `max_run_seconds` are clamped to the configured ceilings.
- **One run at a time** (single generation permit, bounded queue, 429 on
  over-admission); cancellation is cooperative at iteration boundaries — a
  dispatched generation is drained, never interrupted.
- **Bounded store**: terminal snapshots append to `runs.jsonl` (gitignored) as
  previews; events are capped per run.

```bash
RLM_API_TOKEN=secret ./target/release/rlm.exe daemon        # port 8093

curl -X POST http://127.0.0.1:8093/v1/runs \
  -H "Authorization: Bearer secret" -H "Content-Type: application/json" \
  -d '{"prompt":"...","contexts":[{"name":"doc","text":"..."}],
       "capabilities":{"memory":false},"limits":{"max_iterations":6}}'
# -> {"run_id":"run-...","status":"queued",...}

curl -N http://127.0.0.1:8093/v1/runs/<id>/events -H "Authorization: Bearer secret"   # SSE (replay with ?after=<seq>)
curl -X POST http://127.0.0.1:8093/v1/runs/<id>/cancel -H "Authorization: Bearer secret"
```

## Build

```powershell
cargo build --release
```
