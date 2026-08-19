# AirLLM + RLM Local Stack

A self-contained local inference stack built on two upstreams:

- **[AirLLM](https://github.com/lyogavin/airllm)** — layer-streamed inference: run checkpoints far larger than VRAM by keeping one layer resident at a time. This repo is a fork; the original README lives upstream.
- **[RLM](https://github.com/stable-release/rlm)** — Recursive Language Models: long context lives in a code environment the model navigates, not in the prompt. Ported to Rust as a single ~3 MB binary: **[rlm-rs](https://github.com/stable-release/rlm-rs)** (canonical home; vendored here as `rlm-rs/` for the one-clone experience).

Runs on **Windows (CUDA)** and **Apple Silicon (MLX)**.

What you get:

- **Two-tier serving** — a large GGUF for root reasoning, a small worker model answering leaf sub-calls ~20x faster
- **RLM harness** — recursive context navigation, persistent memory, sandboxed script environment (no filesystem/network/exec)
- **Control plane** — loopback-only run API with bearer auth, capability gating, SSE events, cancellation
- **AirLLM streaming** — full-precision safetensors checkpoints in under 4 GB VRAM
- **OpenAI-compatible everywhere** — point any client (AnythingLLM, Open WebUI, ...) at any tier

## Quick start

### Windows (CUDA)

```powershell
# 1. Unpack a llama.cpp CUDA build into runtime\llama.cpp\  (needs llama-server.exe)
# 2. Drop models in:
#      models\your-model.gguf              main model (smallest .gguf is auto-picked)
#      models\worker\small-model.gguf      optional fast worker
#      models\<name>\*.safetensors         optional, for AirLLM streaming
# 3. Build the harness
cd rlm-rs; cargo build --release; cd ..

# 4. Run
.\llm-stack.ps1 start                                  # supervised servers: 8090 main, 8091 airllm, 8092 worker
.\rlm-rs\target\release\rlm.exe run -q "..." -c doc.txt   # one-shot RLM query (auto-starts servers itself)
.\llm-stack.ps1 stop                                   # kill switch
```

### Apple Silicon (MLX)

One-time build prerequisites (skip anything already installed):

```bash
xcode-select --install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
cargo --version
```

The Rust command is the official `rustup` installer. Review it at
[rust-lang.org](https://www.rust-lang.org/tools/install) if your environment
requires a different installation policy.

```bash
# Terminal 1: serve an already-local model (OpenAI-compatible)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade mlx-lm

MODEL_PATH="/path/to/local/model"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m mlx_lm server \
  --model "$MODEL_PATH" --host 127.0.0.1 --port 8090 \
  --decode-concurrency 1 --prompt-concurrency 1 \
  --prefill-step-size 512 --prompt-cache-size 0 --max-tokens 512 \
  --chat-template-args '{"enable_thinking":false}'

# Terminal 2: build, verify the endpoint, then run the harness
cd rlm-rs
cargo build --release
curl --fail http://127.0.0.1:8090/health
./target/release/rlm run \
  --config rlm.config.mlx.example.json --no-server --iters 6 \
  -q "..." -c doc.txt
```

`--no-server` is required for this path: the model server is already running,
and the harness must not try to launch the Windows-oriented llama.cpp runtime.
Do not pass `rlm --model` here; that option selects a GGUF for auto-started
llama.cpp and does not choose the model already loaded by the MLX server. The
MLX example config also keeps client request budgets at 512/256 tokens; the
server's `--max-tokens` value is only a default and does not override a larger
per-request value from the harness.

On the tested 8 GB memory tier, start with prompt plus answer below roughly 8K
tokens. The model architecture may advertise a much larger window, but long
prefill and cache growth reduce practical headroom. The RLM harness is the
preferred way to navigate larger documents without placing all of them in the
live model window.

## Decide before you start

| Decision | Options | Default |
|---|---|---|
| Main model | instruct GGUF through llama.cpp, or a complete local MLX package | smallest `.gguf` in `models\`; explicit `MODEL_PATH` for MLX |
| Worker model | any small GGUF in `models\worker\`, or none | enabled if present |
| Accelerator split | llama.cpp: `n_gpu_layers` vs `worker_n_gpu_layers`; MLX: one serialized resident model | 36 / all for llama.cpp; one model for MLX |
| Context | `ctx_size` — the RLM harness navigates long docs itself, so bigger is rarely better | 32768 llama.cpp / 8192 MLX example |
| Thinking | per-tier chain-of-thought; big latency cost | off everywhere |
| Speculative decoding | llama.cpp: `--spec-type draft-mtp` with 1-token drafts when supported | on in the llama.cpp example / off in the MLX example |
| API exposure | none / `rlm daemon` (loopback + bearer token) | none |

## Parameters (`rlm-rs/rlm.config.json`)

Created from [`rlm.config.example.json`](rlm-rs/rlm.config.example.json) on first
run. Relative paths resolve against the repo root. The table below describes the
llama.cpp defaults; Apple-silicon users should select
[`rlm.config.mlx.example.json`](rlm-rs/rlm.config.mlx.example.json), which uses an
8K context, no worker/autostart, and smaller generation budgets.

| Key | Purpose | Default |
|---|---|---|
| `model_path` / `worker_model_path` | empty = auto-discover | `""` |
| `ctx_size` / `worker_ctx` | context windows | 32768 / 8192 |
| `n_gpu_layers` / `worker_n_gpu_layers` | GPU layer budget per tier | 36 / 99 |
| `kv_cache_type` | KV quantization | `q8_0` |
| `extra_server_args` | raw llama-server flags (MTP, alias, template kwargs) | see example |
| `root_thinking` / `worker_thinking` | chain-of-thought per tier | `false` |
| `temperature`, `max_tokens`, `sub_max_tokens` | sampling and budgets | 0.7 / 4096 / 2048 |
| `max_iterations`, `max_depth`, `recurse_threshold` | RLM loop bounds | 12 / 2 / 30000 |
| `max_run_seconds` | wall-clock cap per run | 900 |
| `daemon_port`, `daemon_allow_memory`, `daemon_max_context_chars` | control plane | 8093 / false / 5M |

## Endpoints

| Port | Service | Use |
|---|---|---|
| 8090 | main local model server (llama.cpp or MLX) | chat clients, RLM root |
| 8091 | AirLLM safetensors streaming (instrumented) | full-precision reference |
| 8092 | llama.cpp, worker (`local-worker`) | fast leaf calls, agent tools |
| 8093 | `rlm daemon` control plane | run API + SSE events |

## Benchmarks

### Apple Silicon (MLX)

Sanitized local measurements on a base-class Apple-silicon host with 8 GB of
unified memory are below. Checkpoint names, host identity, filesystem paths, OS
account/build details, timestamps, prompts, and raw logs are intentionally
omitted. Thinking was disabled and inference was local-only.

| Path | Model class | Precision | Workload | Decode | Reported peak | Swap | Validation |
|---|---:|---|---|---:|---:|---:|---|
| Standard resident MLX package | 9B dense, text-only | affine 4-bit, group 64 | 64 generated tokens | **8.73 tok/s cold; 12.89 tok/s warm** | **5.23 GB** | 0 | 31 greedy decisions matched the streamed target |
| AirLLM component streamer | same 9B target | same packed arrays, all streamed | short greedy decode | **0.317 tok/s** | not recorded | not recorded | target baseline |
| AirLLM partial residency | same 9B target | same packed arrays, 4.0 GiB pinned | short greedy decode | **0.421 tok/s** | not recorded | not recorded | target baseline |
| Isolated exact leaf verifier | 27B dense, streamed | affine 3-bit target + native draft head | favorable single cycle | **0.548 tok/s** | **2.96 GiB** | not recorded | target predictions and recovered-cache next token matched |
| Integrated leaf-verifier smoke | same 27B target | same | 6 steady tokens in 10.800 s | **0.556 tok/s** | **2.07 GiB** | not recorded | independent full-output reference passed |
| AirLLM exact leaf-tree verifier | 27B dense, streamed | affine 3-bit target + native draft head | 30-token sustained continuation | **0.393 tok/s** | **2.07 GiB** | not recorded | fresh full-output reference passed |
| Indiscriminately widened tree | 27B dense, streamed | same target | one scaling cycle | **0.236 tok/s** | **4.61 GiB** | not recorded | exact, but slower |

The resident 9B package contains 4.692 GiB of logical text weights. Its 927
arrays were repackaged without requantization and verified bit-for-bit against
the streamed source. The two 64-token runs are reported separately rather than
presented as a statistical range: `8.73 tok/s` was the first cold run and
`12.89 tok/s` was the later warm run, both at the same reported peak with no
process swaps.

The resident and streamed 9B rows use the same packed arrays. Pinning more
component files did not converge on resident performance: an embedding-plus-
decoder layout crossed the memory-pressure cliff and fell to `0.148 tok/s`,
while a decoder-only layout reached `0.229 tok/s`. The decisive change was the
standard lazy MLX graph, which avoids repeated component loads and per-layer
synchronization; it was not a second quantization pass.

The final 27B result includes tree construction, target verification, selected
cache recovery, and per-cycle cleanup; the separate final reference traversal
is excluded from throughput. It improved the previous adaptive implementation
by 29% while reducing peak allocation by 64%. The wider-tree result is retained
because it establishes that adding branches without a selective policy is a
performance and memory regression.

The `0.548` and `0.556 tok/s` rows are genuine but deliberately labeled as
single-cycle measurements. They used a favorable first tree and do not capture
the acceptance variation of a longer continuation. Both also predate the
recurrence-backend consistency correction: their token-level validations passed,
but they are not the sustained correctness baseline. The `0.393 tok/s` row is
backend-consistent and should be used for capacity planning.

### CUDA

27B-class dense model, measured on a 16 GB VRAM consumer GPU with 32 GB system RAM and PCIe-4 NVMe.

| Path | Precision | Decode | VRAM |
|---|---|---|---|
| llama.cpp, solo | 4-bit | ~7 tok/s | ~15 GB |
| llama.cpp, two-tier + MTP | 4-bit | ~5.5 tok/s | ~12 GB |
| worker (4B-class) | 4-bit | 80–98 tok/s | ~3 GB |
| llama.cpp mmap | **BF16 (~54 GB)** | **25 s/token** | ~15 GB |
| AirLLM layer streaming | **BF16 (~54 GB)** | **57 s/token** | **3.7 GB peak** |

Notes:

- **BF16 runs, but is disk-bound.** A dense model touches every weight per token; whatever exceeds VRAM+RAM streams from SSD each pass (~54 GB/token at ~2.7 GB/s). AirLLM's streaming trades speed for the 3.7 GB VRAM floor — full precision on tiny GPUs is a correctness reference, not a daily driver.
- Prompt prefill is much faster than decode (~50 tok/s even at low GPU residency); long inputs are cheap, long outputs are not.
- 1-token MTP speculative drafts add +17–27% decode on models shipping an MTP head (deeper drafts hurt on weight-edited models — acceptance drops to ~44%).
- Disabling thinking cuts typical leaf answers from ~500 tokens to ~35 with no quality loss on retrieval-style calls; an end-to-end RLM document query dropped 197 s → 110 s.
- The AirLLM server reports per-request `airllm_metrics` (disk/GPU bandwidth, VRAM/RAM peaks) and cumulative `GET /metrics`.

## More

- [LOCAL_SETUP.md](LOCAL_SETUP.md) — full stack layout and rationale
- [rlm-rs/README.md](rlm-rs/README.md) — RLM environment reference, control-plane API, security posture

## License

Fork of [AirLLM](https://github.com/lyogavin/airllm) (Apache-2.0 / MIT per upstream); `rlm-rs` is MIT. RLM concept: [MIT OASYS lab](https://github.com/stable-release/rlm).
