# AirLLM + RLM Local Stack

A self-contained local inference stack built on two upstreams:

- **[AirLLM](https://github.com/lyogavin/airllm)** — layer-streamed inference: run checkpoints far larger than VRAM by keeping one layer resident at a time. This repo is a fork; the original README lives upstream.
- **[RLM](https://github.com/stable-release/rlm)** — Recursive Language Models: long context lives in a code environment the model navigates, not in the prompt. Ported here to Rust (`rlm-rs/`) as a single ~3 MB binary.

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

```bash
# 1. Serve a local model with mlx-lm (OpenAI-compatible, replaces llama-server)
pip install mlx-lm
HF_HUB_OFFLINE=1 python -m mlx_lm server --model /path/to/model \
  --host 127.0.0.1 --port 8090 --chat-template-args '{"enable_thinking":false}'

# 2. Build and use the harness against it
cd rlm-rs && cargo build --release
./target/release/rlm run --no-server -q "..." -c doc.txt
```

AirLLM MLX streaming classes and MTP tree-speculation work live on the
[`agent/qwen38-mlx`](https://github.com/stable-release/airllm/tree/agent/qwen38-mlx) branch.

## Decide before you start

| Decision | Options | Default |
|---|---|---|
| Main model | any instruct GGUF; 4-bit quant recommended | smallest `.gguf` in `models\` |
| Worker model | any small GGUF in `models\worker\`, or none | enabled if present |
| VRAM split | `n_gpu_layers` (main) vs `worker_n_gpu_layers` | 36 / all — budget explicitly, two auto-fits overcommit and paging wrecks both |
| Context | `ctx_size` — the RLM harness navigates long docs itself, so bigger is rarely better | 32768 |
| Thinking | per-tier chain-of-thought; big latency cost | off everywhere |
| Speculative decoding | `--spec-type draft-mtp` with 1-token drafts (models with an MTP head) | on |
| API exposure | none / `rlm daemon` (loopback + bearer token) | none |

## Parameters (`rlm-rs/rlm.config.json`)

Created from [`rlm.config.example.json`](rlm-rs/rlm.config.example.json) on first run. Relative paths resolve against the repo root.

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
| 8090 | llama.cpp, main model (`local-main`) | chat clients, RLM root |
| 8091 | AirLLM safetensors streaming (instrumented) | full-precision reference |
| 8092 | llama.cpp, worker (`local-worker`) | fast leaf calls, agent tools |
| 8093 | `rlm daemon` control plane | run API + SSE events |

## Benchmarks

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
