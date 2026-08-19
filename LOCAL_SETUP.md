# Local stack: AirLLM fork + llama.cpp runtime + rlm-rs

A self-contained local LLM stack: bring your own model files, drop them into
`models\`, and the harnesses auto-discover them. Everything lives inside the repo
root; models and runtime binaries are never committed (see `.gitignore`).

## Layout

| Path | What |
|------|------|
| `models\*.gguf` | your GGUF quants (llama.cpp path) — smallest one is picked by default |
| `models\worker\*.gguf` | optional small fast model for RLM leaf sub-calls (port 8092) |
| `models\<dir>\*.safetensors` | your safetensors checkpoints (AirLLM streaming path) |
| `runtime\llama.cpp\` | llama.cpp Windows CUDA build (`llama-server.exe` etc.) |
| `air_llm\airllm\airllm_gguf.py` | AirLLM GGUF backend (`AutoModel.from_pretrained` on any `.gguf` path routes here) |
| `rlm-rs\` | Rust RLM harness (context + memory management, recursive sub-calls) |
| `.cargo\`, `.venv\` | Rust crate cache and Python deps kept in-repo |

## Managing the servers

```powershell
.\llm-stack.ps1 start     # both servers under a supervisor (auto-restart on crash/hang)
.\llm-stack.ps1 status
.\llm-stack.ps1 stop      # kill switch
```

llama.cpp serves on port 8090, the AirLLM streaming server on 8091, and the
optional worker on 8092. All are OpenAI-compatible under `/v1`.

There is also a loopback-only **control plane** (`rlm-rs\target\release\rlm.exe
daemon`, port 8093): a run API with bearer auth, default-deny capabilities,
inline-context-only ingestion, SSE progress events, cooperative cancellation,
and a single-generation permit — see [rlm-rs/README.md](rlm-rs/README.md).

## Two-tier inference (worker model)

Drop a small quant (e.g. a ~4B) into `models\worker\` and the RLM harness routes
leaf sub-calls (`llm`, single-shot `llm_on`) to it while root reasoning stays on
the big model. Measured on the 16 GB card: worker ~98 tok/s alongside the main
model at ~4.7 tok/s — a ~20x speedup for the calls that carry most of the token
volume, at ~24% cost to root speed.

**VRAM must be budgeted explicitly** (`n_gpu_layers` for the main model,
`worker_n_gpu_layers` for the worker): two auto-fits overcommit the GPU and
Windows silently pages VRAM, collapsing both models to ~1 tok/s. The shipped
defaults (38 main layers + full-GPU ~2-3 GB worker) fit a 16 GB card with a
27B-class main model.

**Thinking is off by default on both tiers** (`root_thinking` / `worker_thinking`).
For reasoning models the RLM's iterative script/REPL cycle is itself the reasoning
scaffold; chain-of-thought on top of it costs minutes per iteration at big-model
speeds (one measured leaf call: 512 reasoning tokens vs 35 for the plain answer).
An end-to-end RLM document query dropped from 197s to 110s with thinking off and
all optimizations active.

**MTP speculative decoding works — but only with 1-token drafts.** On weight-edited
models the MTP head is uncertain rather than broken: 3-token linear drafts accept
only ~44% and lose net speed, while `--spec-draft-n-max 1` keeps the high-confidence
first position (~73% acceptance) for a measured **+17-27%** decode speedup. The
shipped config enables this via `extra_server_args`. (Credit: the fork's
`agent/qwen38-mlx` branch diagnosed rejection ranks and showed the misses are
recoverable near-misses; its MLX tree-speculation work generalizes this further.)

## Why GGUF goes through llama.cpp and not AirLLM's layer streamer

AirLLM's native loader streams **safetensors** checkpoints layer-by-layer; the GGUF
container is a llama.cpp format it cannot parse. The fork routes `.gguf` paths to
a backend that drives `llama-server`, which achieves the same goal for GGUF natively:
mmap'd weights + `--n-gpu-layers` auto-fit stream/offload layers across VRAM, RAM and
disk. If you want AirLLM's own streamer, point it at a safetensors checkpoint instead.

## Hardware fit (measured with a 27B-class dense model on a 16 GB VRAM / 32 GB RAM machine)

- **~4-bit quant (~16 GB)** — most layers on GPU, interactive speeds (~7 tok/s measured).
- **8-bit quant (~29 GB)** — roughly half offloaded, the rest in RAM; slower but
  still fully memory-resident.
- **Full BF16 (~54 GB)** — exceeds VRAM+RAM combined; llama.cpp pages weights from
  disk every token. **Measured: 25 s/token** (identical cold and warm — fully
  SSD-bandwidth-bound). Full precision is a correctness/quality reference only;
  any layer-streaming runtime (llama.cpp mmap or AirLLM safetensors sharding)
  hits the same disk-bandwidth wall because a dense model touches every weight
  for every token.

## Apple-silicon fit (sanitized measurements)

On a base-class Apple-silicon host with 8 GB of unified memory, a text-only
9B-class dense model packaged as MLX affine 4-bit/group-64 sustained **8.73
tok/s on its first 64-token run** and **12.89 tok/s on a later warm 64-token
run**. Both runs reported a **5.23 GB peak** and zero process swaps. The package
held 4.692 GiB of logical weights; all 927 arrays were verified bit-for-bit
against the source representation after repackaging.

With those same packed arrays, full component streaming reached **0.317 tok/s**
and a 4.0 GiB pinned-component plan reached **0.421 tok/s**. Pinning additional
components crossed the memory-pressure cliff: the embedding-plus-decoder plan
fell to **0.148 tok/s**, while streaming both endpoint matrices around resident
decoders reached **0.229 tok/s**. The standard package is faster because it uses
one lazy graph and memory-mapped shards instead of synchronized component loads,
not because it was requantized.

The same memory tier can run a streamed 27B-class affine-3 target with exact
native tree verification, but at a different operating point. An isolated exact
leaf-verifier cycle reached **0.548 tok/s** at **2.96 GiB peak**, and the first
integrated smoke cycle produced six steady tokens in 10.800 seconds (**0.556
tok/s**) at **2.07 GiB peak**. The backend-consistent sustained run settled at
**0.393 tok/s** over a 30-token continuation with the same **2.07 GiB peak** and
a fresh full-output reference pass. Widening every uncertain branch regressed to
**0.236 tok/s** and **4.61 GiB**, so selective speculation is required. The
single-cycle numbers are useful latency observations, not sustained-capacity
figures. They predate the recurrence-backend consistency correction, although
their token-level checks passed; use the `0.393 tok/s` result as the correctness
baseline.

These disclosures intentionally omit checkpoint identity, local paths, host and
account names, OS build, timestamps, prompts, and raw logs. They are performance
records, not claims that every model of the same parameter count will behave
identically.

For this path, start `mlx_lm.server` manually and invoke the Rust harness with
both `--config rlm.config.mlx.example.json` and `--no-server`. The dedicated
config disables the llama.cpp worker/autostart assumptions and caps root/leaf
responses at 512/256 tokens. `--max-tokens` on the MLX server is a default, not
a hard cap on the larger value a client can send, so keeping the harness config
bounded is necessary. The complete tested two-terminal command sequence is in
the top-level README.

## True AirLLM layer-streaming (safetensors)

Safetensors checkpoints route through AirLLM's native per-layer streamer — one
layer on the GPU at a time, so VRAM use tracks the largest layer, not model size.
Hybrid-VLM checkpoints (`*ForConditionalGeneration` with a nested
`language_model`) are supported via the `AirLLMQwen3_5`-style subclasses; first
load splits the checkpoint into per-layer shards (one-time, ~equal extra disk;
`delete_original=True` reclaims it).

Serve it OpenAI-compatible for rlm-rs (port 8091, llama.cpp keeps 8090):

```powershell
.venv\Scripts\python.exe serve_airllm.py            # auto-discovers the checkpoint
rlm-rs\target\release\rlm.exe run --port 8091 --no-server -q "..."
```

**Measured (27B-class BF16, 8-token run):** 57 s/token; **peak VRAM 3.7 GB**
(the layer-streaming promise holds — a ~54 GB model runs in under 4 GB); process
RAM 2.7 GB. Per generated token one full pass (~54 GB) streams through:
430 GB moved for 8 tokens, throughput-bound at ~2.7 GB/s (NVMe speed — the reads
surface in the CPU-to-GPU phase because safetensors loads are mmap-lazy).
Comparison: llama.cpp BF16 GGUF 25 s/token (keeps ~15 GB resident in VRAM);
~4-bit quant ~7 tok/s. Every backend hits the same SSD-bandwidth wall at full
precision; AirLLM just trades speed for minimal VRAM. Metrics: every response
carries `airllm_metrics`, and `GET /metrics` has cumulative counters.

## Python (AirLLM API)

```python
from airllm import AutoModel

model = AutoModel.from_pretrained(r"models\your-model.gguf")
print(model.generate("Write a haiku about GPUs", max_new_tokens=64))
print(model.chat([{"role": "user", "content": "Hello!"}]))
# model.base_url -> OpenAI-compatible endpoint shared with rlm-rs
```

## Rust (RLM harness)

```powershell
rlm-rs\target\release\rlm.exe run -q "What changed in chapter 12?" -c path\to\big_file.txt
```

See [rlm-rs/README.md](rlm-rs/README.md) for the full RLM environment reference.
