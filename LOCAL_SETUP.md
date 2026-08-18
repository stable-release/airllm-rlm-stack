# Local stack: AirLLM fork + llama.cpp runtime + rlm-rs

A self-contained local LLM stack: bring your own model files, drop them into
`models\`, and the harnesses auto-discover them. Everything lives inside the repo
root; models and runtime binaries are never committed (see `.gitignore`).

## Layout

| Path | What |
|------|------|
| `models\*.gguf` | your GGUF quants (llama.cpp path) — smallest one is picked by default |
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

llama.cpp serves on port 8090, the AirLLM streaming server on 8091. Both are
OpenAI-compatible under `/v1`.

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
