"""OpenAI-compatible HTTP server over the AirLLM safetensors streaming backend.

This is the bridge that lets rlm-rs (or any OpenAI client) route through AirLLM's true
layer-streaming loader — one layer resident on the GPU at a time, weights streamed from
per-layer shards on disk. Stdlib only, single in-flight generation (a semaphore guards the
GPU; layer streaming leaves no room for batching anyway).

Usage:
    .venv\\Scripts\\python.exe serve_airllm.py --model models\\<checkpoint-dir> --port 8091
    rlm-rs\\target\\release\\rlm.exe run --port 8091 --no-server -q "..."

With no --model, the first subdirectory of models\\ containing safetensors is used.

First start on a fresh checkpoint splits it into per-layer shards next to the model
(one-time, needs ~equal extra disk; pass --delete-original to reclaim the unsplit copy).
"""

import argparse
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "air_llm"))

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def memory_snapshot():
    """Current process RAM plus CUDA active/peak allocations, in GB."""
    out = {}
    try:
        import psutil
        p = psutil.Process()
        out["process_ram_gb"] = round(p.memory_info().rss / 1e9, 2)
        vm = psutil.virtual_memory()
        out["system_ram_used_gb"] = round(vm.used / 1e9, 2)
        out["system_ram_total_gb"] = round(vm.total / 1e9, 2)
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            out["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
            out["vram_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            out["vram_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 2)
            out["vram_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
            free, total = torch.cuda.mem_get_info()
            out["vram_free_gb"] = round(free / 1e9, 2)
            out["vram_total_gb"] = round(total / 1e9, 2)
    except Exception:
        pass
    return out


def build_model(args):
    from airllm import AutoModel

    print(f"[serve_airllm] loading {args.model} (first run splits the checkpoint into layer shards)")
    t = time.time()
    model = AutoModel.from_pretrained(
        args.model,
        max_seq_len=args.ctx,
        delete_original=args.delete_original,
    )
    print(f"[serve_airllm] ready in {time.time() - t:.0f}s")
    return model


class Handler(BaseHTTPRequestHandler):
    server_version = "airllm-openai/0.1"
    model = None          # set by main()
    gen_lock = threading.Lock()
    default_max_tokens = 256

    def log_message(self, fmt, *log_args):
        print(f"[serve_airllm] {self.address_string()} {fmt % log_args}")

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "backend": "airllm-safetensors-streaming"})
        elif self.path == "/metrics":
            self._json(200, {
                "cumulative_streaming": self.model.stream_stats.snapshot(),
                "memory": memory_snapshot(),
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            return self._json(400, {"error": {"message": f"bad JSON: {e}"}})

        if self.path == "/v1/chat/completions":
            return self._chat(payload)
        if self.path == "/completion":
            return self._completion(payload)
        return self._json(404, {"error": "not found"})

    # ------------------------------------------------------------------ routes
    def _chat(self, payload):
        messages = payload.get("messages", [])
        max_tokens = int(payload.get("max_tokens") or self.default_max_tokens)
        temperature = float(payload.get("temperature", 0.7))

        tok = self.model.tokenizer
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        content, reasoning, n_in, n_out, secs, metrics = self._generate(text, max_tokens, temperature)

        self._json(200, {
            "object": "chat.completion",
            "model": "airllm",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content,
                            "reasoning_content": reasoning},
            }],
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                      "total_tokens": n_in + n_out},
            "timings": {"predicted_n": n_out, "predicted_ms": secs * 1000,
                        "predicted_per_second": (n_out / secs) if secs > 0 else 0},
            "airllm_metrics": metrics,
        })

    def _completion(self, payload):
        prompt = payload.get("prompt", "")
        max_tokens = int(payload.get("n_predict") or self.default_max_tokens)
        temperature = float(payload.get("temperature", 0.7))
        content, _, n_in, n_out, secs, metrics = self._generate(prompt, max_tokens, temperature,
                                                                apply_think_split=False)
        self._json(200, {
            "content": content,
            "timings": {"predicted_n": n_out, "predicted_ms": secs * 1000,
                        "predicted_per_second": (n_out / secs) if secs > 0 else 0},
            "airllm_metrics": metrics,
        })

    # -------------------------------------------------------------- generation
    def _generate(self, text, max_tokens, temperature, apply_think_split=True):
        import torch

        tok = self.model.tokenizer
        inputs = tok(text, return_tensors="pt").to(self.model.device)
        n_in = inputs["input_ids"].shape[1]

        gen_kwargs = dict(max_new_tokens=max_tokens, use_cache=True)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        with self.gen_lock:
            stats_before = self.model.stream_stats.snapshot()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t = time.time()
            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_kwargs)
            secs = time.time() - t
            streaming = self.model.stream_stats.delta_since(stats_before)

        metrics = {"streaming": streaming, "memory": memory_snapshot()}

        new_tokens = out[0][n_in:]
        raw = tok.decode(new_tokens, skip_special_tokens=True)
        n_out = int(new_tokens.shape[0])
        print(
            f"[serve_airllm] {n_out} tokens in {secs:.0f}s ({n_out / secs if secs else 0:.3f} tok/s) | "
            f"disk: {streaming['disk_bytes'] / 1e9:.1f} GB @ {streaming['disk_gb_per_s']:.2f} GB/s "
            f"({streaming['layers_loaded']} layer loads) | "
            f"to-GPU: {streaming['gpu_bytes'] / 1e9:.1f} GB @ {streaming['gpu_gb_per_s']:.2f} GB/s | "
            f"peak VRAM: {metrics['memory'].get('vram_peak_reserved_gb', '?')} GB | "
            f"RAM: {metrics['memory'].get('process_ram_gb', '?')} GB"
        )

        reasoning = ""
        content = raw
        if apply_think_split:
            thinks = THINK_RE.findall(raw)
            if thinks:
                reasoning = "\n".join(t.strip() for t in thinks)
                content = THINK_RE.sub("", raw).strip()
            elif "</think>" in raw:  # template opened the think block for the model
                reasoning, _, content = raw.rpartition("</think>")
                reasoning = reasoning.replace("<think>", "").strip()
                content = content.strip()
        if not content.strip():
            # Never return empty content: budget exhausted mid-reasoning; surface the tail.
            content = (reasoning[-2000:] if reasoning else raw).strip() or "(no output)"
        return content, reasoning, n_in, n_out, secs, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="safetensors checkpoint dir; default: first one found under models\\")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--delete-original", action="store_true",
                    help="delete the unsplit checkpoint after splitting to save ~56GB disk")
    args = ap.parse_args()

    if not args.model:
        candidates = sorted(d for d in (ROOT / "models").glob("*/")
                            if any(d.glob("*.safetensors")))
        if not candidates:
            ap.error("no safetensors checkpoint found under models\\ — pass --model")
        args.model = str(candidates[0])

    Handler.model = build_model(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve_airllm] listening on http://{args.host}:{args.port} (OpenAI-compatible /v1)")
    server.serve_forever()


if __name__ == "__main__":
    main()
