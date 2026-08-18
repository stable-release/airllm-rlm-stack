"""GGUF backend for AirLLM.

AirLLM's native layer-streaming loader works on safetensors checkpoints and cannot parse
GGUF container files. This backend gives the same "run a big model on a small GPU" behavior
for GGUF models by driving a llama.cpp server (llama-server.exe), which streams/offloads
layers between disk, RAM and VRAM natively (mmap + `--n-gpu-layers`).

The class keeps the familiar AirLLM entry point:

    from airllm import AutoModel
    model = AutoModel.from_pretrained(r"models\\your-model.gguf")
    out = model.generate("Hello", max_new_tokens=128)

It also exposes an OpenAI-compatible endpoint (``model.base_url``) so external clients
(e.g. the rlm-rs Rust harness in this repo) can share the same running server.

Only the Python standard library is used (no `requests`, no `llama-cpp-python`).
"""

import atexit
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def _default_server_bin():
    env = os.environ.get("AIRLLM_LLAMA_SERVER")
    if env:
        return env
    # <repo_root>/runtime/llama.cpp/llama-server.exe  (this file: <repo_root>/air_llm/airllm/)
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "runtime" / "llama.cpp" / "llama-server.exe"
    if candidate.exists():
        return str(candidate)
    return "llama-server"  # hope it is on PATH


class AirLLMGGUF:
    """Run a GGUF model through a managed llama.cpp server, AirLLM-style API."""

    def __init__(self, model_local_path_or_repo_id, *args, **kwargs):
        self.model_path = Path(str(model_local_path_or_repo_id))
        if self.model_path.is_dir():
            ggufs = sorted(self.model_path.glob("*.gguf"))
            if not ggufs:
                raise FileNotFoundError(f"no .gguf files found in {self.model_path}")
            self.model_path = ggufs[0]
        if not self.model_path.exists():
            raise FileNotFoundError(f"GGUF model not found: {self.model_path}")

        self.server_bin = kwargs.get("server_bin") or _default_server_bin()
        self.host = kwargs.get("host", "127.0.0.1")
        self.port = int(kwargs.get("port", 8090))
        self.ctx_len = int(kwargs.get("ctx_len", 32768))
        # None -> let llama.cpp auto-fit layers to available VRAM
        self.n_gpu_layers = kwargs.get("n_gpu_layers", None)
        self.kv_cache_type = kwargs.get("kv_cache_type", "q8_0")
        self.flash_attn = kwargs.get("flash_attn", True)
        self.extra_server_args = list(kwargs.get("extra_server_args", []))
        self.startup_timeout = float(kwargs.get("startup_timeout", 900))
        self._proc = None

        if kwargs.get("auto_start", True):
            self.start()

    # ------------------------------------------------------------------ server
    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"

    def _health(self, timeout=2.0):
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace")).get("status") == "ok"
        except Exception:
            return False

    def start(self):
        if self._health():
            return  # a server is already up on this port; attach to it
        cmd = [
            self.server_bin,
            "-m", str(self.model_path),
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.ctx_len),
            "--jinja",
        ]
        if self.flash_attn:
            cmd += ["-fa", "on"]
        if self.kv_cache_type:
            cmd += ["--cache-type-k", self.kv_cache_type, "--cache-type-v", self.kv_cache_type]
        if self.n_gpu_layers is not None:
            cmd += ["-ngl", str(self.n_gpu_layers)]
        cmd += self.extra_server_args

        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        atexit.register(self.stop)

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {self._proc.returncode}; "
                    f"cmd was: {' '.join(cmd)}"
                )
            if self._health():
                return
            time.sleep(2.0)
        raise TimeoutError(f"llama-server did not become healthy within {self.startup_timeout}s")

    def stop(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    # --------------------------------------------------------------- inference
    def _post(self, route, payload, timeout=600):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + route, data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"llama-server {route} failed ({e.code}): {body[:2000]}") from e

    def generate(self, prompt, max_new_tokens=512, temperature=0.7, top_p=0.95,
                 stop=None, **kwargs):
        """Raw completion over the prompt text (no chat template)."""
        payload = {
            "prompt": prompt,
            "n_predict": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        if stop:
            payload["stop"] = list(stop)
        payload.update(kwargs)
        return self._post("/completion", payload).get("content", "")

    def chat(self, messages, max_new_tokens=512, temperature=0.7, top_p=0.95, **kwargs):
        """OpenAI-style chat completion; `messages` is a list of {role, content} dicts."""
        payload = {
            "messages": messages,
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        payload.update(kwargs)
        res = self._post("/v1/chat/completions", payload)
        return res["choices"][0]["message"]["content"]
