"""Example: run a local GGUF model through the AirLLM fork.

The GGUF backend starts (or attaches to) a llama.cpp server, so this script and the
rlm-rs Rust harness can share one loaded model on port 8090.
"""

import sys
from pathlib import Path

# Make the in-repo airllm package importable without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "air_llm"))

from airllm import AutoModel  # noqa: E402

# Smallest non-projector GGUF under models\ — drop your model file there.
models_dir = Path(__file__).resolve().parents[1] / "models"
ggufs = sorted((p for p in models_dir.glob("*.gguf") if not p.name.startswith("mmproj")),
               key=lambda p: p.stat().st_size)
if not ggufs:
    raise SystemExit(f"no .gguf found in {models_dir}")

model = AutoModel.from_pretrained(str(ggufs[0]))
print(f"server: {model.base_url} (OpenAI-compatible under /v1)")

# Reasoning models spend tokens on chain-of-thought before the final answer,
# so give the budget headroom.
print(model.chat(
    [{"role": "user", "content": "In one sentence, what does recursive language modeling mean?"}],
    max_new_tokens=1024,
))
