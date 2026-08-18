from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from .airllm_llama_mlx import AirLLMLlamaMlx
    from .auto_model import AutoModel
else:
    # Core entry points. AutoModel and the GGUF backend are stdlib-only, so a plain
    # `import airllm` always works even without torch/transformers; the safetensors
    # streaming path additionally needs the ML stack and degrades gracefully without it.
    from .auto_model import AutoModel
    from .airllm_gguf import AirLLMGGUF

    try:
        from .airllm_base import AirLLMBaseModel
        from .utils import split_and_save_layers
        from .utils import NotEnoughSpaceException
    except ImportError as _e:
        import warnings as _warnings
        _warnings.warn(
            f"airllm: safetensors streaming backend unavailable ({_e}). "
            f"GGUF models still work; install torch/transformers/accelerate for safetensors models."
        )

    # Dedicated subclasses for a handful of custom-architecture models. Some of them pull in
    # optional extras (e.g. the Baichuan tokenizer needs `sentencepiece`). Import them defensively
    # so a missing optional dependency for one niche family never breaks the whole package; the
    # generic AirLLMBaseModel path keeps working regardless.
    import warnings as _warnings

    for _name, _module in (
        ("AirLLMLlama2", ".airllm"),
        ("AirLLMChatGLM", ".airllm_chatglm"),
        ("AirLLMQWen", ".airllm_qwen"),
        ("AirLLMQWen2", ".airllm_qwen2"),
        ("AirLLMBaichuan", ".airllm_baichuan"),
        ("AirLLMInternLM", ".airllm_internlm"),
        ("AirLLMMistral", ".airllm_mistral"),
        ("AirLLMMixtral", ".airllm_mixtral"),
        ("AirLLMKimiK3", ".airllm_kimi_k3"),
        ("AirLLMQwen3_5", ".airllm_qwen3_5"),
    ):
        try:
            _mod = __import__(__name__ + _module, fromlist=[_name])
            globals()[_name] = getattr(_mod, _name)
        except Exception as _e:  # noqa: BLE001 - optional family, keep package importable
            _warnings.warn(
                f"airllm: optional model class {_name} is unavailable ({_e}). "
                f"This only affects that specific model family; the generic streaming path still works."
            )

