import importlib
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from airllm import AirLLMLlamaMlx

# Architectures that need a dedicated AirLLM subclass because of a non-standard module layout
# (custom remote-code models). Everything else uses the generic AirLLMBaseModel, which streams any
# standard *ForCausalLM (model.model.layers + lm_head / norm) and lets transformers own the
# forward pass, so newly released architectures work without code changes.
ARCH_OVERRIDES = {
    "ChatGLMModel": "AirLLMChatGLM",
    "ChatGLMForConditionalGeneration": "AirLLMChatGLM",
    "QWenLMHeadModel": "AirLLMQWen",
    "BaichuanForCausalLM": "AirLLMBaichuan",
    "BaiChuanForCausalLM": "AirLLMBaichuan",
    "InternLMForCausalLM": "AirLLMInternLM",
    "KimiK3ForConditionalGeneration": "AirLLMKimiK3",
    "Qwen3_5ForConditionalGeneration": "AirLLMQwen3_5",
}


class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        # Imported lazily so the GGUF backend (stdlib-only) works without transformers installed.
        from transformers import AutoConfig
        if 'hf_token' in kwargs:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True,
                                                token=kwargs['hf_token'])
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        architectures = getattr(config, "architectures", None) or []
        arch = architectures[0] if architectures else ""

        cls_name = ARCH_OVERRIDES.get(arch)
        if cls_name is None:
            print(f"using generic AirLLM streaming model for architecture: {arch or 'unknown'}")
            cls_name = "AirLLMBaseModel"
        return "airllm", cls_name

    @classmethod
    def _is_gguf_path(cls, path):
        from pathlib import Path
        p = Path(str(path))
        if p.suffix.lower() == ".gguf":
            return True
        return p.is_dir() and any(p.glob("*.gguf"))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        # GGUF container files are not safetensors checkpoints and cannot be layer-streamed
        # by the native AirLLM loader; route them to the llama.cpp-backed GGUF backend.
        if cls._is_gguf_path(pretrained_model_name_or_path):
            from .airllm_gguf import AirLLMGGUF
            return AirLLMGGUF(pretrained_model_name_or_path, *inputs, **kwargs)

        if is_on_mac_os:
            return AirLLMLlamaMlx(pretrained_model_name_or_path, *inputs, **kwargs)

        module, class_name = AutoModel.get_module_class(pretrained_model_name_or_path, *inputs, **kwargs)
        module = importlib.import_module(module)
        class_ = getattr(module, class_name)
        return class_(pretrained_model_name_or_path, *inputs, **kwargs)
