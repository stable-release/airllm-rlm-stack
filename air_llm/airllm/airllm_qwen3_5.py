from .airllm_base import AirLLMBaseModel


class AirLLMQwen3_5(AirLLMBaseModel):
    """Qwen3.5-family hybrid VLM (``Qwen3_5ForConditionalGeneration``).

    The decoder is nested one level down (``model.language_model``) with the lm_head at the top
    level, alongside a 27-block vision tower (``model.visual``) and a one-layer MTP speculative
    head (``mtp``). The 64 text layers alternate linear attention (Gated DeltaNet) with full
    attention every 4th layer, but transformers owns the forward pass, so the generic per-layer
    streaming applies unchanged.

    The vision tower is kept resident (under 1 GB) so multimodal inputs work; the MTP head is
    left out entirely — plain ``generate()`` never runs it, and streaming its shard would only
    waste disk and load time.
    """

    def get_auto_class(self):
        # AutoModelForCausalLM maps qwen3_5 to the bare text model (module paths model.layers.*),
        # but the checkpoint is saved from the VLM wrapper (model.language_model.layers.*), so
        # build the wrapper to keep module paths and tensor names aligned.
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'model.language_model.embed_tokens',
            'layer_prefix': 'model.language_model.layers',
            'norm': 'model.language_model.norm',
            'lm_head': 'lm_head',
            'resident': [
                'model.visual',
            ],
        }
