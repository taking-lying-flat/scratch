from transformers import AutoConfig

from nanovllm.engine.llm_engine import LLMEngine


class LLM:
    """Offline inference facade.

    Qwen3 keeps using nano-vLLM's native paged-KV engine. Qwen3.5 uses a
    Transformers compatibility engine because its hybrid Gated DeltaNet/full
    attention cache cannot be represented by nano-vLLM's KV-only block cache.
    """

    _TRANSFORMERS_MODEL_TYPES = {"qwen3_5", "qwen3_5_text"}

    def __init__(self, model: str, **kwargs):
        backend = kwargs.pop("backend", "auto")
        if backend not in {"auto", "native", "transformers"}:
            raise ValueError("backend must be one of: auto, native, transformers")

        hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        model_type = hf_config.model_type
        if backend == "auto":
            backend = "transformers" if model_type in self._TRANSFORMERS_MODEL_TYPES else "native"
        if backend == "native" and model_type in self._TRANSFORMERS_MODEL_TYPES:
            raise ValueError(
                "Qwen3.5 uses hybrid Gated DeltaNet/full-attention state and is not supported by "
                "nano-vLLM's native KV-only cache yet. Use backend='transformers' (the default)."
            )

        self.backend = backend
        if backend == "transformers":
            from nanovllm.engine.transformers_engine import TransformersEngine

            self._engine = TransformersEngine(model, hf_config=hf_config, **kwargs)
        else:
            self._engine = LLMEngine(model, **kwargs)

    def generate(self, *args, **kwargs):
        return self._engine.generate(*args, **kwargs)

    def exit(self):
        return self._engine.exit()

    def __getattr__(self, name):
        engine = self.__dict__.get("_engine")
        if engine is None:
            raise AttributeError(name)
        return getattr(engine, name)
