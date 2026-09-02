from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from safetensors import safe_open
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    DynamicCache,
)

from nanovllm.sampling_params import SamplingParams


@dataclass(slots=True)
class _CacheEntry:
    past_key_values: DynamicCache
    attention_mask: torch.Tensor
    rope_deltas: torch.Tensor | None


class TransformersEngine:
    """Compatibility backend for state-space/hybrid models.

    This backend intentionally favors correctness over nano-vLLM's paged-KV
    optimizations. ``DynamicCache`` owns both full-attention K/V tensors and
    the convolution/recurrent state used by Qwen3.5 linear-attention layers.

    Text-only mode loads just the language model from a multimodal checkpoint.
    Multimodal mode loads the complete model and accepts processor-style chat
    messages containing text, images, or videos.
    """

    _SUPPORTED_MODEL_TYPES = {"qwen3_5", "qwen3_5_text"}

    def __init__(
        self,
        model: str,
        *,
        hf_config,
        tokenizer: str | None = None,
        device: str | None = None,
        dtype: str | torch.dtype | None = None,
        attn_implementation: str = "sdpa",
        tensor_parallel_size: int = 1,
        multimodal: bool = False,
        **kwargs,
    ):
        unsupported = set(kwargs) - {
            "enforce_eager",
            "gpu_memory_utilization",
            "max_model_len",
            "max_num_batched_tokens",
            "max_num_seqs",
            "kvcache_block_size",
        }
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(f"Unsupported Transformers backend arguments: {names}")
        if hf_config.model_type not in self._SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Transformers backend does not support model_type={hf_config.model_type!r}")
        if tensor_parallel_size != 1:
            raise ValueError("Qwen3.5 Transformers backend currently requires tensor_parallel_size=1")
        if multimodal and hf_config.model_type != "qwen3_5":
            raise ValueError("multimodal=True requires a Qwen3.5 multimodal checkpoint")

        self.model_path = model
        self.hf_config = hf_config
        self.text_config = hf_config.text_config if hf_config.model_type == "qwen3_5" else hf_config
        self.multimodal = multimodal
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = self._resolve_dtype(dtype)
        self.processor = None
        if multimodal:
            if tokenizer is not None and tokenizer != model:
                raise ValueError("A separate tokenizer is not supported in multimodal mode; use the model processor")
            self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer or model, use_fast=True, trust_remote_code=True)

        if self.device == "cpu":
            self._enable_torch_fallbacks()

        # Keep model imports lazy: on CPU, Transformers must select its torch
        # fallbacks before probing optional CUDA-only causal-conv/FLA modules.
        from transformers import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

        common_load_args = {
            "dtype": self.dtype,
            "device_map": {"": self.device},
            "low_cpu_mem_usage": True,
            "attn_implementation": attn_implementation,
        }
        if multimodal:
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model,
                config=hf_config,
                **common_load_args,
            )
        else:
            self.model = Qwen3_5ForCausalLM.from_pretrained(
                model,
                config=self.text_config,
                key_mapping=self._checkpoint_key_mapping(model),
                **common_load_args,
            )
        self.model.eval()
        self._caches: dict[str, _CacheEntry] = {}
        self._closed = False

    def _resolve_dtype(self, dtype: str | torch.dtype | None) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            value = getattr(torch, dtype.removeprefix("torch."), None)
            if not isinstance(value, torch.dtype):
                raise ValueError(f"Unknown dtype: {dtype}")
            return value
        config_dtype = getattr(self.text_config, "dtype", None)
        return config_dtype if isinstance(config_dtype, torch.dtype) else torch.bfloat16

    @staticmethod
    def _enable_torch_fallbacks():
        # FLA's fused RMSNorm constructor assumes a CUDA device. The upstream
        # Transformers implementation has complete torch fallbacks, which are
        # also useful for CI and checkpoint-loading checks on CPU-only hosts.
        from transformers.utils import import_utils

        import_utils.is_causal_conv1d_available = lambda: False
        import_utils.is_flash_linear_attention_available = lambda: False
        from transformers.models.qwen3_5 import modeling_qwen3_5

        modeling_qwen3_5.FusedRMSNormGated = None
        modeling_qwen3_5.causal_conv1d_fn = None
        modeling_qwen3_5.causal_conv1d_update = None
        modeling_qwen3_5.chunk_gated_delta_rule = None
        modeling_qwen3_5.fused_recurrent_gated_delta_rule = None

    @classmethod
    def _checkpoint_key_mapping(cls, model: str) -> dict[str, str] | None:
        """Map a Qwen3.5 multimodal wrapper checkpoint to the text-only model."""
        index_path = os.path.join(model, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as f:
                keys = json.load(f).get("weight_map", {})
            wrapped = any(key.startswith("model.language_model.") for key in keys)
        else:
            safetensors_path = os.path.join(model, "model.safetensors")
            if not os.path.isfile(safetensors_path):
                return None
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                wrapped = any(key.startswith("model.language_model.") for key in f.keys())
        return {r"^model\.language_model": "model"} if wrapped else None

    def _input_device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    def _check_open(self):
        if self._closed:
            raise RuntimeError("The engine has been closed")

    def _move_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        device = self._input_device()
        return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}

    def _apply_processor(self, messages: list[dict[str, Any]], processor_kwargs: dict[str, Any]):
        def encode(conversation, kwargs):
            return self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs=kwargs,
            )

        messages = deepcopy(messages)
        for message in messages:
            if not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if block.get("type") != "video_url":
                    continue
                video_url = block.pop("video_url", None)
                block["type"] = "video"
                block["video"] = video_url.get("url", "") if isinstance(video_url, dict) else video_url

        try:
            return encode(deepcopy(messages), processor_kwargs)
        except RuntimeError as exc:
            if "libtorchcodec" not in str(exc):
                raise

        # TorchCodec can be installed yet unusable because its FFmpeg/PyTorch
        # ABI does not match. Reuse another installed decoder and pass decoded
        # frames to the processor instead of requiring a rebuilt environment.
        from transformers.utils.import_utils import is_av_available, is_decord_available
        from transformers.video_utils import load_video

        if is_decord_available():
            backend = "decord"
        elif is_av_available():
            backend = "pyav"
        else:
            raise RuntimeError("Video decoding requires a working torchcodec, decord, or PyAV installation")

        num_frames = processor_kwargs.get("num_frames")
        fps = None if num_frames is not None else processor_kwargs.get("fps", self.processor.video_processor.fps)
        decoded_metadata = []
        video_count = 0
        for message in messages:
            if not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if block.get("type") != "video":
                    continue
                video_count += 1
                source = next((block[key] for key in ("video", "url", "path") if key in block), None)
                if not isinstance(source, str):
                    continue
                frames, metadata = load_video(source, backend=backend, num_frames=num_frames, fps=fps)
                decoded_metadata.append(metadata)
                for key in ("url", "path"):
                    block.pop(key, None)
                block["video"] = frames

        retry_kwargs = dict(processor_kwargs)
        retry_kwargs.pop("num_frames", None)
        retry_kwargs.pop("fps", None)
        retry_kwargs["do_sample_frames"] = False
        if video_count and len(decoded_metadata) == video_count:
            retry_kwargs["video_metadata"] = decoded_metadata
        return encode(messages, retry_kwargs)

    def _prepare_prompt(self, prompt: str | list[int] | dict[str, Any]) -> dict[str, torch.Tensor]:
        device = self._input_device()
        if isinstance(prompt, str):
            encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            return self._move_inputs(dict(encoded))
        if isinstance(prompt, dict):
            if not self.multimodal:
                raise ValueError("Message/image/video prompts require multimodal=True")
            messages = prompt.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("A multimodal prompt must contain a non-empty 'messages' list")
            processor_kwargs = prompt.get("mm_processor_kwargs") or {}
            if not isinstance(processor_kwargs, dict):
                raise TypeError("mm_processor_kwargs must be a dict")
            encoded = self._apply_processor(messages, processor_kwargs)
            return self._move_inputs(dict(encoded))
        if isinstance(prompt, list) and prompt and all(isinstance(token_id, int) for token_id in prompt):
            input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
            return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        raise TypeError("Each prompt must be text, token IDs, or {'messages': [...]} in multimodal mode")

    def _rope_deltas(self) -> torch.Tensor | None:
        if not self.multimodal:
            return None
        return self.model.model.rope_deltas

    def _set_rope_deltas(self, value: torch.Tensor | None):
        if self.multimodal:
            self.model.model.rope_deltas = value

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str] | list[list[int]] | list[dict[str, Any]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict[str, str | list[int]]]:
        del use_tqdm
        self._check_open()
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(prompts) != len(sampling_params):
            raise ValueError("prompts and sampling_params must have the same length")

        outputs = []
        for prompt, params in zip(prompts, sampling_params):
            self._set_rope_deltas(None)
            inputs = self._prepare_prompt(prompt)
            input_ids = inputs["input_ids"]

            generation_config = deepcopy(self.model.generation_config)
            generation_config.do_sample = True
            generation_config.temperature = params.temperature
            generation_config.max_new_tokens = params.max_tokens
            generation_config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            if params.ignore_eos:
                generation_config.eos_token_id = []

            generated = self.model.generate(
                **inputs,
                generation_config=generation_config,
                use_cache=True,
            )
            token_ids = generated[0, input_ids.shape[1] :].tolist()
            outputs.append(
                {
                    "text": self.tokenizer.decode(token_ids, skip_special_tokens=self.multimodal),
                    "token_ids": token_ids,
                }
            )
        return outputs

    @torch.inference_mode()
    def prefill(self, request_id: str, prompt: str | list[int] | dict[str, Any]) -> torch.Tensor:
        """Create a request cache and return next-token logits.

        A request cache is batch-size one. For multimodal requests, image/video
        tensors are consumed only here; subsequent ``decode`` calls reuse the
        cached language-model state.
        """
        self._check_open()
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if request_id in self._caches:
            raise ValueError(f"request_id already exists: {request_id!r}")

        self._set_rope_deltas(None)
        inputs = self._prepare_prompt(prompt)
        if inputs["input_ids"].shape[0] != 1:
            raise ValueError("Explicit request caches currently support batch size 1")
        cache = DynamicCache(config=self.text_config)
        outputs = self.model(
            **inputs,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs["input_ids"])
        self._caches[request_id] = _CacheEntry(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
            rope_deltas=self._rope_deltas(),
        )
        return outputs.logits[:, -1, :]

    @torch.inference_mode()
    def decode(self, request_id: str, token_ids: int | list[int] | torch.Tensor) -> torch.Tensor:
        """Append token IDs to an existing request cache and return next logits."""
        self._check_open()
        try:
            entry = self._caches[request_id]
        except KeyError as exc:
            raise KeyError(f"Unknown request_id: {request_id!r}") from exc

        device = self._input_device()
        if isinstance(token_ids, int):
            input_ids = torch.tensor([[token_ids]], dtype=torch.long, device=device)
        elif isinstance(token_ids, list):
            if not token_ids or not all(isinstance(token_id, int) for token_id in token_ids):
                raise TypeError("token_ids must be a non-empty int list")
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        elif isinstance(token_ids, torch.Tensor):
            input_ids = token_ids.to(device=device, dtype=torch.long)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
                raise ValueError("token_ids tensor must have shape [tokens] or [1, tokens]")
        else:
            raise TypeError("token_ids must be an int, int list, or tensor")

        new_mask = torch.ones((1, input_ids.shape[1]), dtype=entry.attention_mask.dtype, device=device)
        entry.attention_mask = torch.cat((entry.attention_mask, new_mask), dim=-1)
        self._set_rope_deltas(entry.rope_deltas)
        position_ids = None
        if self.multimodal:
            position_ids = self.model._prepare_position_ids_for_generation(
                input_ids,
                {
                    "attention_mask": entry.attention_mask,
                    "past_key_values": entry.past_key_values,
                },
            )
            # GenerationMixin normally performs this slicing in
            # prepare_inputs_for_generation. Direct forward needs it here.
            position_ids = position_ids[..., -input_ids.shape[1] :]
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=entry.attention_mask,
            position_ids=position_ids,
            past_key_values=entry.past_key_values,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        entry.past_key_values = outputs.past_key_values
        entry.rope_deltas = self._rope_deltas()
        return outputs.logits[:, -1, :]

    def release(self, request_id: str) -> bool:
        """Release one request's K/V and linear-attention state."""
        entry = self._caches.pop(request_id, None)
        released = entry is not None
        del entry
        return released

    def clear_cache(self):
        """Release all explicit request caches."""
        self._caches.clear()
        self._set_rope_deltas(None)

    def cache_info(self, request_id: str | None = None) -> dict[str, Any]:
        """Return lightweight cache metadata without exposing cache tensors."""
        if request_id is None:
            return {"num_requests": len(self._caches), "request_ids": list(self._caches)}
        try:
            entry = self._caches[request_id]
        except KeyError as exc:
            raise KeyError(f"Unknown request_id: {request_id!r}") from exc
        return {
            "request_id": request_id,
            "sequence_length": entry.past_key_values.get_seq_length(),
            "layers": len(entry.past_key_values.layers),
            "multimodal": self.multimodal,
        }

    def exit(self):
        if self._closed:
            return
        self.clear_cache()
        self._closed = True
        del self.model
        if self.processor is not None:
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
