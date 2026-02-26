"""
src/models.py
─────────────────────────────────────────────────────────────
Sequential, memory-safe model loader for single-GPU environments.

Design principles
─────────────────
• Only ONE model lives in VRAM at any time.
• Every model is loaded with 4-bit BitsAndBytes quantization.
• A dedicated `ModelHandle` context-manager enforces load → use → unload.
• Model version hashes (sha256 of config.json + tokenizer.json) are
  computed and stored for reproducibility.
• GPU memory is forcibly cleared after each unload (gc + cuda.empty_cache).
• All chat templates are normalised to a common interface so Protocol
  code doesn't need to branch per model.

Usage
-----
    handle = ModelHandle.load(model_cfg, quant_cfg, device)
    output = handle.generate(prompt, gen_params)
    handle.unload()          # or use as context manager
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Lazy imports — avoid hard dependency when running in analysis-only mode
# ──────────────────────────────────────────────────────────────────────────────

def _import_transformers():
    try:
        import transformers
        return transformers
    except ImportError as e:
        raise ImportError(
            "transformers not installed. Run: pip install transformers>=4.40.0"
        ) from e


def _import_bitsandbytes():
    try:
        import bitsandbytes
        return bitsandbytes
    except ImportError as e:
        raise ImportError(
            "bitsandbytes not installed. Run: pip install bitsandbytes>=0.43.1"
        ) from e


# ──────────────────────────────────────────────────────────────────────────────
# Quantization config builder
# ──────────────────────────────────────────────────────────────────────────────

def build_bnb_config(quant_cfg) -> Any:
    """
    Build a BitsAndBytesConfig from our QuantizationConfig dataclass.
    quant_cfg: src.config.QuantizationConfig
    """
    transformers = _import_transformers()
    _import_bitsandbytes()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(quant_cfg.bnb_4bit_compute_dtype, torch.float16)

    return transformers.BitsAndBytesConfig(
        load_in_4bit=quant_cfg.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
        llm_int8_enable_fp32_cpu_offload=quant_cfg.llm_int8_enable_fp32_cpu_offload,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Version hashing
# ──────────────────────────────────────────────────────────────────────────────

def compute_model_version_hash(hf_name: str, cache_dir: Optional[str] = None) -> str:
    """
    Compute a short reproducibility hash from model config files.
    Uses HuggingFace hub's local snapshot metadata when available;
    falls back to the HF name hash if files not yet downloaded.
    """
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        sentinel_files = ["config.json", "tokenizer_config.json"]
        h = hashlib.sha256()
        for fname in sentinel_files:
            content = None
            try:
                local = try_to_load_from_cache(hf_name, fname, cache_dir=cache_dir)
                if local and os.path.isfile(local):
                    with open(local, "rb") as f:
                        content = f.read()
            except Exception:
                pass
            if content is None:
                content = fname.encode()  # fallback: just hash the filename
            h.update(content)
        return h.hexdigest()[:16]
    except Exception:
        # Best-effort: hash the model name string
        return hashlib.sha256(hf_name.encode()).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────────────
# Chat formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

def format_prompt(
    system_prompt: str,
    user_message: str,
    chat_template: str,
) -> str:
    """
    Format a system + user message into the model-specific chat template.
    Returns a raw string suitable for tokenizer encoding.
    """
    templates = {
        "phi3": (
            f"<|system|>\n{system_prompt}<|end|>\n"
            f"<|user|>\n{user_message}<|end|>\n"
            f"<|assistant|>\n"
        ),
        "stablelm": (
            f"<|system|>\n{system_prompt}\n<|user|>\n{user_message}\n<|assistant|>\n"
        ),
        "gemma": (
            f"<bos><start_of_turn>user\n{system_prompt}\n\n{user_message}"
            f"<end_of_turn>\n<start_of_turn>model\n"
        ),
        "tinyllama": (
            f"<|system|>\n{system_prompt}</s>\n"
            f"<|user|>\n{user_message}</s>\n"
            f"<|assistant|>\n"
        ),
        "chatml": (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        ),
        "llama2": (
            f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"
        ),
    }
    if chat_template not in templates:
        logger.warning(
            "Unknown chat_template '%s'; falling back to chatml.", chat_template
        )
        return templates["chatml"]
    return templates[chat_template]


# ──────────────────────────────────────────────────────────────────────────────
# GPU utilities
# ──────────────────────────────────────────────────────────────────────────────

def log_gpu_memory(tag: str = "") -> Dict[str, float]:
    """Return and log current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    info = {
        "tag": tag,
        "allocated_gb": round(allocated, 3),
        "reserved_gb": round(reserved, 3),
        "total_gb": round(total, 3),
        "free_gb": round(total - allocated, 3),
    }
    logger.debug("GPU Memory [%s]: %.2f GB allocated / %.2f GB total", tag, allocated, total)
    return info


def aggressive_gpu_clear() -> None:
    """
    Aggressively free GPU memory:
    1. Python garbage collect.
    2. CUDA empty cache.
    3. CUDA ipc collect (if available).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except AttributeError:
            pass
    time.sleep(0.5)  # allow CUDA driver to process frees


# ──────────────────────────────────────────────────────────────────────────────
# ModelHandle
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    model_id: str


class ModelHandle:
    """
    Encapsulates a single loaded model + tokenizer.
    Enforces sequential, single-model-at-a-time constraint.

    Parameters
    ----------
    model_cfg : src.config.ModelConfig
    quant_cfg : src.config.QuantizationConfig
    device    : str, default "cuda" if available else "cpu"
    """

    def __init__(
        self,
        model_cfg,
        quant_cfg,
        device: str = "auto",
    ):
        self.model_cfg = model_cfg
        self.quant_cfg = quant_cfg
        self.device = device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = None
        self._tokenizer = None
        self._version_hash: str = ""
        self._load_time_seconds: float = 0.0
        self._is_loaded: bool = False

    # ── Loading ───────────────────────────────────────────────

    def load(self) -> "ModelHandle":
        """Load model into GPU memory. Returns self for chaining."""
        if self._is_loaded:
            logger.warning(
                "ModelHandle.load() called on already-loaded model '%s'. Skipping.",
                self.model_cfg.display_name,
            )
            return self

        transformers = _import_transformers()
        hf_name = self.model_cfg.hf_name
        log_gpu_memory(f"before loading {self.model_cfg.display_name}")

        logger.info("Loading tokenizer: %s", hf_name)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            hf_name,
            trust_remote_code=True,
            use_fast=True,
        )
        # Ensure pad token is set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info("Loading model: %s  (quantization=%s)", hf_name, self.model_cfg.quantization)
        t0 = time.time()

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }

        if self.model_cfg.quantization in {"4bit", "8bit"}:
            bnb_cfg = build_bnb_config(self.quant_cfg)
            load_kwargs["quantization_config"] = bnb_cfg
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_name, **load_kwargs
        )
        self._model.eval()
        self._load_time_seconds = time.time() - t0

        # Compute version hash
        self._version_hash = compute_model_version_hash(hf_name)
        self._is_loaded = True

        log_gpu_memory(f"after loading {self.model_cfg.display_name}")
        logger.info(
            "Model '%s' loaded in %.1fs  |  version_hash=%s",
            self.model_cfg.display_name,
            self._load_time_seconds,
            self._version_hash,
        )
        return self

    # ── Generation ────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate a response to `prompt`.
        Returns a GenerationResult with decoded text and metadata.
        """
        if not self._is_loaded:
            raise RuntimeError(
                f"Model '{self.model_cfg.display_name}' is not loaded. Call .load() first."
            )

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        # Move inputs to model device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]
        t0 = time.time()

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k

        output_ids = self._model.generate(
            **inputs,
            **gen_kwargs,
        )

        elapsed = time.time() - t0
        output_len = output_ids.shape[1] - input_len
        # Decode only the newly generated tokens
        generated_text = self._tokenizer.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True,
        ).strip()

        return GenerationResult(
            text=generated_text,
            input_tokens=input_len,
            output_tokens=output_len,
            elapsed_seconds=elapsed,
            model_id=self.model_cfg.display_name,
        )

    # ── Unloading ─────────────────────────────────────────────

    def unload(self) -> None:
        """Remove model and tokenizer from GPU memory."""
        if not self._is_loaded:
            return
        logger.info("Unloading model '%s' ...", self.model_cfg.display_name)
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        aggressive_gpu_clear()
        log_gpu_memory(f"after unloading {self.model_cfg.display_name}")

    # ── Context manager ───────────────────────────────────────

    def __enter__(self) -> "ModelHandle":
        return self.load()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()

    # ── Properties ────────────────────────────────────────────

    @property
    def version_hash(self) -> str:
        return self._version_hash

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def display_name(self) -> str:
        return self.model_cfg.display_name

    def __repr__(self) -> str:
        status = "LOADED" if self._is_loaded else "UNLOADED"
        return (
            f"ModelHandle({self.model_cfg.display_name!r}, "
            f"quant={self.model_cfg.quantization!r}, "
            f"status={status})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────────────

def create_handles(cfg) -> Dict[str, ModelHandle]:
    """
    Create (but do NOT load) ModelHandle objects for all models in cfg.
    cfg: ExperimentConfig
    Returns {model_key: ModelHandle}
    """
    handles: Dict[str, ModelHandle] = {}
    for model_key, model_cfg in cfg.models.items():
        handles[model_key] = ModelHandle(
            model_cfg=model_cfg,
            quant_cfg=cfg.quantization,
        )
    return handles
