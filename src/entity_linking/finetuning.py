"""Encoder loading and checkpoint helpers for entity-linker fine-tuning modes."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel


FINETUNE_MODES = ("fft", "frozen", "lora", "qlora")
ADAPTER_MODES = ("lora", "qlora")


def normalize_finetune_mode(mode: str, freeze_encoder: bool = False) -> str:
    """Validate the new mode flag while keeping the old flag backward compatible."""
    if mode not in FINETUNE_MODES:
        raise ValueError(f"finetune mode must be one of {FINETUNE_MODES}, got {mode!r}")
    if freeze_encoder:
        if mode not in {"fft", "frozen"}:
            raise ValueError("--freeze-encoder can only be combined with --finetune-mode fft or frozen")
        return "frozen"
    return mode


def compute_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported compute dtype: {name}")


def select_lora_targets(encoder: nn.Module, requested: str = "auto") -> list[str]:
    """Choose attention projection suffixes that PEFT can match across encoder families."""
    if requested and requested != "auto":
        targets = [item.strip() for item in requested.split(",") if item.strip()]
        if not targets:
            raise ValueError("--lora-target-modules must contain at least one module name")
        return targets

    linear_suffixes = {
        name.rsplit(".", 1)[-1]
        for name, module in encoder.named_modules()
        if isinstance(module, nn.Linear) or "linear" in module.__class__.__name__.lower()
    }
    preferred_groups = (
        ("query", "key", "value"),
        ("q_proj", "k_proj", "v_proj"),
    )
    for group in preferred_groups:
        found = [name for name in group if name in linear_suffixes]
        if len(found) == len(group):
            return found
    fallback = sorted(linear_suffixes - {"classifier", "lm_head", "pooler"})
    if not fallback:
        raise ValueError("could not find Linear modules for automatic LoRA target selection")
    return fallback


def _load_peft():
    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as error:
        raise RuntimeError(
            "LoRA/QLoRA requires the optional dependencies; install with "
            '`python -m pip install ".[efficient]"`'
        ) from error
    return LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training


def build_encoder(
    model_name: str,
    mode: str,
    device: torch.device,
    tokenizer_size: int | None = None,
    add_tokens: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "auto",
    adapter_path: Path | None = None,
    qlora_compute_dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, dict]:
    """Build a base, frozen, LoRA, or QLoRA encoder and return metadata."""
    mode = normalize_finetune_mode(mode)
    load_kwargs = {}
    if mode == "qlora":
        if device.type != "cuda":
            raise RuntimeError("QLoRA requires a CUDA device with bitsandbytes support")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError("this Transformers version does not provide BitsAndBytesConfig") from error
        load_kwargs = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=qlora_compute_dtype,
            ),
            "device_map": {"": device.index if device.index is not None else 0},
        }
    base = AutoModel.from_pretrained(model_name, **load_kwargs)
    if add_tokens and tokenizer_size is not None:
        base.resize_token_embeddings(tokenizer_size)

    metadata = {"mode": mode, "lora_target_modules": []}
    if mode == "frozen":
        for parameter in base.parameters():
            parameter.requires_grad = False
    elif mode in ADAPTER_MODES:
        LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training = _load_peft()
        if mode == "qlora":
            base = prepare_model_for_kbit_training(base)
        targets = select_lora_targets(base, lora_target_modules)
        metadata["lora_target_modules"] = targets
        if adapter_path is not None:
            base = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
        else:
            base = get_peft_model(
                base,
                LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=targets,
                    bias="none",
                ),
            )
    return base, metadata


def adapter_checkpoint(model: nn.Module, path: Path) -> None:
    """Atomically-ish replace an adapter directory with PEFT's portable files."""
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        for child in temporary.iterdir():
            if child.is_dir():
                import shutil
                shutil.rmtree(child)
            else:
                child.unlink()
        temporary.rmdir()
    temporary.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(temporary, safe_serialization=True)
    if path.exists():
        import shutil
        shutil.rmtree(path)
    temporary.rename(path)
