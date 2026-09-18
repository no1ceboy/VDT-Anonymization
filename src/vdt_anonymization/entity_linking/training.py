"""Train a Siamese encoder + pair-feature MLP for within-document entity linking."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from itertools import islice
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - transformers normally brings tqdm along
    class _FallbackProgress:
        def __init__(self, iterable, total, desc, **_kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.count = 0
            self.postfix = ""

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.count += 1
                self.refresh()
            print(flush=True)

        def set_postfix(self, values, **_kwargs):
            self.postfix = " ".join(f"{key}={value}" for key, value in values.items())

        def refresh(self):
            print(f"\r{self.desc}: {self.count}/{self.total} {self.postfix}", end="", flush=True)

        def close(self):
            pass

    def tqdm(iterable, total=None, desc="", **kwargs):
        return _FallbackProgress(iterable, total, desc, **kwargs)

from .dataset import MENTION_CLOSE, MENTION_OPEN, PAIR_FEATURE_NAMES
from .raw_dataset import RAW_PAIR_FEATURE_NAMES
from .finetuning import (
    ADAPTER_MODES,
    FINETUNE_MODES,
    adapter_checkpoint,
    build_encoder,
    compute_dtype,
    normalize_finetune_mode,
)


class JsonlPairDataset(Dataset):
    """Random-access JSONL dataset without retaining all contexts in memory."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.offsets = []
        self._handle = None
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict:
        if self._handle is None:
            self._handle = self.path.open("rb")
        self._handle.seek(self.offsets[index])
        return json.loads(self._handle.readline())

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __del__(self):
        if self._handle is not None:
            self._handle.close()


class PairCollator:
    def __init__(self, tokenizer, max_length: int, feature_names: tuple[str, ...]):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feature_names = feature_names

    def __call__(self, rows: list[dict]) -> dict:
        contexts_a = [row["mention_a"]["context"] for row in rows]
        contexts_b = [row["mention_b"]["context"] for row in rows]
        encoded_a = self.tokenizer(
            contexts_a, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        encoded_b = self.tokenizer(
            contexts_b, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        features = torch.tensor(
            [[float(row.get("features", {}).get(name, 0.0)) for name in self.feature_names] for row in rows],
            dtype=torch.float32,
        )
        labels = torch.tensor([float(row["target_linked"]) for row in rows], dtype=torch.float32)
        return {"a": encoded_a, "b": encoded_b, "features": features, "labels": labels}


class EntityLinkingModel(nn.Module):
    def __init__(self, encoder: nn.Module, feature_count: int, dropout: float = 0.15):
        super().__init__()
        self.encoder = encoder
        config = encoder.config
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None and hasattr(encoder, "base_model"):
            hidden_size = encoder.base_model.config.hidden_size
        if hidden_size is None:
            raise ValueError("encoder config has no hidden_size")
        hidden_size = int(hidden_size)
        pair_size = hidden_size * 4 + feature_count
        self.classifier = nn.Sequential(
            nn.LayerNorm(pair_size),
            nn.Dropout(dropout),
            nn.Linear(pair_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    @staticmethod
    def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        result = self.encoder(**batch)
        return self.mean_pool(result.last_hidden_state, batch["attention_mask"])

    def forward(self, a: dict[str, torch.Tensor], b: dict[str, torch.Tensor],
                features: torch.Tensor) -> torch.Tensor:
        embedding_a, embedding_b = self.encode(a), self.encode(b)
        pair = torch.cat(
            [embedding_a, embedding_b, torch.abs(embedding_a - embedding_b), embedding_a * embedding_b, features],
            dim=-1,
        )
        return self.classifier(pair).squeeze(-1)


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for target, score in zip(labels, scores):
        prediction = score >= threshold
        if target and prediction:
            tp += 1
        elif target:
            fn += 1
        elif prediction:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    total = tp + fp + tn + fn
    return {
        "threshold": threshold,
        "examples": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def best_threshold(labels: list[int], scores: list[float]) -> tuple[float, dict]:
    candidates = [index / 100 for index in range(5, 96, 5)]
    ranked = [(binary_metrics(labels, scores, threshold), threshold) for threshold in candidates]
    result, threshold = max(ranked, key=lambda item: (item[0]["f1"], item[0]["accuracy"], -abs(item[1] - 0.5)))
    return threshold, result


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "a": {key: value.to(device) for key, value in batch["a"].items()},
        "b": {key: value.to(device) for key, value in batch["b"].items()},
        "features": batch["features"].to(device),
        "labels": batch["labels"].to(device),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             fp16: bool = False, amp_dtype: torch.dtype = torch.float16,
             max_steps: int | None = None, desc: str = "Evaluation",
             show_progress: bool = True) -> tuple[list[int], list[float], float]:
    model.eval()
    labels, scores, loss_sum = [], [], 0.0
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    total = min(len(loader), max_steps) if max_steps is not None else len(loader)
    progress = tqdm(islice(loader, total), total=total, desc=desc, unit="batch", disable=not show_progress)
    for step, raw_batch in enumerate(progress, 1):
        batch = move_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=fp16 and device.type == "cuda"):
            logits = model(batch["a"], batch["b"], batch["features"])
        batch_loss = criterion(logits, batch["labels"])
        loss_sum += float(batch_loss.item())
        labels.extend(batch["labels"].int().cpu().tolist())
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        if show_progress:
            progress.set_postfix({"loss": f"{float(batch_loss.mean().item()):.4f}"})
    if show_progress:
        progress.close()
    return labels, scores, loss_sum / max(1, len(labels))


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_payload(model: nn.Module, optimizer, scheduler, epoch: int,
                        history: list[dict], config: dict, mode: str, adapter_dir: str | None) -> dict:
    payload = {
        "epoch": epoch,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": history,
        "config": config,
    }
    if mode in ADAPTER_MODES:
        payload["adapter_dir"] = adapter_dir
        payload["classifier_state"] = model.classifier.state_dict()
    else:
        payload["model_state"] = model.state_dict()
    return payload


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, epoch: int,
                    history: list[dict], config: dict, mode: str) -> None:
    adapter_dir = None
    if mode in ADAPTER_MODES:
        adapter_dir = "last_adapter"
        adapter_checkpoint(model, path.parent / adapter_dir)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(_checkpoint_payload(model, optimizer, scheduler, epoch, history, config, mode, adapter_dir), temporary)
    os.replace(temporary, path)


def _load_checkpoint_weights(model: EntityLinkingModel, checkpoint: dict, mode: str) -> None:
    if mode in ADAPTER_MODES:
        model.classifier.load_state_dict(checkpoint["classifier_state"])
    else:
        model.load_state_dict(checkpoint["model_state"])


def _memory_stats(device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    return {
        "peak_memory_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
        "peak_memory_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 1024**3, 3),
    }


def train(args: argparse.Namespace) -> dict:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError("TensorBoard is required; install requirements.txt before training") from error

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    mode = normalize_finetune_mode(args.finetune_mode, args.freeze_encoder)
    amp_dtype = compute_dtype(args.amp_dtype)
    qlora_dtype = compute_dtype(args.qlora_compute_dtype) if args.qlora_compute_dtype else amp_dtype
    if args.feature_set == "embeddings_only":
        feature_names = tuple()
    elif args.feature_set == "context":
        feature_names = ("same_label", "same_role", "character_distance_log_scaled")
    elif args.feature_set == "raw":
        feature_names = RAW_PAIR_FEATURE_NAMES
    else:
        feature_names = PAIR_FEATURE_NAMES

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = None
    adapter_path = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_config = resume_checkpoint.get("config") or {}
        mode = normalize_finetune_mode(resume_config.get("finetune_mode", mode))
        if mode in ADAPTER_MODES:
            adapter_path = args.resume.parent / resume_checkpoint.get("adapter_dir", "last_adapter")
            if not adapter_path.is_dir():
                raise ValueError(f"adapter checkpoint directory is missing: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer has no pad or eos token; provide a tokenizer with padding support")
        tokenizer.pad_token = tokenizer.eos_token
    # QLoRA keeps the base vocabulary immutable, avoiding trainable random rows in a 4-bit model.
    add_mention_tokens = mode != "qlora"
    added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": [MENTION_OPEN, MENTION_CLOSE]}) \
        if add_mention_tokens else 0
    encoder, finetune_metadata = build_encoder(
        args.model_name,
        mode,
        device,
        tokenizer_size=len(tokenizer) if added_tokens else None,
        add_tokens=bool(added_tokens),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        adapter_path=adapter_path,
        qlora_compute_dtype=qlora_dtype,
    )
    model = EntityLinkingModel(encoder, len(feature_names), args.dropout)
    if mode == "qlora":
        model.classifier.to(device)
    else:
        model.to(device)

    collator = PairCollator(tokenizer, args.max_length, feature_names)
    datasets = {
        "train": JsonlPairDataset(args.train_pairs),
        "validation": JsonlPairDataset(args.validation_pairs),
        "test": JsonlPairDataset(args.test_pairs),
    }
    loaders = {
        name: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=name == "train",
            num_workers=args.num_workers, collate_fn=collator,
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }
    if not all(len(dataset) for dataset in datasets.values()):
        raise ValueError("train, validation, and test pair files must all be non-empty")

    encoder_parameters = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    head_parameters = list(model.classifier.parameters())
    parameter_groups = []
    if encoder_parameters:
        parameter_groups.append({"params": encoder_parameters, "lr": args.encoder_learning_rate, "name": "encoder"})
    parameter_groups.append({"params": head_parameters, "lr": args.head_learning_rate, "name": "head"})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    batches_per_epoch = min(len(loaders["train"]), args.max_train_steps or len(loaders["train"]))
    total_steps = max(1, math.ceil(batches_per_epoch / args.gradient_accumulation) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(1, warmup_steps))
        if step < warmup_steps else max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.fp16 and device.type == "cuda" and amp_dtype == torch.float16
    )
    criterion = nn.BCEWithLogitsLoss()
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    config = {
        "architecture": "shared_encoder_mean_pool_[a,b,abs(a-b),a*b,features]_mlp",
        "model_name": args.model_name,
        "feature_set": args.feature_set,
        "feature_names": list(feature_names),
        "max_length": args.max_length,
        "finetune_mode": mode,
        "lora_target_modules": finetune_metadata.get("lora_target_modules", []),
        "weak_supervision_warning": "Training targets originate from the rule-based reconstruction linker.",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    save_json(args.output_dir / "config.json", config)

    history, start_epoch = [], 1
    if resume_checkpoint is not None:
        _load_checkpoint_weights(model, resume_checkpoint, mode)
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        history = resume_checkpoint.get("history", [])
        start_epoch = int(resume_checkpoint["epoch"]) + 1

    writer = SummaryWriter(log_dir=str(args.output_dir / "tensorboard"), purge_step=start_epoch)
    best_f1 = max((epoch.get("validation", {}).get("f1", -1.0) for epoch in history), default=-1.0)
    best_model_path = args.output_dir / "best_model.pt"
    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, examples = 0.0, 0
        progress = tqdm(
            islice(loaders["train"], batches_per_epoch), total=batches_per_epoch,
            desc=f"Epoch {epoch}/{args.epochs}", unit="batch", disable=not args.progress,
        )
        for step, raw_batch in enumerate(progress, 1):
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=args.fp16 and device.type == "cuda"):
                logits = model(batch["a"], batch["b"], batch["features"])
                loss = criterion(logits, batch["labels"]) / args.gradient_accumulation
            unscaled_loss = float(loss.item()) * args.gradient_accumulation
            scaler.scale(loss).backward()
            running_loss += unscaled_loss * len(batch["labels"])
            examples += len(batch["labels"])
            global_step = (epoch - 1) * batches_per_epoch + step
            if step == 1 or step % args.log_every == 0:
                writer.add_scalar("loss/train_step", unscaled_loss, global_step)
                for group in optimizer.param_groups:
                    writer.add_scalar(f"learning_rate/{group['name']}_step", group["lr"], global_step)
            if args.progress:
                progress.set_postfix({"loss": f"{unscaled_loss:.4f}", "lr": f"{optimizer.param_groups[-1]['lr']:.2e}"})
            if step % args.gradient_accumulation == 0 or step == batches_per_epoch:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        val_labels, val_scores, val_loss = evaluate(
            model, loaders["validation"], device, args.fp16, amp_dtype, args.max_eval_steps,
            desc=f"Validation {epoch}/{args.epochs}", show_progress=args.progress,
        )
        threshold, val_metrics = best_threshold(val_labels, val_scores)
        memory = _memory_stats(device)
        epoch_result = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, examples),
            "validation_loss": val_loss,
            "validation": val_metrics,
            "finetune_mode": mode,
            "train_batches": batches_per_epoch,
            **memory,
        }
        history.append(epoch_result)
        writer.add_scalar("loss/train", epoch_result["train_loss"], epoch)
        writer.add_scalar("loss/validation", val_loss, epoch)
        for metric_name in ("accuracy", "precision", "recall", "f1", "threshold"):
            writer.add_scalar(f"validation/{metric_name}", val_metrics[metric_name], epoch)
        for key, value in memory.items():
            writer.add_scalar(f"hardware/{key}", value, epoch)
        for group in optimizer.param_groups:
            writer.add_scalar(f"learning_rate/{group['name']}", group["lr"], epoch)
        writer.flush()
        save_json(args.output_dir / "training_history.json", {"epochs": history})
        save_checkpoint(args.output_dir / "last_checkpoint.pt", model, optimizer, scheduler, epoch, history, config, mode)
        if val_metrics["f1"] > best_f1 or not best_model_path.exists():
            best_f1 = val_metrics["f1"]
            best_adapter_dir = None
            if mode in ADAPTER_MODES:
                best_adapter_dir = "best_adapter"
                adapter_checkpoint(model, args.output_dir / best_adapter_dir)
            payload = {"threshold": threshold, "config": config, "epoch": epoch, "adapter_dir": best_adapter_dir}
            if mode in ADAPTER_MODES:
                payload["classifier_state"] = model.classifier.state_dict()
            else:
                payload["model_state"] = model.state_dict()
            temporary = best_model_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, best_model_path)
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)

    best = torch.load(best_model_path, map_location="cpu", weights_only=False)
    _load_checkpoint_weights(model, best, mode)
    test_labels, test_scores, test_loss = evaluate(
        model, loaders["test"], device, args.fp16, amp_dtype, args.max_eval_steps,
        desc="Test", show_progress=args.progress,
    )
    test_metrics = binary_metrics(test_labels, test_scores, float(best["threshold"]))
    result = {"best_epoch": best["epoch"], "validation_selected_threshold": best["threshold"],
              "test_loss": test_loss, "test": test_metrics, "finetune_mode": mode, **_memory_stats(device)}
    writer.add_scalar("loss/test", test_loss, int(best["epoch"]))
    for metric_name in ("accuracy", "precision", "recall", "f1"):
        writer.add_scalar(f"test/{metric_name}", test_metrics[metric_name], int(best["epoch"]))
    writer.flush()
    writer.close()
    save_json(args.output_dir / "test_metrics.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--validation-pairs", type=Path, required=True)
    parser.add_argument("--test-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="NlpHUST/ner-vietnamese-electra-base")
    parser.add_argument("--feature-set", choices=["raw", "context", "all", "embeddings_only"], default="raw",
                        help="raw is the production-direction profile; context/all are legacy profiles")
    parser.add_argument("--finetune-mode", choices=FINETUNE_MODES, default="fft",
                        help="fft=full fine-tuning; frozen=head only; lora/qlora=PEFT adapters")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="auto",
                        help="comma-separated module suffixes, or auto for query/key/value or q_proj/k_proj/v_proj")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-steps", type=int,
                        help="maximum train batches per epoch; useful for a short pilot")
    parser.add_argument("--max-eval-steps", type=int,
                        help="maximum validation/test batches per evaluation; useful for a short pilot")
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Keep 0 on Windows; increase only after verifying worker stability")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Write batch loss and learning rates to TensorBoard every N batches")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                        help="Show live train/validation/test progress bars")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16",
                        help="B200 recommendation: bf16; --no-fp16 disables autocast entirely")
    parser.add_argument("--qlora-compute-dtype", choices=["fp16", "bf16"],
                        help="4-bit matmul dtype; defaults to --amp-dtype")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="deprecated alias for --finetune-mode frozen")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    try:
        normalize_finetune_mode(args.finetune_mode, args.freeze_encoder)
    except ValueError as error:
        parser.error(str(error))
    for field in ("train_pairs", "validation_pairs", "test_pairs"):
        value = getattr(args, field).resolve()
        if not value.is_file():
            parser.error(f"{field.replace('_', '-')} does not exist: {value}")
        setattr(args, field, value)
    if args.resume:
        args.resume = args.resume.resolve()
        if not args.resume.is_file():
            parser.error(f"resume checkpoint does not exist: {args.resume}")
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        parser.error("epochs, batch size, and gradient accumulation must be positive")
    if args.max_length < 16 or args.log_every < 1 or args.num_workers < 0:
        parser.error("max length must be >=16; log every >=1; num workers >=0")
    if args.max_train_steps is not None and args.max_train_steps < 1:
        parser.error("max train steps must be positive")
    if args.max_eval_steps is not None and args.max_eval_steps < 1:
        parser.error("max eval steps must be positive")
    if args.lora_r < 1 or args.lora_alpha < 1 or not 0 <= args.lora_dropout < 1:
        parser.error("LoRA rank and alpha must be positive; dropout must be in [0, 1)")
    args.output_dir = args.output_dir.resolve()
    return args


def main() -> None:
    result = train(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
