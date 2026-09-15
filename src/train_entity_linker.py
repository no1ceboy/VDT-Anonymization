"""Train a Siamese encoder + pair-feature MLP for within-document entity linking.

This trainer consumes JSONL emitted by entity_linking_data.py.  It supports
safe epoch checkpoints and does not require synthetic reconstructed names.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

try:
    from .entity_linking_data import MENTION_CLOSE, MENTION_OPEN, PAIR_FEATURE_NAMES
except ImportError:
    from entity_linking_data import MENTION_CLOSE, MENTION_OPEN, PAIR_FEATURE_NAMES


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
        hidden_size = int(encoder.config.hidden_size)
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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[float], float]:
    model.eval()
    labels, scores, loss_sum = [], [], 0.0
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        logits = model(batch["a"], batch["b"], batch["features"])
        loss_sum += float(criterion(logits, batch["labels"]).item())
        labels.extend(batch["labels"].int().cpu().tolist())
        scores.extend(torch.sigmoid(logits).cpu().tolist())
    return labels, scores, loss_sum / max(1, len(labels))


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, epoch: int,
                    history: list[dict], config: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": history,
        "config": config,
    }, temporary)
    os.replace(temporary, path)


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
    if args.feature_set == "embeddings_only":
        feature_names = tuple()
    elif args.feature_set == "context":
        # Exclude marker equality so the experiment measures value beyond the rule baseline.
        feature_names = ("same_label", "same_role", "character_distance_log_scaled")
    else:
        feature_names = PAIR_FEATURE_NAMES

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": [MENTION_OPEN, MENTION_CLOSE]})
    encoder = AutoModel.from_pretrained(args.model_name)
    if added_tokens:
        encoder.resize_token_embeddings(len(tokenizer))
    model = EntityLinkingModel(encoder, len(feature_names), args.dropout).to(device)
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

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
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_learning_rate},
            {"params": head_parameters, "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, math.ceil(len(loaders["train"]) / args.gradient_accumulation) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(1, warmup_steps))
        if step < warmup_steps else max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    config = {
        "architecture": "shared_encoder_mean_pool_[a,b,abs(a-b),a*b,features]_mlp",
        "model_name": args.model_name,
        "feature_set": args.feature_set,
        "feature_names": list(feature_names),
        "max_length": args.max_length,
        "freeze_encoder": args.freeze_encoder,
        "weak_supervision_warning": "Training targets originate from the rule-based reconstruction linker.",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    save_json(args.output_dir / "config.json", config)

    history, start_epoch = [], 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint["epoch"]) + 1

    writer = SummaryWriter(log_dir=str(args.output_dir / "tensorboard"), purge_step=start_epoch)
    best_f1 = max((epoch.get("validation", {}).get("f1", -1.0) for epoch in history), default=-1.0)
    best_model_path = args.output_dir / "best_model.pt"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, examples = 0.0, 0
        for step, raw_batch in enumerate(loaders["train"], 1):
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"):
                logits = model(batch["a"], batch["b"], batch["features"])
                loss = criterion(logits, batch["labels"]) / args.gradient_accumulation
            unscaled_loss = float(loss.item()) * args.gradient_accumulation
            scaler.scale(loss).backward()
            running_loss += unscaled_loss * len(batch["labels"])
            examples += len(batch["labels"])
            global_step = (epoch - 1) * len(loaders["train"]) + step
            if step == 1 or step % args.log_every == 0:
                writer.add_scalar("loss/train_step", unscaled_loss, global_step)
                writer.add_scalar("learning_rate/encoder_step", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("learning_rate/head_step", optimizer.param_groups[1]["lr"], global_step)
            if step % args.gradient_accumulation == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        val_labels, val_scores, val_loss = evaluate(model, loaders["validation"], device)
        threshold, val_metrics = best_threshold(val_labels, val_scores)
        epoch_result = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, examples),
            "validation_loss": val_loss,
            "validation": val_metrics,
        }
        history.append(epoch_result)
        writer.add_scalar("loss/train", epoch_result["train_loss"], epoch)
        writer.add_scalar("loss/validation", val_loss, epoch)
        for metric_name in ("accuracy", "precision", "recall", "f1", "threshold"):
            writer.add_scalar(f"validation/{metric_name}", val_metrics[metric_name], epoch)
        writer.add_scalar("learning_rate/encoder", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("learning_rate/head", optimizer.param_groups[1]["lr"], epoch)
        writer.flush()
        save_json(args.output_dir / "training_history.json", {"epochs": history})
        save_checkpoint(args.output_dir / "last_checkpoint.pt", model, optimizer, scheduler, epoch, history, config)
        if val_metrics["f1"] > best_f1 or not best_model_path.exists():
            best_f1 = val_metrics["f1"]
            temporary = best_model_path.with_suffix(".pt.tmp")
            torch.save({"model_state": model.state_dict(), "threshold": threshold, "config": config, "epoch": epoch}, temporary)
            os.replace(temporary, best_model_path)
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)

    best = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    test_labels, test_scores, test_loss = evaluate(model, loaders["test"], device)
    test_metrics = binary_metrics(test_labels, test_scores, float(best["threshold"]))
    result = {"best_epoch": best["epoch"], "validation_selected_threshold": best["threshold"],
              "test_loss": test_loss, "test": test_metrics}
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
    parser.add_argument("--feature-set", choices=["context", "all", "embeddings_only"], default="context",
                        help="context excludes direct marker-match flags for a fairer comparison to the rule baseline")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    for field in ("train_pairs", "validation_pairs", "test_pairs"):
        value = getattr(args, field).resolve()
        if not value.is_file():
            parser.error(f"{field.replace('_', '-')} does not exist: {value}")
        setattr(args, field, value)
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        parser.error("epochs, batch size, and gradient accumulation must be positive")
    if args.max_length < 16 or args.log_every < 1 or args.num_workers < 0:
        parser.error("max length must be >=16; log every >=1; num workers >=0")
    args.output_dir = args.output_dir.resolve()
    return args


def main() -> None:
    result = train(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
