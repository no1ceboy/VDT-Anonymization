# Entity-linker commands

Run these commands from the repository root.

```powershell
python -m pip install .
```

For LoRA/QLoRA only:

```powershell
python -m pip install ".[efficient]"
```

Set the pair files once:

```powershell
$pairs = "outputs/entity_linking_v1/pairs"
```

Pilot one mode first. It trains 200 batches and evaluates 100 batches, which
is enough to catch device, tokenizer, target-module, and memory problems:

```powershell
vdt train-linker --train-pairs "$pairs/train.jsonl" --validation-pairs "$pairs/validation.jsonl" --test-pairs "$pairs/test.jsonl" --output-dir outputs/linker_pilot_lora --finetune-mode lora --model-name NlpHUST/ner-vietnamese-electra-base --feature-set context --epochs 1 --max-train-steps 200 --max-eval-steps 100 --batch-size 64 --device cuda --fp16 --amp-dtype bf16
```

The available modes are `fft` (full fine-tuning), `frozen` (MLP head only),
`lora`, and `qlora` (4-bit NF4 base plus LoRA). Use one output directory per
run. TensorBoard and JSON include peak allocated/reserved GPU memory.

For the overnight ablation, run sequentially on one GPU. Change only the mode
or feature set so the comparison remains meaningful:

```powershell
vdt train-linker --train-pairs "$pairs/train.jsonl" --validation-pairs "$pairs/validation.jsonl" --test-pairs "$pairs/test.jsonl" --output-dir outputs/linker_frozen --finetune-mode frozen --feature-set context --epochs 5 --batch-size 64 --device cuda --fp16 --amp-dtype bf16
vdt train-linker --train-pairs "$pairs/train.jsonl" --validation-pairs "$pairs/validation.jsonl" --test-pairs "$pairs/test.jsonl" --output-dir outputs/linker_lora --finetune-mode lora --feature-set context --epochs 5 --batch-size 64 --device cuda --fp16 --amp-dtype bf16
vdt train-linker --train-pairs "$pairs/train.jsonl" --validation-pairs "$pairs/validation.jsonl" --test-pairs "$pairs/test.jsonl" --output-dir outputs/linker_lora_no_context --finetune-mode lora --feature-set embeddings_only --epochs 5 --batch-size 64 --device cuda --fp16 --amp-dtype bf16
vdt train-linker --train-pairs "$pairs/train.jsonl" --validation-pairs "$pairs/validation.jsonl" --test-pairs "$pairs/test.jsonl" --output-dir outputs/linker_qlora --finetune-mode qlora --feature-set context --epochs 5 --batch-size 64 --device cuda --fp16 --amp-dtype bf16
vdt train-linker --train-pairs "$pairs/train.jsonl" --validation-pairs "$pairs/validation.jsonl" --test-pairs "$pairs/test.jsonl" --output-dir outputs/linker_fft --finetune-mode fft --feature-set context --epochs 3 --batch-size 64 --device cuda --fp16 --amp-dtype bf16
```

Start with `--batch-size 64`; after the pilot, increase it if throughput rises
and peak memory remains below about 80--85% of HBM. Do not run these jobs in
parallel on one GPU: competing jobs reduce throughput and make OOM recovery
less safe. Every epoch writes `last_checkpoint.pt`; resume with
`--resume outputs/<run>/last_checkpoint.pt`.

```powershell
tensorboard --logdir outputs/linker_lora/tensorboard
```

Evaluation of a completed run:

```powershell
vdt eval-linker --checkpoint outputs/linker_lora/best_model.pt --pairs "$pairs/test.jsonl" --output outputs/linker_lora/evaluation.json --device cuda --fp16 --amp-dtype bf16
```
