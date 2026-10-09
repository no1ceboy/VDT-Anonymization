# V4 entity-linking commands

Run the blocks from the repository root in one Bash session.

```bash
DATA=outputs/entity_linking_v4_location_hierarchy_20261009
V3_DATA=outputs/entity_linking_v3_location_strict_20260924
MODEL=NlpHUST/ner-vietnamese-electra-base
RUNS=outputs/entity_linker_v4_ablation_20261009
# Optional GPU selection:
# export CUDA_VISIBLE_DEVICES=0
```

## Data

Extract the V4 data archive to `$DATA`. Optional rebuild from V3; output must be a new directory.

```bash
python3 -m src.vdt_anonymization.entity_linking.v4_dataset \
  --v3-dir "$V3_DATA" \
  --output-dir outputs/entity_linking_v4_rebuilt_20261009 \
  --context-chars 160 \
  --max-positives-per-entity 8 \
  --max-pairs-per-document 128 \
  --train-collision-rate 0.5 \
  --max-train-variants-per-document 3 \
  --seed vdt-link-v4
```

## Smoke test

```bash
python3 -m src.vdt_anonymization.entity_linking.training \
  --train-pairs "$DATA/pairs/train.jsonl" \
  --validation-pairs "$DATA/pairs/validation.jsonl" \
  --test-pairs "$DATA/pairs/test.jsonl" \
  --output-dir "$RUNS/smoke" \
  --model-name "$MODEL" \
  --finetune-mode frozen \
  --epochs 1 \
  --max-train-steps 100 \
  --max-eval-steps 20 \
  --batch-size 16 \
  --gradient-accumulation 1 \
  --max-length 256 \
  --device cuda \
  --fp16 \
  --amp-dtype bf16 \
  --seed 42
```

## Training ablations

```bash
train_linker() {
  local run_name="$1"
  local mode="$2"
  local train_file="$3"
  shift 3
  python3 -m src.vdt_anonymization.entity_linking.training \
    --train-pairs "$DATA/pairs/$train_file" \
    --validation-pairs "$DATA/pairs/validation.jsonl" \
    --test-pairs "$DATA/pairs/test.jsonl" \
    --output-dir "$RUNS/$run_name" \
    --model-name "$MODEL" \
    --finetune-mode "$mode" \
    --epochs 3 \
    --batch-size 32 \
    --gradient-accumulation 1 \
    --max-length 256 \
    --encoder-learning-rate 2e-5 \
    --head-learning-rate 5e-4 \
    --weight-decay 0.01 \
    --warmup-ratio 0.1 \
    --dropout 0.15 \
    --max-grad-norm 1.0 \
    --num-workers 0 \
    --log-every 100 \
    --progress \
    --seed 42 \
    --device cuda \
    --fp16 \
    --amp-dtype bf16 \
    "$@"
}

# Natural-vs-augmented data ablation; then compare encoder tuning modes.
train_linker frozen_natural frozen train_natural.jsonl
train_linker frozen_full frozen train.jsonl
train_linker fft_full fft train.jsonl
train_linker lora_full lora train.jsonl \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules auto

# Optional; requires PEFT and bitsandbytes already installed.
train_linker qlora_full qlora train.jsonl \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules auto \
  --qlora-compute-dtype bf16
```

## Baselines and pair evaluation

```bash
for PAIR_SET in test test_challenge test_location_hard test_surface_hard; do
  python3 -m src.vdt_anonymization.entity_linking.baseline \
    --pairs "$DATA/pairs/${PAIR_SET}.jsonl" \
    --methods exact_marker marker_family exact_surface \
    --output "$RUNS/baseline_${PAIR_SET}.json"
done

# Choose a run after comparing its validation cluster scores.
SELECTED="$RUNS/fft_full"

eval_pairs() {
  local pair_set="$1"
  shift
  python3 -m src.vdt_anonymization.entity_linking.evaluate \
    --checkpoint "$SELECTED/best_model.pt" \
    --pairs "$DATA/pairs/${pair_set}.jsonl" \
    --output "$SELECTED/${pair_set}_evaluation.json" \
    --batch-size 32 \
    --max-length 256 \
    --device cuda \
    --fp16 \
    --amp-dtype bf16 \
    "$@"
}

eval_pairs test \
  --collision-pairs "$DATA/pairs/test_challenge.jsonl"
eval_pairs test_location_hard
eval_pairs test_surface_hard
```

## Document-level evaluation

```bash
for EXPERIMENT in frozen_natural frozen_full lora_full qlora_full fft_full; do
  if [[ -f "$RUNS/$EXPERIMENT/best_model.pt" ]]; then
    python3 -m src.vdt_anonymization.entity_linking.cluster_evaluate \
      --checkpoint "$RUNS/$EXPERIMENT/best_model.pt" \
      --documents "$DATA/documents/validation.jsonl" \
      --maps "$DATA/maps/validation.jsonl" \
      --select-threshold \
      --output "$RUNS/$EXPERIMENT/cluster_validation.json" \
      --device cuda \
      --amp-dtype bf16
  fi
done

python3 -m src.vdt_anonymization.entity_linking.cluster_evaluate \
  --checkpoint "$SELECTED/best_model.pt" \
  --documents "$DATA/documents/test.jsonl" \
  --maps "$DATA/maps/test.jsonl" \
  --threshold-file "$SELECTED/cluster_validation.json" \
  --output "$SELECTED/cluster_test.json" \
  --device cuda \
  --amp-dtype bf16

python3 -m src.vdt_anonymization.entity_linking.cluster_evaluate \
  --checkpoint "$SELECTED/best_model.pt" \
  --documents "$DATA/documents/test_challenge.jsonl" \
  --maps "$DATA/maps/test_challenge.jsonl" \
  --threshold-file "$SELECTED/cluster_validation.json" \
  --output "$SELECTED/cluster_test_challenge.json" \
  --device cuda \
  --amp-dtype bf16
```

## TensorBoard and inference

```bash
tensorboard --logdir "$RUNS"

python3 -m src.vdt_anonymization.entity_linking.inference \
  --checkpoint "$SELECTED/best_model.pt" \
  --input ner_documents.jsonl \
  --output linked_documents.jsonl \
  --threshold-file "$SELECTED/cluster_validation.json" \
  --device cuda \
  --amp-dtype bf16
```
