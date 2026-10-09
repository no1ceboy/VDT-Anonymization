# V4 entity-linking commands

Run from the repository root. Extract the V4 archive to `outputs/entity_linking_v4_location_hierarchy_20261009`.

## Optional V4 rebuild

Run only if the prepared V4 dataset is unavailable. The output path must not already exist.

```bash
python3 -m src.vdt_anonymization.entity_linking.v4_dataset \
  --v3-dir outputs/entity_linking_v3_location_strict_20260924 \
  --output-dir outputs/entity_linking_v4_location_hierarchy_20261009 \
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
  --train-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/train.jsonl \
  --validation-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/validation.jsonl \
  --test-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/test.jsonl \
  --output-dir outputs/entity_linker_v4_ablation_20261009/smoke \
  --model-name NlpHUST/ner-vietnamese-electra-base \
  --finetune-mode frozen \
  --epochs 1 \
  --max-train-steps 100 \
  --max-eval-steps 20 \
  --batch-size 16 \
  --gradient-accumulation 1 \
  --device cuda \
  --fp16 \
  --amp-dtype bf16 \
  --seed 42
```

## Training ablations

Natural-only frozen baseline:

```bash
python3 -m src.vdt_anonymization.entity_linking.training \
  --train-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/train_natural.jsonl \
  --validation-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/validation.jsonl \
  --test-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/test.jsonl \
  --output-dir outputs/entity_linker_v4_ablation_20261009/frozen_natural \
  --model-name NlpHUST/ner-vietnamese-electra-base \
  --finetune-mode frozen \
  --epochs 3 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --device cuda \
  --fp16 \
  --amp-dtype bf16 \
  --seed 42
```

Full V4 frozen baseline:

```bash
python3 -m src.vdt_anonymization.entity_linking.training \
  --train-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/train.jsonl \
  --validation-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/validation.jsonl \
  --test-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/test.jsonl \
  --output-dir outputs/entity_linker_v4_ablation_20261009/frozen_full \
  --model-name NlpHUST/ner-vietnamese-electra-base \
  --finetune-mode frozen \
  --epochs 3 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --device cuda \
  --fp16 \
  --amp-dtype bf16 \
  --seed 42
```

Full V4 LoRA:

```bash
python3 -m src.vdt_anonymization.entity_linking.training \
  --train-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/train.jsonl \
  --validation-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/validation.jsonl \
  --test-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/test.jsonl \
  --output-dir outputs/entity_linker_v4_ablation_20261009/lora_full \
  --model-name NlpHUST/ner-vietnamese-electra-base \
  --finetune-mode lora \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules auto \
  --epochs 3 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --device cuda \
  --fp16 \
  --amp-dtype bf16 \
  --seed 42
```

Full V4 QLoRA; requires PEFT and bitsandbytes already available:

```bash
python3 -m src.vdt_anonymization.entity_linking.training \
  --train-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/train.jsonl \
  --validation-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/validation.jsonl \
  --test-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/test.jsonl \
  --output-dir outputs/entity_linker_v4_ablation_20261009/qlora_full \
  --model-name NlpHUST/ner-vietnamese-electra-base \
  --finetune-mode qlora \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules auto \
  --qlora-compute-dtype bf16 \
  --epochs 3 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --device cuda \
  --fp16 \
  --amp-dtype bf16 \
  --seed 42
```

Full V4 FFT, primary run:

```bash
python3 -m src.vdt_anonymization.entity_linking.training \
  --train-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/train.jsonl \
  --validation-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/validation.jsonl \
  --test-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/test.jsonl \
  --output-dir outputs/entity_linker_v4_ablation_20261009/fft_full \
  --model-name NlpHUST/ner-vietnamese-electra-base \
  --finetune-mode fft \
  --epochs 3 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --device cuda \
  --fp16 \
  --amp-dtype bf16 \
  --seed 42
```

## Baselines and pair evaluation

```bash
python3 -m src.vdt_anonymization.entity_linking.baseline \
  --pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/test.jsonl \
  --methods exact_marker marker_family exact_surface \
  --output outputs/entity_linker_v4_ablation_20261009/baseline_test.json

python3 -m src.vdt_anonymization.entity_linking.baseline \
  --pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/test_challenge.jsonl \
  --methods exact_marker marker_family exact_surface \
  --output outputs/entity_linker_v4_ablation_20261009/baseline_test_challenge.json

python3 -m src.vdt_anonymization.entity_linking.evaluate \
  --checkpoint outputs/entity_linker_v4_ablation_20261009/fft_full/best_model.pt \
  --pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/test.jsonl \
  --collision-pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/test_challenge.jsonl \
  --output outputs/entity_linker_v4_ablation_20261009/fft_full/pair_evaluation.json \
  --batch-size 32 \
  --device cuda \
  --fp16 \
  --amp-dtype bf16

python3 -m src.vdt_anonymization.entity_linking.evaluate \
  --checkpoint outputs/entity_linker_v4_ablation_20261009/fft_full/best_model.pt \
  --pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/test_location_hard.jsonl \
  --output outputs/entity_linker_v4_ablation_20261009/fft_full/location_hard_evaluation.json \
  --batch-size 32 \
  --device cuda \
  --fp16 \
  --amp-dtype bf16

python3 -m src.vdt_anonymization.entity_linking.evaluate \
  --checkpoint outputs/entity_linker_v4_ablation_20261009/fft_full/best_model.pt \
  --pairs outputs/entity_linking_v4_location_hierarchy_20261009/pairs/test_surface_hard.jsonl \
  --output outputs/entity_linker_v4_ablation_20261009/fft_full/surface_hard_evaluation.json \
  --batch-size 32 \
  --device cuda \
  --fp16 \
  --amp-dtype bf16
```

## Document-level evaluation

Select a threshold using validation documents/maps only:

```bash
python3 -m src.vdt_anonymization.entity_linking.cluster_evaluate \
  --checkpoint outputs/entity_linker_v4_ablation_20261009/fft_full/best_model.pt \
  --documents outputs/entity_linking_v4_location_hierarchy_20261009/documents/validation.jsonl \
  --maps outputs/entity_linking_v4_location_hierarchy_20261009/maps/validation.jsonl \
  --select-threshold \
  --output outputs/entity_linker_v4_ablation_20261009/fft_full/cluster_validation.json \
  --device cuda \
  --amp-dtype bf16
```

Evaluate natural test documents/maps:

```bash
python3 -m src.vdt_anonymization.entity_linking.cluster_evaluate \
  --checkpoint outputs/entity_linker_v4_ablation_20261009/fft_full/best_model.pt \
  --documents outputs/entity_linking_v4_location_hierarchy_20261009/documents/test.jsonl \
  --maps outputs/entity_linking_v4_location_hierarchy_20261009/maps/test.jsonl \
  --threshold-file outputs/entity_linker_v4_ablation_20261009/fft_full/cluster_validation.json \
  --output outputs/entity_linker_v4_ablation_20261009/fft_full/cluster_test.json \
  --device cuda \
  --amp-dtype bf16
```

Evaluate held-out collision challenge documents/maps:

```bash
python3 -m src.vdt_anonymization.entity_linking.cluster_evaluate \
  --checkpoint outputs/entity_linker_v4_ablation_20261009/fft_full/best_model.pt \
  --documents outputs/entity_linking_v4_location_hierarchy_20261009/documents/test_challenge.jsonl \
  --maps outputs/entity_linking_v4_location_hierarchy_20261009/maps/test_challenge.jsonl \
  --threshold-file outputs/entity_linker_v4_ablation_20261009/fft_full/cluster_validation.json \
  --output outputs/entity_linker_v4_ablation_20261009/fft_full/cluster_test_challenge.json \
  --device cuda \
  --amp-dtype bf16
```

## TensorBoard and inference

```bash
tensorboard --logdir outputs/entity_linker_v4_ablation_20261009

python3 -m src.vdt_anonymization.entity_linking.inference \
  --checkpoint outputs/entity_linker_v4_ablation_20261009/fft_full/best_model.pt \
  --input ner_documents.jsonl \
  --output linked_documents.jsonl \
  --threshold-file outputs/entity_linker_v4_ablation_20261009/fft_full/cluster_validation.json \
  --device cuda \
  --amp-dtype bf16
```
