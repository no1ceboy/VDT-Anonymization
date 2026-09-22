# Pilot 1: marker-free entity linking on name-bearing synthetic text

The ZIP includes the prepared dataset at `outputs/entity_linking_raw_v1`. Run
these commands from the extracted project root. No dataset rebuild is needed.

```bash
# Measure the text-only exact-surface baseline on the raw-name test pairs.
CUDA_VISIBLE_DEVICES=0 python -m src.vdt_anonymization.entity_linking.baseline \
  --pairs outputs/entity_linking_raw_v1/pairs/test.jsonl \
  --methods exact_surface \
  --output outputs/entity_linking_raw_v1/baseline_test.json

# Train the first Pilot 1 linker; names are visible, marker features are excluded.
CUDA_VISIBLE_DEVICES=0 python -m src.vdt_anonymization.entity_linking.training \
  --train-pairs outputs/entity_linking_raw_v1/pairs/train.jsonl \
  --validation-pairs outputs/entity_linking_raw_v1/pairs/validation.jsonl \
  --test-pairs outputs/entity_linking_raw_v1/pairs/test.jsonl \
  --output-dir outputs/entity_linker_raw_frozen \
  --finetune-mode frozen \
  --epochs 3 --batch-size 64 --device cuda --fp16 --amp-dtype bf16

# Evaluate the trained linker on the held-out raw-name test pairs.
CUDA_VISIBLE_DEVICES=0 python -m src.vdt_anonymization.entity_linking.evaluate \
  --checkpoint outputs/entity_linker_raw_frozen/best_model.pt \
  --pairs outputs/entity_linking_raw_v1/pairs/test.jsonl \
  --output outputs/entity_linker_raw_frozen/evaluation.json \
  --device cuda --fp16 --amp-dtype bf16

# View training and validation curves.
tensorboard --logdir outputs/entity_linker_raw_frozen/tensorboard
```
