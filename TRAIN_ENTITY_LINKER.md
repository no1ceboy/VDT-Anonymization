# V3 pairwise pilot — run from the repository root

The V3 pair files are generated already; copy `outputs/entity_linking_v3_location_strict_20260924/pairs/` to the training machine. These labels are weak rule-generated supervision on reconstructed pseudonyms, not human gold labels. Python, PyTorch, Transformers, TensorBoard and the encoder model must already be available; no package installation is performed here.

```bash
# Exact-surface baseline on the regular held-out test set.
python -m src.vdt_anonymization.entity_linking.baseline --pairs outputs/entity_linking_v3_location_strict_20260924/pairs/test.jsonl --methods exact_surface --output outputs/entity_linker_v3_frozen_full/baseline_test.json

# Exact-surface baseline on the separate, negative-only name-collision diagnostic.
python -m src.vdt_anonymization.entity_linking.baseline --pairs outputs/entity_linking_v3_location_strict_20260924/pairs/test_collision_eval.jsonl --methods exact_surface --output outputs/entity_linker_v3_frozen_full/baseline_collision.json

# Train one full epoch, updating only the pair-classification head.
python -m src.vdt_anonymization.entity_linking.training --train-pairs outputs/entity_linking_v3_location_strict_20260924/pairs/train.jsonl --validation-pairs outputs/entity_linking_v3_location_strict_20260924/pairs/validation.jsonl --test-pairs outputs/entity_linking_v3_location_strict_20260924/pairs/test.jsonl --output-dir outputs/entity_linker_v3_frozen_full --finetune-mode frozen --epochs 1 --batch-size 16 --device cuda --fp16 --amp-dtype bf16

# Optional: detailed test breakdown plus collision false-link count/rate.
# This rescans the full main test split after training; skip it for the first smoke run
# if you only need the standard overall metrics in test_metrics.json.
python -m src.vdt_anonymization.entity_linking.evaluate --checkpoint outputs/entity_linker_v3_frozen_full/best_model.pt --pairs outputs/entity_linking_v3_location_strict_20260924/pairs/test.jsonl --collision-pairs outputs/entity_linking_v3_location_strict_20260924/pairs/test_collision_eval.jsonl --output outputs/entity_linker_v3_frozen_full/evaluation.json --device cuda --fp16 --amp-dtype bf16

# Optional: view TensorBoard training curves.
tensorboard --logdir outputs/entity_linker_v3_frozen_full/tensorboard
```
