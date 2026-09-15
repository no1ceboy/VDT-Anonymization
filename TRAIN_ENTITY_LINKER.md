# Train entity linker

```powershell
python src/train_entity_linker.py `
  --train-pairs outputs/entity_linking_v1/pairs/train.jsonl `
  --validation-pairs outputs/entity_linking_v1/pairs/validation.jsonl `
  --test-pairs outputs/entity_linking_v1/pairs/test.jsonl `
  --output-dir outputs/entity_linker_model `
  --model-name NlpHUST/ner-vietnamese-electra-base `
  --feature-set context `
  --epochs 3 `
  --batch-size 8 `
  --gradient-accumulation 2 `
  --max-length 256 `
  --num-workers 0 `
  --fp16
```

```powershell
tensorboard --logdir outputs/entity_linker_model/tensorboard
```
