```bash
CUDA_VISIBLE_DEVICES=0 python -m src.vdt_anonymization.entity_linking.training \
  --train-pairs outputs/entity_linking_v1/pairs/train.jsonl \
  --validation-pairs outputs/entity_linking_v1/pairs/validation.jsonl \
  --test-pairs outputs/entity_linking_v1/pairs/test.jsonl \
  --output-dir outputs/test_frozen \
  --model-name NlpHUST/ner-vietnamese-electra-base \
  --feature-set context \
  --finetune-mode frozen \
  --epochs 1 \
  --max-train-steps 20 \
  --max-eval-steps 10 \
  --batch-size 16 \
  --device cuda \
  --fp16 \
  --amp-dtype bf16
```
