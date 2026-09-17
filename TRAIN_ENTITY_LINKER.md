```bash
CUDA_VISIBLE_DEVICES=0 python -m src.vdt_anonymization.entity_linking.training \
  --train-pairs outputs/entity_linking_v1/pairs/train.jsonl \
  --validation-pairs outputs/entity_linking_v1/pairs/validation.jsonl \
  --test-pairs outputs/entity_linking_v1/pairs/test.jsonl \
  --output-dir outputs/test_frozen \
  --model-name NlpHUST/ner-vietnamese-electra-base \
  --feature-set context --finetune-mode frozen \
  --epochs 1 --max-train-steps 20 --max-eval-steps 10 \
  --batch-size 16 --device cuda --fp16 --amp-dtype bf16

CUDA_VISIBLE_DEVICES=0 python -m src.vdt_anonymization.entity_linking.baseline \
  --pairs outputs/entity_linking_v1/pairs/test.jsonl \
  --output outputs/entity_linking_v1/baseline_test.json

CUDA_VISIBLE_DEVICES=0 python -m src.vdt_anonymization.entity_linking.training \
  --train-pairs outputs/entity_linking_v1/pairs/train.jsonl \
  --validation-pairs outputs/entity_linking_v1/pairs/validation.jsonl \
  --test-pairs outputs/entity_linking_v1/pairs/test.jsonl \
  --output-dir outputs/ablation_frozen \
  --model-name NlpHUST/ner-vietnamese-electra-base \
  --feature-set context --finetune-mode frozen \
  --epochs 5 --batch-size 64 --device cuda --fp16 --amp-dtype bf16

CUDA_VISIBLE_DEVICES=0 python -m src.vdt_anonymization.entity_linking.training \
  --train-pairs outputs/entity_linking_v1/pairs/train.jsonl \
  --validation-pairs outputs/entity_linking_v1/pairs/validation.jsonl \
  --test-pairs outputs/entity_linking_v1/pairs/test.jsonl \
  --output-dir outputs/ablation_lora_context \
  --model-name NlpHUST/ner-vietnamese-electra-base \
  --feature-set context --finetune-mode lora \
  --epochs 5 --batch-size 64 --device cuda --fp16 --amp-dtype bf16

CUDA_VISIBLE_DEVICES=0 python -m src.vdt_anonymization.entity_linking.training \
  --train-pairs outputs/entity_linking_v1/pairs/train.jsonl \
  --validation-pairs outputs/entity_linking_v1/pairs/validation.jsonl \
  --test-pairs outputs/entity_linking_v1/pairs/test.jsonl \
  --output-dir outputs/ablation_lora_embeddings_only \
  --model-name NlpHUST/ner-vietnamese-electra-base \
  --feature-set embeddings_only --finetune-mode lora \
  --epochs 5 --batch-size 64 --device cuda --fp16 --amp-dtype bf16

CUDA_VISIBLE_DEVICES=0 python -m src.vdt_anonymization.entity_linking.training \
  --train-pairs outputs/entity_linking_v1/pairs/train.jsonl \
  --validation-pairs outputs/entity_linking_v1/pairs/validation.jsonl \
  --test-pairs outputs/entity_linking_v1/pairs/test.jsonl \
  --output-dir outputs/ablation_qlora_context \
  --model-name NlpHUST/ner-vietnamese-electra-base \
  --feature-set context --finetune-mode qlora \
  --epochs 5 --batch-size 64 --device cuda --fp16 --amp-dtype bf16

CUDA_VISIBLE_DEVICES=0 python -m src.vdt_anonymization.entity_linking.training \
  --train-pairs outputs/entity_linking_v1/pairs/train.jsonl \
  --validation-pairs outputs/entity_linking_v1/pairs/validation.jsonl \
  --test-pairs outputs/entity_linking_v1/pairs/test.jsonl \
  --output-dir outputs/ablation_fft_context \
  --model-name NlpHUST/ner-vietnamese-electra-base \
  --feature-set context --finetune-mode fft \
  --epochs 3 --batch-size 64 --device cuda --fp16 --amp-dtype bf16

CUDA_VISIBLE_DEVICES=0 python -m src.vdt_anonymization.entity_linking.training \
  --train-pairs outputs/entity_linking_v1/pairs/train.jsonl \
  --validation-pairs outputs/entity_linking_v1/pairs/validation.jsonl \
  --test-pairs outputs/entity_linking_v1/pairs/test.jsonl \
  --output-dir outputs/entity_linker_full \
  --model-name NlpHUST/ner-vietnamese-electra-base \
  --feature-set context --finetune-mode lora \
  --epochs 10 --batch-size 64 --device cuda --fp16 --amp-dtype bf16

CUDA_VISIBLE_DEVICES=0 python -m src.vdt_anonymization.entity_linking.evaluate \
  --checkpoint outputs/entity_linker_full/best_model.pt \
  --pairs outputs/entity_linking_v1/pairs/test.jsonl \
  --output outputs/entity_linker_full/evaluation.json \
  --device cuda --fp16 --amp-dtype bf16

tensorboard --logdir outputs/entity_linker_full/tensorboard
```
