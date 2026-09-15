# Entity-linker training runbook

This runbook is for a Windows PowerShell company workstation. Run every command
from the repository root. The recommended first experiment trains the context
model without direct marker-equality features, so its result can be compared
fairly with the existing exact-marker rule baseline.

## 0. Get the code

For an existing checkout:

```powershell
git pull --ff-only
```

For a new checkout:

```powershell
git clone https://github.com/no1ceboy/VDT-Anonymization.git
Set-Location VDT-Anonymization
```

## 1. Copy the training data

The generated pair files are intentionally excluded from Git. Copy this folder
from the development machine to the same relative location on the training
machine:

```text
outputs/entity_linking_v1/pairs/
```

Only these three files are required for model training:

| File | Examples | Bytes | SHA-256 |
|---|---:|---:|---|
| `train.jsonl` | 204,079 | 380,810,526 | `11ffc0bee4189cd025c17772e1be321c761f9da4616ed001d5b8cf62f1b63bdd` |
| `validation.jsonl` | 24,780 | 46,346,858 | `9ee1f19ed3c5d73b62856f71c01af850119188deea6d572c10b54b55400937d4` |
| `test.jsonl` | 24,409 | 45,533,562 | `3fd235d8d4805afcc2ef049f95e2404f1e96cd7d8130c47cdbd9b8b33cc0fb37` |

Verify the copied files:

```powershell
Get-FileHash -Algorithm SHA256 outputs/entity_linking_v1/pairs/train.jsonl
Get-FileHash -Algorithm SHA256 outputs/entity_linking_v1/pairs/validation.jsonl
Get-FileHash -Algorithm SHA256 outputs/entity_linking_v1/pairs/test.jsonl
```

Do not train if a hash differs; recopy the affected file first.

## 2. Create the environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the company has an approved CUDA-specific PyTorch package or internal wheel
index, install that PyTorch build before `requirements.txt`. The requirements
command will retain it when it already satisfies `torch>=2.1`.

Confirm that PyTorch can use the GPU:

```powershell
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Do not start the full run on CPU unless a very long training time is acceptable.

## 3. Make the encoder available

If the workstation can access Hugging Face, use the model ID directly:

```powershell
$ModelPath = "NlpHUST/ner-vietnamese-electra-base"
```

For an offline workstation, copy a complete Transformers model snapshot into an
approved local directory and use that path instead. It must include the model
configuration, tokenizer files, and model weights.

```powershell
$ModelPath = "D:\models\ner-vietnamese-electra-base"
Test-Path "$ModelPath\config.json"
```

The final command must print `True`. The training program does not upload the
documents or configure any external experiment tracker. TensorBoard logs remain
local.

## 4. Verify the code

```powershell
python -B -m unittest discover -s tests -v
```

The expected result for this revision is 56 passing tests.

Reproduce the fixed rule baseline locally:

```powershell
python src/evaluate_entity_linking_baseline.py `
  --pairs outputs/entity_linking_v1/pairs/test.jsonl `
  --output outputs/entity_linking_v1/baseline_test.json
```

## 5. Start TensorBoard

Open a second PowerShell terminal, activate the same environment, and run:

```powershell
.\.venv\Scripts\Activate.ps1
tensorboard --logdir outputs/entity_linker_model/tensorboard --host 127.0.0.1 --port 6006
```

Then open `http://127.0.0.1:6006`. The trainer logs batch loss every 100 batches,
encoder/head learning rates, epoch losses, validation precision/recall/F1 and
threshold, and final test metrics.

## 6. Train the recommended context model

An effective batch size of 16 is achieved with batch size 8 and two gradient
accumulation steps. This is the safer starting point for an 8–12 GB GPU.

```powershell
python src/train_entity_linker.py `
  --train-pairs outputs/entity_linking_v1/pairs/train.jsonl `
  --validation-pairs outputs/entity_linking_v1/pairs/validation.jsonl `
  --test-pairs outputs/entity_linking_v1/pairs/test.jsonl `
  --output-dir outputs/entity_linker_model `
  --model-name $ModelPath `
  --feature-set context `
  --epochs 3 `
  --batch-size 8 `
  --gradient-accumulation 2 `
  --max-length 256 `
  --encoder-learning-rate 2e-5 `
  --head-learning-rate 5e-4 `
  --num-workers 0 `
  --log-every 100 `
  --fp16
```

If CUDA runs out of memory, replace the two batch arguments with:

```powershell
  --batch-size 4 `
  --gradient-accumulation 4 `
```

On a GPU with at least 24 GB, try `--batch-size 16` and
`--gradient-accumulation 1`.

## 7. Resume after an interruption

The trainer atomically writes `last_checkpoint.pt` after each completed epoch.
Resume with:

```powershell
python src/train_entity_linker.py `
  --train-pairs outputs/entity_linking_v1/pairs/train.jsonl `
  --validation-pairs outputs/entity_linking_v1/pairs/validation.jsonl `
  --test-pairs outputs/entity_linking_v1/pairs/test.jsonl `
  --output-dir outputs/entity_linker_model `
  --model-name $ModelPath `
  --feature-set context `
  --epochs 3 `
  --batch-size 8 `
  --gradient-accumulation 2 `
  --max-length 256 `
  --encoder-learning-rate 2e-5 `
  --head-learning-rate 5e-4 `
  --num-workers 0 `
  --log-every 100 `
  --fp16 `
  --resume outputs/entity_linker_model/last_checkpoint.pt
```

Keep `--epochs 3`: it is the total target epoch count, not the number of extra
epochs.

## 8. Inspect the result

The primary files are:

```text
outputs/entity_linker_model/best_model.pt
outputs/entity_linker_model/last_checkpoint.pt
outputs/entity_linker_model/training_history.json
outputs/entity_linker_model/test_metrics.json
outputs/entity_linker_model/config.json
outputs/entity_linker_model/tensorboard/
```

The fixed rule baseline on the same test split has F1 `0.9617`, precision
`0.9261`, and recall `1.0000`. Compare the model against it using test F1 and,
especially, false positives. The test labels are weak supervision produced by
the reconstruction linker, not independently human-reviewed ground truth, so
the reported result measures agreement with those labels rather than proven
real-world linking accuracy.
