# Company handoff: Vietnamese court-document anonymization

## 1. Purpose and scope

This package takes a Vietnamese court decision, detects protected person,
location, and organization mentions, links mentions that refer to the same
entity, and prepares a law-based anonymization plan. It provides both a local
HTTP API and the commands needed to reproduce the entity-linking experiment.
Document text and model inference stay on the machine running the service.

The handoff has two separate uses:

1. **Run the service.** Install the code, place the model assets on the
   company machine, and send text or a DOCX to the local API.
2. **Reproduce the experiment.** Use the saved document-disjoint data splits,
   rebuild or fine-tune the linker, and evaluate both mention pairs and whole
   document clusters.

The current API uses the trained pairwise linker plus the production
clustering and legal anonymization code. The document-memory attention model
is a separate experiment; its checkpoint format is not connected to this API.

## 2. What runs on a document

The runtime follows this sequence:

```text
Text or DOCX
  → extract text and preserve source offsets
  → optional OCR spacing normalization
  → NER for PER / LOC / ORG
  → compare same-type mention pairs with the trained scorer
  → complete-link clustering with name and location constraints
  → identify court personnel to preserve
  → prepare anonymization codes and source-span replacements
  → return anonymized text / DOCX, or request review
```

For each pair of mentions with the same NER type, the ELECTRA-based scorer
combines its contextual mention representations with the pair features saved
in the checkpoint. Complete-link clustering uses the configured document
threshold and enforces location consistency rules. For people, mentions whose
final name words differ are kept apart. The document threshold is separate
from the checkpoint's pair threshold.

The anonymization planner applies the project’s current coding policy. Person
names keep the initial required by the publication format; repeated initials
receive distinguishing suffixes. Locations use administrative level and
hierarchy, and province-level names are preserved. Organization names receive
document-local codes; a locality inside a public-body name can reuse the code
for that locality elsewhere in the document. Mentions classified as protected
court personnel are preserved by the existing role heuristic. If a location
cannot be safely assigned a level, the plan asks for review instead of
producing partial anonymized output.

For DOCX, the API edits text in regular paragraphs, tables, headers, and
footers while retaining ordinary run formatting. It refuses unsupported
paragraph structures rather than returning a partially changed file.

## 3. Source and model assets

Install the repository itself as the code package. Keep model files in a
separate, access-controlled directory; model weights and court documents are
not included in a source checkout or Python wheel.

The current deployable linker checkpoint is:

```text
outputs/ocr_ngram_document_memory_20261022/baseline_run/
  entity_linker_ocr_ngram_20261021/pairwise_model/best_model.pt
```

Its sibling `tokenizer/` directory must travel with it. The checkpoint is
about 517 MiB. It was fine-tuned from
`NlpHUST/ner-vietnamese-electra-base` in full fine-tuning mode. The company
machine also needs a local copy of that base encoder and a local NER checkpoint
for the same NlpHUST model. The NER checkpoint and linker base encoder may be
copies of the same downloaded model, but the NER tokenizer/config and linker
tokenizer are loaded from their configured locations and should both be
present.

Optional: the pre-NER OCR spacing model is
`outputs/ocr_ngram_retrain_20261021/ocr_char_trigram.json`. Without it, the API
skips the pre-NER spacing correction and runs NER on the submitted text as-is.

Suggested machine layout:

```text
vdt-deploy/
  models/
    ner/                       # complete local NER checkpoint directory
    electra-base/               # complete local base encoder directory
    linker/
      best_model.pt
      tokenizer/                # copy from the checkpoint's sibling folder
    ocr_char_trigram.json        # optional
```

Use a verified transfer method for these assets. Record SHA-256 checksums,
repository revision, model source, and transfer date in the company’s model
inventory. Do not include the private DOCX used for local review in a shared
handoff archive.

## 4. Install on a company Windows machine

Use Python 3.11 (the project supports Python 3.10 or newer). Install the
company-approved PyTorch build first, selecting CPU or CUDA to match the
machine. Then install the repository and its API extra:

```powershell
cd C:\path\to\VDT-Anonymization
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[api]"
```

For the experiment commands, add the data and training extras:

```powershell
python -m pip install -e ".[data,training]"
```

On Linux, create the environment with `python3.11 -m venv .venv`, activate it
with `source .venv/bin/activate`, and use the same `pip install` and `vdt`
commands. Export the settings below as `VDT_*` environment variables before
starting the service.

For a restricted network, prepare a wheelhouse on an approved connected
machine using the same Python version, platform, and extras, scan it, transfer
it, and install with `python -m pip install --no-index --find-links
<wheelhouse> ".[api]"` (and `".[data,training]"` on the experiment host). Copy
all model directories separately. Set Hugging Face
offline mode so missing files fail locally instead of attempting a download:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

Set the four asset paths and a secret API key in the service process. Use a
company secret store for persistent deployments; this example is only for a
local shell session:

```powershell
$env:VDT_NER_MODEL_PATH = "D:\vdt-deploy\models\ner"
$env:VDT_LINKER_BASE_MODEL = "D:\vdt-deploy\models\electra-base"
$env:VDT_LINKER_CHECKPOINT = "D:\vdt-deploy\models\linker\best_model.pt"
$env:VDT_OCR_NGRAM_MODEL = "D:\vdt-deploy\models\ocr_char_trigram.json" # optional
$env:VDT_API_KEY = "set-this-from-the-company-secret-store"
$env:VDT_DEVICE = "auto"             # or cpu / cuda
$env:VDT_CLUSTER_THRESHOLD = "0.10"   # calibrated document-level threshold
```

Start one worker. The default listens only on the local machine:

```powershell
vdt serve-api --host 127.0.0.1 --port 8000
```

The API loads the models once during startup and serializes inference. Keep one
worker: starting multiple workers duplicates the model and GPU memory. For
internal network access, bind to the approved interface only after placing the
service behind the company VPN or reverse proxy with TLS and access controls;
keep the API key enabled. The built-in key check is not a substitute for the
company’s network controls.

## 5. Call the API

Check process readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/readyz
```

Submit plain text and receive a JSON result containing NER mentions, clusters,
the anonymization plan, and anonymized text when the plan is ready:

```powershell
$body = @{ doc_id = "case-001"; text = "Nội dung quyết định cần xử lý." } | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/v1/anonymize/text `
  -Headers @{ "X-API-Key" = $env:VDT_API_KEY } `
  -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

Submit a DOCX and save the anonymized DOCX returned by the service:

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/anonymize/docx `
  -H "X-API-Key: $env:VDT_API_KEY" `
  -F "file=@D:\input\decision.docx" `
  --output D:\output\decision_anonymized.docx
```

The DOCX endpoint returns HTTP 200 with a DOCX only when every planned action
is ready. HTTP 409 means the response contains a review-required plan and no
anonymized DOCX was produced; this includes the case where NER found no
supported entities, so unchanged text is never reported as a successful
anonymization. HTTP 422 means the document or spans could not be processed
safely; correct the input or review its structure. `/healthz` and `/readyz` are
unauthenticated liveness checks; document endpoints require `X-API-Key`.

Limits are configurable with `VDT_MAX_CHARS` (default 1,000,000),
`VDT_MAX_MENTIONS` (500), `VDT_MAX_DOCX_BYTES` (20 MiB), and
`VDT_MAX_DOCX_EXPANDED_BYTES` (200 MiB). Increase them only after evaluating
latency and memory on the target machine. The service does not retain uploaded
documents; callers remain responsible for handling returned files and audit
records securely.

## 6. Reproduce the current linker experiment

### Data used

The checked local experiment contains 8,206 document-disjoint natural
documents: 6,620 train, 793 validation, and 793 test. The raw natural training
split has 160,271 candidate pairs. The actual training file is a balanced
33,600-pair file: 26,880 selected natural pairs plus 6,720 reconstructed
challenge pairs. The challenge pairs make up 20% of the training rows, are
unique (not repeated or sample-weighted), and are balanced across linked vs.
unlinked and same vs. different mention surface. The validation and test each
have a separate 720-pair challenge set. The full dataset manifest and
challenge construction notes are saved beside the experiment artifacts.

These identity labels come from reconstruction maps created from already
anonymized court decisions. They support controlled comparisons but are not
independently verified original names. Report natural and injected-challenge
results separately.

Copy these folders from the handoff source to the company machine, preserving
their relative files:

```text
outputs/ocr_ngram_retrain_20261021/raw_dataset/
outputs/ocr_ngram_document_memory_20261022/baseline_run/
  entity_linker_ocr_ngram_20261021/challenge_dataset/
```

Also copy the local base encoder directory. Let `DATA` point to the copied
`raw_dataset` folder, `CHALLENGE` to the copied `challenge_dataset` folder,
`BASE` to the base encoder directory, and `RUN` to a new empty output folder.
Do not overwrite the saved result directory.

```powershell
$DATA = "D:\transfer\raw_dataset"
$CHALLENGE = "D:\transfer\challenge_dataset"
$BASE = "D:\models\electra-base"
$RUN = "D:\runs\vdt-linker-001"
```

### Fine-tune the pair scorer

The following matches the saved run’s central settings: full fine-tuning for
three epochs, 8 examples per device batch, gradient accumulation 2, maximum
length 256, seed 42. The base model must be a local folder when the machine is
offline.

```powershell
vdt train-linker `
  --train-pairs "$CHALLENGE\pairs\train_balanced_independent.jsonl" `
  --validation-pairs "$DATA\pairs\validation.jsonl" `
  --test-pairs "$DATA\pairs\test.jsonl" `
  --output-dir "$RUN\pairwise_model" `
  --model-name "$BASE" `
  --finetune-mode fft `
  --epochs 3 --batch-size 8 --gradient-accumulation 2 `
  --max-length 256 --num-workers 0 --seed 42 `
  --device cuda --fp16 --amp-dtype fp16
```

On CPU, use `--device cpu --no-fp16`; expect training to take substantially
longer. The full fine-tuning run requires a GPU with enough memory for the
chosen batch and sequence length. If that is unavailable, run the same
experiment on the approved GPU host rather than silently changing the setup.

### Evaluate pairs and whole documents

Score the natural test pairs and the separate challenge test pairs. The second
file is not mixed into the natural metric:

```powershell
vdt eval-linker `
  --checkpoint "$RUN\pairwise_model\best_model.pt" `
  --model-name "$BASE" `
  --pairs "$DATA\pairs\test.jsonl" `
  --collision-pairs "$CHALLENGE\pairs\test_independent_challenge.jsonl" `
  --output "$RUN\pairwise_model\test_evaluation.json" `
  --device cuda --amp-dtype fp16
```

Evaluate clustering on natural documents and on the held-out challenge
documents separately. Keep the document threshold at the configured 0.10 for
reproduction; if selecting a new threshold, select it on validation only and
pass that validation result to the test run.

```powershell
vdt eval-clusters `
  --checkpoint "$RUN\pairwise_model\best_model.pt" `
  --model-name "$BASE" `
  --documents "$DATA\documents\test.jsonl" --maps "$DATA\maps\test.jsonl" `
  --threshold 0.10 --output "$RUN\cluster_test_natural.json" --device cuda

vdt eval-clusters `
  --checkpoint "$RUN\pairwise_model\best_model.pt" `
  --model-name "$BASE" `
  --documents "$CHALLENGE\documents\test_independent_challenge.jsonl" `
  --maps "$CHALLENGE\maps\test_independent_challenge.jsonl" `
  --threshold 0.10 --output "$RUN\cluster_test_challenge.json" --device cuda
```

Cluster metrics include B-cubed precision, recall, and F1, pairwise cluster
metrics, and exact-partition accuracy. Use both pair and whole-document
results: pair scoring measures local decisions, while clustering measures the
final groups consumed by anonymization.

## 7. Build new training data from court decisions

Use this route when creating a new experiment rather than reproducing the
saved split. First prepare the source documents, then assign document splits
before fitting the OCR spacing model or running NER. This keeps text used to
fit the cleaner out of validation and test. Process each split separately and
carry the same assignment file through entity-linking-data construction.

```powershell
$NER = "D:\models\ner"

vdt prepare-court --output-dir datasets\court_documents

vdt split-documents `
  --input-file datasets\court_documents\documents.jsonl `
  --output-dir datasets\court_split `
  --validation-ratio 0.1 --test-ratio 0.1 --seed source-split-001

vdt train-ocr-ngram `
  --input-file datasets\court_split\documents\train.jsonl `
  --text-field markdown --output-file outputs\ocr_char_trigram.json

vdt run-ner `
  --model-source local --model-path "$NER" --local-files-only `
  --input-file datasets\court_split\documents\train.jsonl `
  --output-file outputs\ner_train.jsonl `
  --ocr-ngram-model outputs\ocr_char_trigram.json --entity-types PER,LOC,ORG --device auto

vdt run-ner `
  --model-source local --model-path "$NER" --local-files-only `
  --input-file datasets\court_split\documents\validation.jsonl `
  --output-file outputs\ner_validation.jsonl `
  --ocr-ngram-model outputs\ocr_char_trigram.json --entity-types PER,LOC,ORG --device auto

vdt run-ner `
  --model-source local --model-path "$NER" --local-files-only `
  --input-file datasets\court_split\documents\test.jsonl `
  --output-file outputs\ner_test.jsonl `
  --ocr-ngram-model outputs\ocr_char_trigram.json --entity-types PER,LOC,ORG --device auto

vdt reconstruct `
  --input-file datasets\court_split\documents\train.jsonl `
  --ner-file outputs\ner_train.jsonl `
  --output-file outputs\reconstructed_train.jsonl `
  --links-file outputs\links_train.jsonl --maps-file outputs\maps_train.jsonl

vdt reconstruct `
  --input-file datasets\court_split\documents\validation.jsonl `
  --ner-file outputs\ner_validation.jsonl `
  --output-file outputs\reconstructed_validation.jsonl `
  --links-file outputs\links_validation.jsonl --maps-file outputs\maps_validation.jsonl

vdt reconstruct `
  --input-file datasets\court_split\documents\test.jsonl `
  --ner-file outputs\ner_test.jsonl `
  --output-file outputs\reconstructed_test.jsonl `
  --links-file outputs\links_test.jsonl --maps-file outputs\maps_test.jsonl

vdt filter `
  --synthetic-file outputs\reconstructed_train.jsonl --links-file outputs\links_train.jsonl `
  --maps-input outputs\maps_train.jsonl --clean-output outputs\clean_train.jsonl `
  --maps-output outputs\clean_maps_train.jsonl --report-file outputs\quality_train.json

vdt filter `
  --synthetic-file outputs\reconstructed_validation.jsonl --links-file outputs\links_validation.jsonl `
  --maps-input outputs\maps_validation.jsonl --clean-output outputs\clean_validation.jsonl `
  --maps-output outputs\clean_maps_validation.jsonl --report-file outputs\quality_validation.json

vdt filter `
  --synthetic-file outputs\reconstructed_test.jsonl --links-file outputs\links_test.jsonl `
  --maps-input outputs\maps_test.jsonl --clean-output outputs\clean_test.jsonl `
  --maps-output outputs\clean_maps_test.jsonl --report-file outputs\quality_test.json

# Preserve the same document-to-split assignment for linking-data construction.
Get-Content outputs\clean_train.jsonl, outputs\clean_validation.jsonl, outputs\clean_test.jsonl |
  Set-Content -Encoding utf8 outputs\clean_all.jsonl
Get-Content outputs\clean_maps_train.jsonl, outputs\clean_maps_validation.jsonl, outputs\clean_maps_test.jsonl |
  Set-Content -Encoding utf8 outputs\clean_maps_all.jsonl

vdt build-raw-linking-data `
  --clean-input outputs\clean_all.jsonl `
  --maps-input outputs\clean_maps_all.jsonl `
  --output-dir outputs\linking_dataset `
  --split-assignments datasets\court_split\split_assignments.jsonl `
  --person-collision-rate 0.30 --collision-eval-rate 1.0 `
  --seed company-run-001
```

Train and evaluate the new natural-plus-augmentation dataset with:

```powershell
vdt train-linker `
  --train-pairs outputs\linking_dataset\pairs\train.jsonl `
  --validation-pairs outputs\linking_dataset\pairs\validation.jsonl `
  --test-pairs outputs\linking_dataset\pairs\test.jsonl `
  --output-dir outputs\linker_new\pairwise_model --model-name "$BASE" `
  --finetune-mode fft --epochs 3 --batch-size 8 --gradient-accumulation 2 `
  --max-length 256 --num-workers 0 --seed 42 --device cuda --fp16 --amp-dtype fp16

vdt eval-linker `
  --checkpoint outputs\linker_new\pairwise_model\best_model.pt --model-name "$BASE" `
  --pairs outputs\linking_dataset\pairs\test.jsonl `
  --collision-pairs outputs\linking_dataset\pairs\test_collision_eval.jsonl `
  --output outputs\linker_new\pairwise_test.json --device cuda

vdt eval-clusters `
  --checkpoint outputs\linker_new\pairwise_model\best_model.pt --model-name "$BASE" `
  --documents outputs\linking_dataset\documents\validation.jsonl `
  --maps outputs\linking_dataset\maps\validation.jsonl `
  --select-threshold --output outputs\linker_new\cluster_validation.json --device cuda

vdt eval-clusters `
  --checkpoint outputs\linker_new\pairwise_model\best_model.pt --model-name "$BASE" `
  --documents outputs\linking_dataset\documents\test.jsonl `
  --maps outputs\linking_dataset\maps\test.jsonl `
  --threshold-file outputs\linker_new\cluster_validation.json `
  --output outputs\linker_new\cluster_test.json --device cuda
```

The natural test and forced-collision diagnostic stay separate. The new-data
route creates the 30% train-only collision augmentation configured above and
separate forced-collision validation/test diagnostics. It does not
automatically reproduce the saved experiment’s four-cell balanced 20% hard
pair mixture. For a like-for-like reproduction, use the saved
`challenge_dataset/pairs/train_balanced_independent.jsonl` described in
Section 6 and retain its accompanying challenge test documents/maps.

The `split-documents` manifest is the source of truth for all later splits.
The raw-data builder checks that each retained document ID appears in it and
uses the same train/validation/test assignment.

## 8. What is and is not included

Included in this checkout: Python package, CLI, local FastAPI service, DOCX
adapter, current entity-linking and anonymization code, and this runbook.
Separate transfer artifacts: NER weights, base encoder weights, linker
checkpoint and tokenizer, optional OCR model, and the dataset splits used for
the saved experiment. The API does not use the experimental document-memory
checkpoint. It also does not promise to anonymize every possible Word object;
unsupported DOCX paragraph structures return an error for review.

The clean source snapshot is distributed as
`dist/VDT-company-handoff-0.3.0.zip`. It contains the package source,
`pyproject.toml`, the README, and this runbook, with local datasets, checkpoints,
Kaggle work files, and private documents excluded. Extract it on the target
machine and follow Section 4. The source bundle does not include Python
dependencies or model/data weights.

Before handing this to another team, make a named source-control commit that
contains the code and runbook, transfer model/data artifacts through the
company’s approved storage, and give the recipient the commit ID plus
checksums for each transferred artifact. The current checkout contains other
local research changes, so do not treat an uncommitted working copy as a
versioned release.
