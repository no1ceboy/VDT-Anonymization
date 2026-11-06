# Company handoff: Vietnamese court-document anonymization

## 1. Purpose and scope

This package takes a Vietnamese court decision, detects protected person,
location, and organization mentions, links mentions that refer to the same
entity, and prepares a law-based anonymization plan. The recommended company
workflow is a local command-line script: the document and model inference stay
on the company machine, and no server needs to be deployed.

The handoff has two separate uses:

1. **Process documents.** Install the code, place the model assets on the
   company machine, then run one command on DOCX, UTF-8 text, JSON, or JSONL.
2. **Reproduce the experiment.** Use the saved document-disjoint data splits,
   rebuild or fine-tune the linker, and evaluate both mention pairs and whole
   document clusters.

The local script uses the trained pairwise linker plus the production
clustering and legal anonymization code. The document-memory attention model
is a separate experiment and is not the model used by this script.

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

For DOCX, the script edits text in regular paragraphs, tables, headers, and
footers while retaining ordinary run formatting. It refuses unsupported
paragraph structures rather than returning a partially changed file.

## 3. Source and model assets

Install the repository itself as the code package. A source checkout or Python
wheel does not include model weights or court documents. The complete handoff
bundle includes the reconstructed linker datasets, pairwise linker checkpoint,
shared NER/linker encoder, and OCR spacing model. See
[ASSET_BUNDLE.md](ASSET_BUNDLE.md) for the bundle layout and extraction steps.

The packaged linker checkpoint is:

```text
experiment_assets/models/linker/best_model.pt
```

Its sibling `tokenizer/` directory is included. The checkpoint was fine-tuned
from `NlpHUST/ner-vietnamese-electra-base` in full fine-tuning mode. The
complete model files are in
`experiment_assets/models/ner_linker_base/` and are used by both NER and the
linker. The OCR spacing model is included at
`experiment_assets/models/ocr_char_trigram.json`.

Suggested machine layout:

```text
court-anonymization/
  experiment_assets/models/
    ner_linker_base/            # shared NER and linker encoder
    linker/
      best_model.pt
      tokenizer/
    ocr_char_trigram.json
```

The bundle includes `ASSET_CHECKSUMS.sha256` and the model revision metadata.
Do not include the private DOCX used for local review in a shared handoff
archive.

## 4. Install on a company machine

Use Python 3.11 (the project supports Python 3.10 or newer). Install the
company-approved PyTorch build first, selecting CPU or CUDA to match the
machine. Then install the document-processing and experiment dependencies:

```powershell
Set-Location -LiteralPath "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization"
py -3.11 -m venv "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\.venv"
& "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
# Install the company-approved CPU or CUDA PyTorch build here, if needed.
python -m pip install -e "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization[local,data,training]"
```

On Linux, create and activate a virtual environment from the repository root.
For a restricted network, prepare and scan a wheelhouse on an approved connected
machine, then install from that wheelhouse. Extract the complete handoff bundle
on the target machine. When running offline, configure `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` as operating-system environment settings.

## 5. Run the end-to-end script

The command loads all models locally, processes one document, and writes an
anonymized file plus a JSON report. DOCX formatting is retained for supported
paragraphs, tables, headers, and footers. The script does not upload or retain
the source document elsewhere.

```powershell
python -m src.local_cli `
  --input "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\input\decision.docx" `
  --output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\anonymized\decision_anonymized.docx" `
  --ner-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" `
  --linker-base-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" `
  --linker-checkpoint "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\linker\best_model.pt" `
  --ocr-ngram-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ocr_char_trigram.json" `
  --device auto
```

For plain text, use `.txt` for both `--input` and `--output`; it uses UTF-8.
JSON can be one object, an array of objects, or a top-level string. JSONL has
one document record per non-empty line. For object records, the runner
auto-detects `text`, `content`, or `markdown`; use `--text-field` for another
field. It preserves the rest of each record and replaces only the text field.
JSONL reuses the loaded models across records. If any record needs review, the
runner writes the report and withholds the complete output file. Omit
`--ocr-ngram-model` to skip spacing correction. The current
document-clustering threshold is 0.10 and is the default; override it only for
a documented experiment.

The command refuses to overwrite existing files unless `--overwrite` is set.
If the result needs review, it writes a report containing detected mentions
and the review reasons, exits with status 3, and does not produce a partially
anonymized document. Exit status 2 indicates an input, asset, or processing
error. Treat both the source and report as confidential because the report can
contain original entity text.

The HTTP API remains in the repository as an optional interface, but it is not
needed for this local workflow and does not need to be deployed on the company
server.

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

Extract the natural data and challenge data as described in
[ASSET_BUNDLE.md](ASSET_BUNDLE.md). The commands below use the resulting
absolute paths under
`C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets` and
write a new run under
`C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\reproduction_20261028`.
They do not overwrite the checkpoint included in the bundle.

### Fine-tune the pair scorer

The following matches the saved run’s central settings: full fine-tuning for
three epochs, 8 examples per device batch, gradient accumulation 2, maximum
length 256, seed 42. The base model must be a local folder when the machine is
offline.

```powershell
python -m src.entity_linking.training `
  --train-pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\challenge_dataset\pairs\train_balanced_independent.jsonl" `
  --validation-pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\raw_dataset\pairs\validation.jsonl" `
  --test-pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\raw_dataset\pairs\test.jsonl" `
  --output-dir "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\reproduction_20261028\pairwise_model" `
  --model-name "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" `
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
python -m src.entity_linking.evaluate `
  --checkpoint "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\reproduction_20261028\pairwise_model\best_model.pt" `
  --model-name "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" `
  --pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\raw_dataset\pairs\test.jsonl" `
  --collision-pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\challenge_dataset\pairs\test_independent_challenge.jsonl" `
  --output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\reproduction_20261028\pairwise_model\test_evaluation.json" `
  --device cuda --amp-dtype fp16
```

Evaluate clustering on natural documents and on the held-out challenge
documents separately. Keep the document threshold at the configured 0.10 for
reproduction; if selecting a new threshold, select it on validation only and
pass that validation result to the test run.

```powershell
python -m src.entity_linking.cluster_evaluate `
  --checkpoint "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\reproduction_20261028\pairwise_model\best_model.pt" `
  --model-name "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" `
  --documents "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\raw_dataset\documents\test.jsonl" --maps "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\raw_dataset\maps\test.jsonl" `
  --threshold 0.10 --output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\reproduction_20261028\cluster_test_natural.json" --device cuda

python -m src.entity_linking.cluster_evaluate `
  --checkpoint "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\reproduction_20261028\pairwise_model\best_model.pt" `
  --model-name "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" `
  --documents "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\challenge_dataset\documents\test_independent_challenge.jsonl" `
  --maps "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\challenge_dataset\maps\test_independent_challenge.jsonl" `
  --threshold 0.10 --output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\reproduction_20261028\cluster_test_challenge.json" --device cuda
```

Cluster metrics include B-cubed precision, recall, and F1, pairwise cluster
metrics, and exact-partition accuracy. Use both pair and whole-document
results: pair scoring measures local decisions, while clustering measures the
final groups consumed by anonymization.

## 7. Build a new dataset and train from source

Use this workflow when creating a new experiment. It runs every stage from
source decisions to linker evaluation. Run the commands from the repository root in PowerShell. They invoke the
Python modules under `src/` directly; there is no console shortcut in these
commands.

The modular source-preparation command in this checkout reads
`tmquan/congbobanan-toaan-gov-vn` and expects `case_id` plus `markdown` in its
Parquet shards. The separate Kaggle 10k builder reads
`tmquan/cbba-toaan-gov-vn`; see its command in the README. These ingestion paths
use different sources and should not be mixed within one run.

Start with new, unused output folders. The commands below use the absolute
paths under this checkout. Change the experiment folder names if they already
exist.

### 7.1 Prepare and split source documents

The preparation step downloads the Parquet shards and writes source text as
JSONL. It does not clean, reconstruct, or split the documents. The next step
assigns whole documents to train, validation, or test, before fitting any
corpus-derived cleaner or running NER.

```powershell
python -m src.data.prepare `
  --output-dir "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_001\prepared"

python -m src.data.split_documents `
  --input-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_001\prepared\documents.jsonl" `
  --output-dir "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_001\split" `
  --text-field markdown --id-field case_id `
  --validation-ratio 0.1 --test-ratio 0.1 --seed source-split-001
```

Optional: if the source collection is too large for the first run, make a
deterministic marker-bearing subset. Run preparation first, then this curation
command, then split the curated file instead of
`prepared\documents.jsonl`:

```powershell
python -m src.data.curate `
  --input-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_001\prepared\documents.jsonl" `
  --output-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_001\curated\documents.jsonl" `
  --per-stratum 100 --min-chars 500
```

For the curated run, use
`C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_001\curated\documents.jsonl`
as `--input-file` in the split command above.

Curation selects documents with anonymization markers across case and document
types. It changes the source population; record that choice in experiment
results. Do not curate separately after splitting, because that would discard
the original split manifest needed by the linker-data builder.

### 7.2 Fit the OCR spacing model and run NER

The OCR cleaner is a character-trigram model, not a neural language model. Fit
it on training text only. NER then runs independently on all three splits using
the same cleaner; it maps predicted mention offsets back to the unchanged source
text.

```powershell
python -m src.pipeline.train_ocr_ngram `
  --input-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_001\split\documents\train.jsonl" `
  --text-field markdown --output-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\ocr_char_trigram.json"

python -m src.pipeline.ner `
  --model-source local --model-path "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" `
  --input-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_001\split\documents\train.jsonl" `
  --output-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\ner_train.jsonl" `
  --ocr-ngram-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\ocr_char_trigram.json" `
  --text-field markdown --id-field case_id --entity-types PER,LOC,ORG --device auto

python -m src.pipeline.ner `
  --model-source local --model-path "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" `
  --input-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_001\split\documents\validation.jsonl" `
  --output-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\ner_validation.jsonl" `
  --ocr-ngram-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\ocr_char_trigram.json" `
  --text-field markdown --id-field case_id --entity-types PER,LOC,ORG --device auto

python -m src.pipeline.ner `
  --model-source local --model-path "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" `
  --input-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_001\split\documents\test.jsonl" `
  --output-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\ner_test.jsonl" `
  --ocr-ngram-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\ocr_char_trigram.json" `
  --text-field markdown --id-field case_id --entity-types PER,LOC,ORG --device auto
```

### 7.3 Reconstruct synthetic text, then filter it

Reconstruction applies the repository's marker and document rules, including
its entity-aware post-NER spacing repair, to create
pseudonymous names and locations, along with aligned entity maps and an audit
file. It does not recover the hidden original identities. Filtering uses the
reconstruction audit to exclude records with low quality or unresolved review
issues while keeping each split's maps aligned with its selected documents.

```powershell
python -m src.pipeline.reconstruct `
  --input-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_001\split\documents\train.jsonl" `
  --ner-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\ner_train.jsonl" `
  --output-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\reconstructed_train.jsonl" `
  --links-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\links_train.jsonl" --maps-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\maps_train.jsonl"

python -m src.pipeline.reconstruct `
  --input-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_001\split\documents\validation.jsonl" `
  --ner-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\ner_validation.jsonl" `
  --output-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\reconstructed_validation.jsonl" `
  --links-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\links_validation.jsonl" --maps-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\maps_validation.jsonl"

python -m src.pipeline.reconstruct `
  --input-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_001\split\documents\test.jsonl" `
  --ner-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\ner_test.jsonl" `
  --output-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\reconstructed_test.jsonl" `
  --links-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\links_test.jsonl" --maps-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\maps_test.jsonl"

python -m src.data.filtering `
  --synthetic-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\reconstructed_train.jsonl" --links-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\links_train.jsonl" `
  --maps-input "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\maps_train.jsonl" --clean-output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\clean_train.jsonl" `
  --maps-output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\clean_maps_train.jsonl" --report-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\quality_train.json"

python -m src.data.filtering `
  --synthetic-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\reconstructed_validation.jsonl" --links-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\links_validation.jsonl" `
  --maps-input "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\maps_validation.jsonl" --clean-output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\clean_validation.jsonl" `
  --maps-output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\clean_maps_validation.jsonl" --report-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\quality_validation.json"

python -m src.data.filtering `
  --synthetic-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\reconstructed_test.jsonl" --links-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\links_test.jsonl" `
  --maps-input "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\maps_test.jsonl" --clean-output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\clean_test.jsonl" `
  --maps-output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\clean_maps_test.jsonl" --report-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\quality_test.json"
```

Combine the three filtered splits only for the linker-data builder; its split
manifest keeps each document in its assigned partition. The same ordering is
used for documents and maps:

```powershell
Get-Content "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\clean_train.jsonl", "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\clean_validation.jsonl", "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\clean_test.jsonl" |
  Set-Content -Encoding utf8 "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\clean_all.jsonl"
Get-Content "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\clean_maps_train.jsonl", "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\clean_maps_validation.jsonl", "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\clean_maps_test.jsonl" |
  Set-Content -Encoding utf8 "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\clean_maps_all.jsonl"
```

### 7.4 Create mention pairs, train, and evaluate

The pair builder creates positive pairs from mentions with the same
reconstructed entity ID and negative pairs from different entity IDs. It
augments 30% of eligible training documents with same-surface person collisions;
validation and test remain natural, with separate forced-collision pair files
for diagnostic evaluation. It targets one negative per positive pair by default.

```powershell
python -m src.entity_linking.raw_dataset `
  --clean-input "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\clean_all.jsonl" `
  --maps-input "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\clean_maps_all.jsonl" `
  --output-dir "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\linking_dataset" `
  --split-assignments "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_001\split\split_assignments.jsonl" `
  --negatives-per-positive 1.0 `
  --person-collision-rate 0.30 --collision-eval-rate 1.0 `
  --seed company-run-001

# Review retained-document counts, pair counts, split counts, and collision counts.
Get-Content "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\quality_train.json" -Raw
Get-Content "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\quality_validation.json" -Raw
Get-Content "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\quality_test.json" -Raw
Get-Content "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\linking_dataset\manifest.json" -Raw

python -m src.entity_linking.training `
  --train-pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\linking_dataset\pairs\train.jsonl" `
  --validation-pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\linking_dataset\pairs\validation.jsonl" `
  --test-pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\linking_dataset\pairs\test.jsonl" `
  --output-dir "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\trained_model\pairwise_model" --model-name "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" `
  --finetune-mode fft --epochs 3 --batch-size 8 --gradient-accumulation 2 `
  --max-length 256 --num-workers 0 --seed 42 --device cuda --fp16 --amp-dtype fp16

python -m src.entity_linking.evaluate `
  --checkpoint "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\trained_model\pairwise_model\best_model.pt" --model-name "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" `
  --pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\linking_dataset\pairs\test.jsonl" `
  --collision-pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\linking_dataset\pairs\test_collision_eval.jsonl" `
  --output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\trained_model\pairwise_test.json" --device cuda --amp-dtype fp16

python -m src.entity_linking.cluster_evaluate `
  --checkpoint "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\trained_model\pairwise_model\best_model.pt" --model-name "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" `
  --documents "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\linking_dataset\documents\validation.jsonl" `
  --maps "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\linking_dataset\maps\validation.jsonl" `
  --select-threshold --output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\trained_model\cluster_validation.json" --device cuda

python -m src.entity_linking.cluster_evaluate `
  --checkpoint "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\trained_model\pairwise_model\best_model.pt" --model-name "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" `
  --documents "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\linking_dataset\documents\test.jsonl" `
  --maps "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\linking_dataset\maps\test.jsonl" `
  --threshold-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\trained_model\cluster_validation.json" `
  --output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_001\trained_model\cluster_test.json" --device cuda
```

This fresh-data route is not identical to the saved experiment in Section 6:
it makes a 30% training-only collision augmentation and separate collision
pair diagnostics. It does not create the saved experiment's balanced
four-category challenge mixture or a challenge-document cluster test. Keep
those comparisons separate when reporting results. The forced-collision file
contains only different-person, identical-surface pairs; report its false-link
rate, not an F1 score.

## 8. What is included

The complete handoff archive contains the source tree, natural and challenge
entity-linking datasets, the matching linker checkpoint and tokenizer, the
shared NER/linker encoder model, and the optional OCR spacing model. The
supplied private DOCX and Kaggle credentials are excluded. Python dependencies
are listed in `requirements.txt` and must be installed on the target machine.
See [ASSET_BUNDLE.md](ASSET_BUNDLE.md) for the archive layout, checksums, and
commands to use the bundled assets. The API does not use the experimental
document-memory checkpoint. Unsupported DOCX paragraph structures return an
error for review.

The ZIP is a transfer bundle, not a versioned software release. Before
production use, make a named source-control commit and follow the company’s
approved process for reviewing and transferring the bundle.
