# Vietnamese court-document anonymization and entity linking

This run guide assumes you run from the existing checkout at C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization. All command examples use absolute paths rooted there. The handoff ZIP stores the model and experiment assets; the steps below extract only those assets into this checkout.

## 1. Install and extract the bundled assets

The source code is already in C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization. Install the project in a virtual environment, then extract only the model and experiment assets from the handoff ZIP into this checkout.

~~~powershell
Set-Location -LiteralPath "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization"
py -3.11 -m venv "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\.venv"
& "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -e "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization[local,data,training]"
~~~

Extract the asset folders from the ZIP without replacing this source checkout:

~~~powershell
@'
from pathlib import Path
from zipfile import ZipFile

bundle = Path(r"C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\dist\court-anonymization-handoff-20261028.zip")
project = Path(r"C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization")
prefix = "court-anonymization/experiment_assets/"
with ZipFile(bundle) as archive:
    for name in archive.namelist():
        if not name.startswith(prefix) or name.endswith("/"):
            continue
        relative = Path(*name[len(prefix):].split("/"))
        target = project / "experiment_assets" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(archive.read(name))
print("Experiment assets extracted.")
'@ | py -3.11 -
~~~

For a GPU workstation, install the company-approved PyTorch build in the virtual environment before installing the project extras. Run project commands from C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization.

Bundled model paths:

- NER and linker encoder: C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base
- Linker checkpoint and tokenizer: C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\linker
- OCR spacing model: C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ocr_char_trigram.json

NER and the linker use the same NlpHUST model directory. Keep the linker's tokenizer folder beside best_model.pt.

## 2. Anonymize a document

The local script accepts DOCX, UTF-8 TXT, JSON, and JSONL. The input and output extensions must match.

Replace the example input filename with the absolute path to your document. Create the output folder first:

~~~powershell
New-Item -ItemType Directory -Force -Path "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\anonymized" | Out-Null
~~~

### DOCX

~~~powershell
python -m src.local_cli --input "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\input\decision.docx" --output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\anonymized\decision_anonymized.docx" --ner-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --linker-base-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --linker-checkpoint "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\linker\best_model.pt" --ocr-ngram-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ocr_char_trigram.json" --device auto
~~~

### TXT

TXT files are read and written as UTF-8.

~~~powershell
python -m src.local_cli --input "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\input\decision.txt" --output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\anonymized\decision_anonymized.txt" --ner-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --linker-base-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --linker-checkpoint "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\linker\best_model.pt" --ocr-ngram-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ocr_char_trigram.json" --device auto
~~~

### JSON

JSON can contain a text string, one object, or an array of objects. For objects, the runner detects text, content, or markdown by default. Use --text-field when the document text is in a different field.

~~~powershell
python -m src.local_cli --input "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\input\decision.json" --output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\anonymized\decision_anonymized.json" --text-field "markdown" --ner-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --linker-base-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --linker-checkpoint "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\linker\best_model.pt" --ocr-ngram-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ocr_char_trigram.json" --device auto
~~~

### JSONL

JSONL processes one document per non-empty line and reuses the loaded models across the batch.

~~~powershell
python -m src.local_cli --input "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\input\decisions.jsonl" --output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\anonymized\decisions_anonymized.jsonl" --text-field "markdown" --ner-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --linker-base-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --linker-checkpoint "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\linker\best_model.pt" --ocr-ngram-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ocr_char_trigram.json" --device auto
~~~

For JSON and JSONL, metadata fields are preserved and only the selected text field is replaced. The default review report is written beside the output with .report.json appended. If a record needs review, the runner withholds the anonymized output and writes the report. Existing files are not overwritten unless --overwrite is supplied.

## 3. Reproduce the packaged linker experiment

Use this path to reproduce the experiment represented by the bundled challenge training file and natural validation/test splits. It does not download or reconstruct source decisions.

The natural dataset has 8,206 document-disjoint items: 6,620 train, 793 validation, and 793 test. The training file contains 33,600 pairs, including the balanced challenge mixture. Separate challenge test pairs and challenge documents are included.

Extract the natural split files once:

~~~powershell
Expand-Archive -LiteralPath "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\raw_dataset_bundle.zip" -DestinationPath "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\raw_dataset"
~~~

Train a new checkpoint in a new output directory:

~~~powershell
python -m src.entity_linking.training --train-pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\challenge_dataset\pairs\train_balanced_independent.jsonl" --validation-pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\raw_dataset\pairs\validation.jsonl" --test-pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\raw_dataset\pairs\test.jsonl" --output-dir "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\reproduction_20261028\pairwise_model" --model-name "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --finetune-mode fft --epochs 3 --batch-size 8 --gradient-accumulation 2 --max-length 256 --num-workers 0 --seed 42 --device cuda --fp16 --amp-dtype fp16
~~~

For a CPU-only machine, use --device cpu --no-fp16; full fine-tuning will take substantially longer. This writes a new checkpoint and does not replace the one in the bundle.

Evaluate natural pairs and the held-out challenge pairs separately:

~~~powershell
python -m src.entity_linking.evaluate --checkpoint "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\reproduction_20261028\pairwise_model\best_model.pt" --model-name "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\raw_dataset\pairs\test.jsonl" --collision-pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\challenge_dataset\pairs\test_independent_challenge.jsonl" --output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\reproduction_20261028\pairwise_test.json" --device cuda --amp-dtype fp16
~~~

Evaluate document clusters on natural test documents:

~~~powershell
python -m src.entity_linking.cluster_evaluate --checkpoint "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\reproduction_20261028\pairwise_model\best_model.pt" --model-name "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --documents "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\raw_dataset\documents\test.jsonl" --maps "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\raw_dataset\maps\test.jsonl" --threshold 0.10 --output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\reproduction_20261028\cluster_test_natural.json" --device cuda
~~~

Evaluate document clusters on the held-out challenge documents:

~~~powershell
python -m src.entity_linking.cluster_evaluate --checkpoint "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\reproduction_20261028\pairwise_model\best_model.pt" --model-name "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --documents "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\challenge_dataset\documents\test_independent_challenge.jsonl" --maps "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\challenge_dataset\maps\test_independent_challenge.jsonl" --threshold 0.10 --output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\reproduction_20261028\cluster_test_challenge.json" --device cuda
~~~

To evaluate the checkpoint already in the bundle instead of a newly trained one, use C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\linker\best_model.pt as the checkpoint path in both evaluation commands.

## 4. Build a new training dataset from source decisions

Use this longer path only when creating a new experiment. It prepares source decisions, splits by document, learns an OCR spacing cleaner, runs NER, reconstructs pseudonymous training text and entity maps, filters low-quality reconstructions, then extracts mention pairs.

### 4.1 Prepare and split the documents

The preparation module downloads document Parquet shards from tmquan/congbobanan-toaan-gov-vn. If access requires a Hugging Face token, configure it in the process environment before running. For an existing local shard, add --local-parquet and its absolute path to the preparation command.

~~~powershell
python -m src.data.prepare --output-dir "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_002\prepared"
python -m src.data.split_documents --input-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_002\prepared\documents.jsonl" --output-dir "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_002\split" --text-field markdown --id-field case_id --validation-ratio 0.1 --test-ratio 0.1 --seed source-split-002
~~~

The split is 80% train, 10% validation, and 10% test. Split whole documents before fitting the cleaner or running NER.

### 4.2 Train the OCR cleaner and run NER

The OCR cleaner is a character-trigram model, not a neural language model. Fit it on training documents only. It makes a normalized view for NER and maps mention offsets back to the source text.

~~~powershell
python -m src.pipeline.train_ocr_ngram --input-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_002\split\documents\train.jsonl" --text-field markdown --output-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\ocr_char_trigram.json"

python -m src.pipeline.ner --model-source local --model-path "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --input-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_002\split\documents\train.jsonl" --output-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\ner_train.jsonl" --ocr-ngram-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\ocr_char_trigram.json" --text-field markdown --id-field case_id --entity-types PER,LOC,ORG --device auto

python -m src.pipeline.ner --model-source local --model-path "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --input-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_002\split\documents\validation.jsonl" --output-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\ner_validation.jsonl" --ocr-ngram-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\ocr_char_trigram.json" --text-field markdown --id-field case_id --entity-types PER,LOC,ORG --device auto

python -m src.pipeline.ner --model-source local --model-path "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --input-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_002\split\documents\test.jsonl" --output-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\ner_test.jsonl" --ocr-ngram-model "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\ocr_char_trigram.json" --text-field markdown --id-field case_id --entity-types PER,LOC,ORG --device auto
~~~

### 4.3 Reconstruct and quality-filter

Reconstruction applies the project marker and document rules to produce synthetic names and locations, entity maps, and an audit file. It does not restore hidden real identities. Filtering removes reconstructions that fail quality checks and keeps maps aligned.

~~~powershell
python -m src.pipeline.reconstruct --input-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_002\split\documents\train.jsonl" --ner-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\ner_train.jsonl" --output-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\reconstructed_train.jsonl" --links-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\links_train.jsonl" --maps-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\maps_train.jsonl"
python -m src.data.filtering --synthetic-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\reconstructed_train.jsonl" --links-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\links_train.jsonl" --maps-input "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\maps_train.jsonl" --clean-output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\clean_train.jsonl" --maps-output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\clean_maps_train.jsonl" --report-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\quality_train.json"

python -m src.pipeline.reconstruct --input-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_002\split\documents\validation.jsonl" --ner-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\ner_validation.jsonl" --output-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\reconstructed_validation.jsonl" --links-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\links_validation.jsonl" --maps-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\maps_validation.jsonl"
python -m src.data.filtering --synthetic-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\reconstructed_validation.jsonl" --links-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\links_validation.jsonl" --maps-input "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\maps_validation.jsonl" --clean-output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\clean_validation.jsonl" --maps-output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\clean_maps_validation.jsonl" --report-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\quality_validation.json"

python -m src.pipeline.reconstruct --input-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_002\split\documents\test.jsonl" --ner-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\ner_test.jsonl" --output-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\reconstructed_test.jsonl" --links-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\links_test.jsonl" --maps-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\maps_test.jsonl"
python -m src.data.filtering --synthetic-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\reconstructed_test.jsonl" --links-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\links_test.jsonl" --maps-input "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\maps_test.jsonl" --clean-output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\clean_test.jsonl" --maps-output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\clean_maps_test.jsonl" --report-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\quality_test.json"
~~~

Join the cleaned documents and maps in the same split order:

~~~powershell
Get-Content "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\clean_train.jsonl", "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\clean_validation.jsonl", "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\clean_test.jsonl" | Set-Content -Encoding utf8 "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\clean_all.jsonl"
Get-Content "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\clean_maps_train.jsonl", "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\clean_maps_validation.jsonl", "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\clean_maps_test.jsonl" | Set-Content -Encoding utf8 "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\clean_maps_all.jsonl"
~~~

### 4.4 Extract pairs, train, and evaluate

The pair builder creates positives from mentions with the same reconstructed entity ID and negatives from different IDs. It targets one negative per positive and adds same-surface, different-person collisions to eligible training documents.

~~~powershell
python -m src.entity_linking.raw_dataset --clean-input "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\clean_all.jsonl" --maps-input "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\clean_maps_all.jsonl" --output-dir "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\linking_dataset" --split-assignments "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\datasets\court_experiment_002\split\split_assignments.jsonl" --negatives-per-positive 1.0 --person-collision-rate 0.30 --collision-eval-rate 1.0 --seed fresh-build-002

python -m src.entity_linking.training --train-pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\linking_dataset\pairs\train.jsonl" --validation-pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\linking_dataset\pairs\validation.jsonl" --test-pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\linking_dataset\pairs\test.jsonl" --output-dir "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\trained_model\pairwise_model" --model-name "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --finetune-mode fft --epochs 3 --batch-size 8 --gradient-accumulation 2 --max-length 256 --num-workers 0 --seed 42 --device cuda --fp16 --amp-dtype fp16

python -m src.entity_linking.evaluate --checkpoint "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\trained_model\pairwise_model\best_model.pt" --model-name "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\linking_dataset\pairs\test.jsonl" --collision-pairs "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\linking_dataset\pairs\test_collision_eval.jsonl" --output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\trained_model\pairwise_test.json" --device cuda --amp-dtype fp16

python -m src.entity_linking.cluster_evaluate --checkpoint "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\trained_model\pairwise_model\best_model.pt" --model-name "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --documents "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\linking_dataset\documents\validation.jsonl" --maps "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\linking_dataset\maps\validation.jsonl" --select-threshold --output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\trained_model\cluster_validation.json" --device cuda

python -m src.entity_linking.cluster_evaluate --checkpoint "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\trained_model\pairwise_model\best_model.pt" --model-name "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\models\ner_linker_base" --documents "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\linking_dataset\documents\test.jsonl" --maps "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\linking_dataset\maps\test.jsonl" --threshold-file "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\trained_model\cluster_validation.json" --output "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\outputs\court_experiment_002\trained_model\cluster_test.json" --device cuda
~~~

This fresh-data recipe adds same-surface collision examples; it does not recreate the packaged experiment's balanced four-category challenge mix. Use Section 3 when comparing against that saved challenge setup.

## 5. How to interpret the stages

- Preparation downloads court decisions into JSONL files.
- OCR cleaning learns spacing patterns from training text and supplies a normalized view to NER.
- NER records person, location, and organization mentions with offsets mapped to the source text.
- Reconstruction makes synthetic training text and entity maps from already-anonymized decisions; it does not recover hidden original names.
- Quality filtering removes reconstructions that fail configured checks.
- Pair extraction creates same-entity positive pairs and different-entity negative pairs.
- Training fits the pair scorer. Pair evaluation measures mention links; cluster evaluation measures the final entity groups.
- The local anonymization command runs detection, linking, and replacement on a new file.

The bundle inventory and checksums are documented in C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\docs\ASSET_BUNDLE.md.
