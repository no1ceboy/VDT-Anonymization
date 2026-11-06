# Model and experiment assets

The current local handoff archive is
`C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\dist\court-anonymization-handoff-20261028.zip`.
It is a local transfer bundle and is not stored in GitHub. Its checksum is in
`C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\dist\court-anonymization-handoff-20261028.zip.sha256`.

The archive contains the source tree, the natural and challenge entity-linking
data, the trained pairwise linker checkpoint and tokenizer, the NER/linker
encoder, and the OCR spacing model. It excludes the private DOCX and Kaggle
credentials.

## What can be downloaded or rebuilt

- **NER/linker encoder:** downloadable from the public Hugging Face model
  `NlpHUST/ner-vietnamese-electra-base`. The bundle records the exact revision
  and checksums in `experiment_assets/models/ner_linker_base/MODEL_SOURCE.json`.
- **Source court decisions:** available from the public dataset sources used
  by the repository. The preparation commands can fetch them again, subject
  to current access and availability.
- **OCR spacing model:** reproducible by fitting the repository's character
  trigram model on the training split.
- **Reconstructed splits and challenge data:** reproducible from source data
  and the repository pipeline, but rebuilding requires rerunning preparation,
  NER, reconstruction, filtering, and pair generation.
- **Fine-tuned linker checkpoint:** this is a project-trained artifact, not
  the public base model. Keep the bundle or recover the saved Kaggle output to
  reuse that exact checkpoint; otherwise it must be retrained.

Therefore the archive is not required to get the source code from GitHub, but
it is the simplest way to preserve the exact experiment and avoid rebuilding
or retraining its assets.

## Extract the natural dataset

After extracting the archive's `experiment_assets` into the project checkout,
expand the raw dataset bundle once:

```powershell
Expand-Archive -LiteralPath "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\raw_dataset_bundle.zip" -DestinationPath "C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\raw_dataset"
```

The challenge dataset is already under
`C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\challenge_dataset`.
The natural and challenge run summaries are under
`C:\Users\Minh\Desktop\PROJECT\VDT-Anonymization\experiment_assets\model_run_results`.

See the README for the absolute-path commands to extract the archive's assets,
run anonymization, and reproduce or rebuild experiments.
