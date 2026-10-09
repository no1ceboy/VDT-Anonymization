# V4 location-hierarchy linker commands

Run from the repository root. V4 trains on original anonymized text with weak reconstruction-map labels (not human gold). High-level province/district/commune names and parent codes come from the bundled, pinned DVHCN administrative snapshot; Vietnamese address-unit cues handle lower-level aliases. NER type is authoritative: location rules only run for LOC, and ambiguous cases defer to the learned scorer. Natural pairs that contradict a high-confidence location rule are omitted, not relabeled.

Training is pairwise: the encoder/MLP learns mention-pair scores. Deployment applies that scorer to all same-type mention pairs in one document, then forms clusters with complete-link and location constraints; it is not trained with a direct cluster loss. These commands use existing V3 document/map splits and do not run NER or Kaggle. No package installation is needed.

```bash
VDT_DATA=outputs/entity_linking_v4_location_hierarchy_20261009
VDT_RUN=outputs/entity_linker_v4_location_hierarchy_fft

# Build V4 locally from existing V3 documents/maps; no NER/Kaggle rerun.
python3 -m src.vdt_anonymization.entity_linking.v4_dataset --v3-dir outputs/entity_linking_v3_location_strict_20260924 --output-dir "$VDT_DATA"

# Smoke test: verify V4 pair-schema/model compatibility with 100 train steps.
python3 -m src.vdt_anonymization.entity_linking.training --train-pairs "$VDT_DATA/pairs/train.jsonl" --validation-pairs "$VDT_DATA/pairs/validation.jsonl" --test-pairs "$VDT_DATA/pairs/test.jsonl" --output-dir outputs/entity_linker_v4_location_hierarchy_smoke --model-name NlpHUST/ner-vietnamese-electra-base --finetune-mode frozen --epochs 1 --max-train-steps 100 --max-eval-steps 20 --batch-size 16 --device cuda --fp16 --amp-dtype bf16

# Main run: full encoder fine-tuning; TensorBoard logs are written under VDT_RUN.
python3 -m src.vdt_anonymization.entity_linking.training --train-pairs "$VDT_DATA/pairs/train.jsonl" --validation-pairs "$VDT_DATA/pairs/validation.jsonl" --test-pairs "$VDT_DATA/pairs/test.jsonl" --output-dir "$VDT_RUN" --model-name NlpHUST/ner-vietnamese-electra-base --finetune-mode fft --epochs 3 --batch-size 32 --gradient-accumulation 1 --device cuda --fp16 --amp-dtype bf16

# Compare marker/surface heuristics on natural test pairs and controlled collisions.
python3 -m src.vdt_anonymization.entity_linking.baseline --pairs "$VDT_DATA/pairs/test.jsonl" --methods exact_marker marker_family exact_surface --output "$VDT_RUN/baseline_test.json"
python3 -m src.vdt_anonymization.entity_linking.baseline --pairs "$VDT_DATA/pairs/test_challenge.jsonl" --methods exact_marker marker_family exact_surface --output "$VDT_RUN/baseline_test_challenge.json"
python3 -m src.vdt_anonymization.entity_linking.evaluate --checkpoint "$VDT_RUN/best_model.pt" --pairs "$VDT_DATA/pairs/test.jsonl" --collision-pairs "$VDT_DATA/pairs/test_challenge.jsonl" --output "$VDT_RUN/pair_evaluation.json" --device cuda --fp16 --amp-dtype bf16

# Choose a clustering threshold on V4 validation documents/maps only.
python3 -m src.vdt_anonymization.entity_linking.cluster_evaluate --checkpoint "$VDT_RUN/best_model.pt" --documents "$VDT_DATA/documents/validation.jsonl" --maps "$VDT_DATA/maps/validation.jsonl" --select-threshold --output "$VDT_RUN/cluster_validation.json" --device cuda --amp-dtype bf16

# Final natural-document cluster evaluation on held-out V4 test documents/maps.
python3 -m src.vdt_anonymization.entity_linking.cluster_evaluate --checkpoint "$VDT_RUN/best_model.pt" --documents "$VDT_DATA/documents/test.jsonl" --maps "$VDT_DATA/maps/test.jsonl" --threshold-file "$VDT_RUN/cluster_validation.json" --output "$VDT_RUN/cluster_test.json" --device cuda --amp-dtype bf16

# Evaluate whole-document clustering on controlled marker-collision challenge documents.
python3 -m src.vdt_anonymization.entity_linking.cluster_evaluate --checkpoint "$VDT_RUN/best_model.pt" --documents "$VDT_DATA/documents/test_challenge.jsonl" --maps "$VDT_DATA/maps/test_challenge.jsonl" --threshold-file "$VDT_RUN/cluster_validation.json" --output "$VDT_RUN/cluster_test_challenge.json" --device cuda --amp-dtype bf16

# Deployment: each input JSONL row contains doc_id, text, and NER mentions [{start,end,label}].
python3 -m src.vdt_anonymization.entity_linking.inference --checkpoint "$VDT_RUN/best_model.pt" --input ner_documents.jsonl --output linked_documents.jsonl --threshold-file "$VDT_RUN/cluster_validation.json" --device cuda --amp-dtype bf16

# View training curves.
tensorboard --logdir "$VDT_RUN/tensorboard"
```

The smoke run checks model/data compatibility only. Use the full run for results. Pair metrics are diagnostics; cluster_test.json is the whole-document pipeline result. Both remain agreement with weak map labels, not independent accuracy. NER is supplied externally; location rules never relabel ORG as LOC. In Windows PowerShell, set $env:CUDA_VISIBLE_DEVICES="0" first if you need a specific GPU; --device cuda selects CUDA rather than CPU.
