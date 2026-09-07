# Vietnamese legal synthetic reconstruction

Run commands from the repository root. Existing NER JSONL and notebook commands remain supported.
The reconstruction and HTML inspector use only Python's standard library. NER dependencies are in `requirements.txt`.

## Pipeline

1. Read NER and recover publication markers with document rules.
2. Identify the encoding: whole-name alias, visible name initial, numbered initial, procedural code, or address/organization component.
3. Link compatible visible name prefixes, explicit participant declarations and complete procedural codes within each document.
4. Optionally ask Gemini to classify unknown procedural participants or resolve ambiguous references against supplied candidates.
5. Generate consistent replacements and verify exact recovery of the source using the replacement map.

`NLQ 1` and `NLQ1` share one code; `NLQ2` is distinct. `NLQ` means a related party, which can be a person or organization. Unknown types remain unchanged until evidence resolves them. `NLC` means witness. `H 12` and `H12` normalize together; document identifiers such as `A01` are excluded from the generic initial rule.

A complete declared alias does not constrain a synthetic given-name initial. A visible name prefix does. Roles provide context; multiple people can share a procedural role, and one person can hold several roles. Distance alone never resolves competing names.

The rules follow [Article 7 of NQ 03/2017/NQ-HĐTP](https://thuvienphapluat.vn/van-ban/cong-nghe-thong-tin/nghi-quyet-03-2017-nq-hdtp-cong-bo-ban-an-quyet-dinh-tren-cong-thong-tin-dien-tu-toa-an-343156.aspx) and [section 2 of CV 144/TANDTC-PC](https://congbobanan.toaan.gov.vn/1t2atcvn/thi-hanh-nq-so-03-2017/nq-hdtp). This is synthetic reconstruction, not recovery of hidden identities. Existing unmasked-name evidence handling remains supported; these outputs are not a guarantee that every identity is fictitious. Omitted source details cannot be recovered.

## Run locally or in a notebook shell cell

```bash
python src/build_synthetic_unanonymized.py --input-file datasets/legal_test.jsonl --ner-file outputs/nlphust_legal_sample.jsonl --limit 20
```

To use Gemini, set `GOOGLE_API_KEY` or `GEMINI_API_KEY` in the process environment. Alternatively, create a local `.env` containing `GOOGLE_API_KEY=...` and pass `--env-file .env`. These files are ignored by Git. No Google SDK installation is needed.

```bash
python src/build_synthetic_unanonymized.py --input-file datasets/legal_test.jsonl --ner-file outputs/nlphust_legal_sample.jsonl --limit 20 --llm-provider gemini --llm-model YOUR_GEMINI_MODEL_ID --llm-max-calls 20
```

Replace the model placeholder with a model available to your Google project. Gemini sends the selected document excerpts to Google's API. The call budget applies across the whole run; `--llm-max-calls 0` makes no requests. Timeouts, failed requests, invalid choices and abstentions leave mentions unresolved and appear in the audit. There are no automatic retries. The implementation uses Google's [generateContent REST API](https://ai.google.dev/api/generate-content).

The resolver requires a valid candidate ID and an exact supporting quote from supplied text. These checks catch malformed responses and invented quotes, but do not prove that the reasoning is correct. Review accepted links before treating them as ground truth. No self-reported confidence score is treated as a calibrated probability.

## Outputs and review

Only three reconstruction outputs are written, using the existing filenames (reruns overwrite them):

- `outputs/synthetic_unanonymized.jsonl`: source, reconstruction and per-document counts.
- `outputs/entity_links.jsonl`: entities, encoding schemes, procedural roles and per-mention LLM decisions/evidence.
- `outputs/replacement_maps.jsonl`: entity mappings and applied replacements, including original text and source/synthetic offsets.

Use `--output-file`, `--links-file`, and `--maps-file` to change these paths. The input NER file is preserved. LLM audit data are embedded, with no extra debug output files.

```bash
python src/inspect_entity_links.py --source-file datasets/legal_test.jsonl --links-file outputs/entity_links.jsonl --doc-id 1003850 --output-file outputs/document_review.html
```

The inspector shows encoding, role, unresolved status and expandable LLM decisions. Round-trip verification checks replacement integrity, not semantic correctness. `skipped_overlapping_replacements` reports remaining overlap conflicts. Rules are heuristics and cannot certify NER recall or geographic plausibility.

```bash
python -B -m unittest discover -s tests -v
```
