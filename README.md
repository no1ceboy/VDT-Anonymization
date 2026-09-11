# Vietnamese legal synthetic reconstruction

## Prepare the court-document collection

`src/prepare_court_dataset.py` downloads only the document Parquet shards from
`tmquan/congbobanan-toaan-gov-vn`, pins the resolved repository revision, and
converts them to one UTF-8 JSONL with the existing `case_id` and `markdown`
fields. All source columns are retained. It requires `pyarrow` and, for online
downloads, `huggingface_hub`; neither is needed to read the prepared JSONL.

```powershell
python src/prepare_court_dataset.py --output-dir datasets/court_documents
```

Set `HF_TOKEN` in the process environment if repository access requires a token.
Use `--max-shards 1` for a first-shard download, or `--local-parquet PATH` for
offline conversion of existing shards. Existing prepared outputs are never
overwritten. Failed conversion leaves a `.partial` file, which is not pipeline
input. The manifest records source hashes, revision (when known), row counts,
and skipped empty/duplicate records. No train/test split is inferred.

```powershell
python src/run_ner.py --input-file datasets/court_documents/documents.jsonl --output-file outputs/ner_predictions.jsonl --limit 20 --device auto
python src/build_synthetic_unanonymized.py --input-file datasets/court_documents/documents.jsonl --ner-file outputs/ner_predictions.jsonl --limit 20
```

Use the same `documents.jsonl` as `--source-file` for the HTML inspector.
`datasets/court_documents_local/`, if present, is an independently prepared
local-shard subset, not a verified complete/current Hugging Face snapshot.

The pipeline generates synthetic replacements for masked references in published court documents. It does not recover hidden identities. Run commands from the repository root; existing notebook shell commands remain compatible.

## Design: evidence-v2

| Stage | Responsibility | Evidence required |
|---|---|---|
| Observe | Preserve NER predictions and find complete candidate tokens, including procedural codes and dotted initials | Original offsets and immutable NER observations |
| Type | Decide whether an occurrence is a person, organization, address component, or unresolved | A direct syntactic prefix or NER observation; nearby unrelated words cannot change its label |
| Link | Build identities and attach references | Complete procedural codes, contiguous name prefixes, explicit alias declarations, or audited LLM evidence |
| Generate | Create a replacement appropriate to the supported identity and encoding | Compatible names, known organization form, address unit and parent scope |
| Validate | Check spans, apply replacements, and verify exact reversal | Non-overlapping replacements and matching source text |
| Review | Record unresolved candidates and uncertainty | Document and entity review reasons, not model confidence alone |

No document-wide marker-label propagation is performed. A typed person M elsewhere cannot classify an unknown occurrence of M. Address groups include the precise unit and parent context; an identical letter in two different address hierarchies does not establish identity.

The modules separate these responsibilities:

- `src/reconstruction.py`: candidate evidence, type decisions, typed identity registry, replacement planning.
- `src/synthetic_lexicon.py`: synthetic vocabulary only.
- `src/legal_linking.py`: bounded Gemini requests, validated decisions, and round-trip verification.
- `src/build_synthetic_unanonymized.py`: compatible CLI and dataset/audit/map output.
- `src/inspect_entity_links.py`: standalone HTML review.

The former proximity/marker-propagation engine has been removed. NER evidence retains its original label and score in `ner_evidence`; that score is not presented as final linking confidence.

## Supported conventions and deliberate abstentions

- `NLQ 1` and `NLQ1` are the same code; `NLQ2` is distinct. NLQ does not determine whether the participant is a person or organization.
- `Nguyễn Văn H12` retains the visible name prefix. A synthetic given name must match its initial.
- A declared whole-name alias such as `Bị đơn: Ông A` does not constrain the generated name's initial.
- Dotted initials such as `N.H.H` are atomic. Supported three-part codes receive compatible family/middle/given initials; unsupported forms remain intact.
- Abbreviations and technical identifiers such as `V/v` and `loại A50` are preserved. Uppercase fragments inside Vietnamese headings are excluded.
- Address-unit words stay outside replacement spans. Roman numerals, foreign addresses and conflicting explicit parents require review. Missing parents alone do not prevent reconstruction.

### Offline administrative location vocabulary

Reconstruction uses the bundled [dvhcvn vocabulary](resources/dvhcvn/SOURCE.md): a pinned **1 March 2025** snapshot with 63 province-level, 696 district-level and 10,047 commune-level records. Source IDs and parent IDs are retained in `resources/dvhcvn/units.tsv`, with upstream attribution and license alongside it. No download is needed at runtime. `python src/import_dvhcvn.py` regenerates these resources from the pinned upstream revision when internet is available.

- Administrative replacements use real names of the exact unit (`huyện`, `quận`, `xã`, `phường`, etc.), excluding purely numeric names for alphabetic aliases. Hamlet/street names retain synthetic fallback pools. Alias initials are not enforced for locations.
- Detection still prioritizes explicit address grammar. The vocabulary recognizes complete adjacent parent names, avoiding truncated names and arbitrary prose. Dictionary-supported spacing repair operates within NER spans, protects anonymization markers and leaves source offsets/text untouched. It is not a standalone full-document location NER replacement.
- Repeated unit-plus-alias mentions keep the existing document-level linking; explicit conflicting parents trigger review. Names are sampled independently: a generated address is **not guaranteed to be a valid administrative hierarchy**. The snapshot is historical, not a current-boundary database.
- Unsupported foreign name prefixes, fragmented OCR names, possible unmasked aliases, incompatible name pools and conflicting evidence remain unresolved.
- Procedural organizations need a complete organization form before synthetic replacement. A representative's job title does not establish that form.
- Existing full names are not copied as recovered identities. Potential links to unmasked names are flagged for review; unchanged source content can still contain real identities.
- Overlapping plans block all affected entities; the code never silently chooses one edit.

These are conservative evidence requirements, not a complete implementation of every court convention. Cross-convention links such as NLQ4 to a separate unmasked given name still need review. Geographic names are synthetic vocabulary, not a verified administrative gazetteer.

Reference conventions: [Article 7 of NQ 03/2017/NQ-HĐTP](https://thuvienphapluat.vn/van-ban/cong-nghe-thong-tin/nghi-quyet-03-2017-nq-hdtp-cong-bo-ban-an-quyet-dinh-tren-cong-thong-tin-dien-tu-toa-an-343156.aspx) and [section 2 of CV 144/TANDTC-PC](https://congbobanan.toaan.gov.vn/1t2atcvn/thi-hanh-nq-so-03-2017/nq-hdtp).

## Run locally or in a notebook shell cell

Reconstruction and inspection require only Python's standard library. NER requirements are in `requirements.txt`.

```bash
python src/build_synthetic_unanonymized.py --input-file datasets/legal_test.jsonl --ner-file outputs/nlphust_legal_sample.jsonl --limit 20
```

Optional Gemini: set `GOOGLE_API_KEY` or `GEMINI_API_KEY` in the same terminal that runs Python. Alternatively use `--env-file .env` with a local key file ignored by Git. No Google SDK is required.

```bash
python src/build_synthetic_unanonymized.py --input-file datasets/legal_test.jsonl --ner-file outputs/nlphust_legal_sample.jsonl --limit 20 --llm-provider gemini --llm-model YOUR_GEMINI_MODEL_ID --llm-max-calls 20
```

Choose a model available to your Google project. Gemini receives selected document excerpts. Its tasks are type conflicts, unknown procedural types, ambiguous references and untyped references to existing person candidates. It chooses supplied IDs, NOT_ENTITY where offered, or ABSTAIN. It cannot invent names or overwrite protected lexical identifiers.

The model's choice must match an offered candidate and include a reason and an exact quote from supplied text. This checks response validity, not semantic truth. Accepted LLM decisions remain flagged for human review. They are not used as fresh evidence for subsequent model calls.

The budget applies across the run; `--llm-max-calls 0` makes no calls. Timeout/API errors, missing evidence and abstentions are recorded. There are no automatic retries. Requests use Google's [generateContent REST API](https://ai.google.dev/api/generate-content).

## Outputs and review

The same three reconstruction outputs are written; reruns overwrite these paths:

- `outputs/synthetic_unanonymized.jsonl`: source and synthetic text, pipeline version, review status/reasons and statistics.
- `outputs/entity_links.jsonl`: entities, raw NER observations, rejected candidates, evidence, review reasons and applied replacements.
- `outputs/replacement_maps.jsonl`: mappings with original text and source/synthetic offsets.

Override paths using `--output-file`, `--links-file`, and `--maps-file`. Paths must be distinct and cannot overwrite the input. The NER file is preserved. No debug artifact files are generated.

Documents are `needs_review` when any recorded issue remains (including missing NER); otherwise they are `passed_automatic_checks`. Neither state is a human approval or a guarantee of recall. Round-trip verification establishes reversible edits only. `overlap_conflict_entities` counts entities blocked by conflicting spans.

## Large-scale curation and quality filtering

The source collection is much larger than a practical first NER run. The
curation command selects a deterministic, marker-bearing subset while keeping
legal diversity across `case_type`, `doc_type`, and `cap_xet_xu`:

```powershell
python src/curate_court_dataset.py `
  --input-file datasets/court_documents_local/documents.jsonl `
  --output-file datasets/court_documents_curated/documents.jsonl `
  --per-stratum 100 --min-chars 500
```

The output contains the original text and metadata plus a `curation` audit
object. Change `--seed` to create a different reproducible sample. The default
selector requires at least one procedural code, dotted initial, address alias,
or initial-like marker. Use `--include-no-marker` only when a negative/control
set is desired.

After NER and reconstruction, score and keep conservative records:

```powershell
python src/filter_reconstructed_dataset.py `
  --synthetic-file outputs/synthetic_unanonymized.jsonl `
  --links-file outputs/entity_links.jsonl `
  --clean-output outputs/synthetic_unanonymized_clean.jsonl `
  --report-file outputs/reconstruction_quality.json `
  --min-score 85 --min-replacements 1
```

`reconstruction_quality` is a deterministic triage score, not a calibrated
probability. The default filter requires no review reasons, a successful
round-trip, and at least one applied replacement. Rejected records are not
deleted; they remain in the original reconstruction and link-audit files.
Use `--allow-review` only to create a broader candidate pool for manual review.
The report includes rejection reasons and selected distributions by case and
document type, so a high-quality output can still be checked for diversity.

```bash
python src/inspect_entity_links.py --source-file datasets/legal_test.jsonl --links-file outputs/entity_links.jsonl --synthetic-file outputs/synthetic_unanonymized.jsonl --doc-id 1000001 --output-file outputs/document_review.html
```

The viewer shows original and reconstructed text side by side with highlights, document review reasons, and the entity table. Click a mention or table row to highlight the same entity throughout; hover for evidence. LLM decisions are expandable. Mismatched source hashes are rejected. Old outputs remain readable as legacy outputs but need regeneration to show evidence-v2 metadata.

To inspect a complete demo run, generate a folder containing an index and one page per document:

```bash
python src/inspect_demo.py \
  --source-file datasets/demo_court_documents_v2/documents.jsonl \
  --links-file outputs/entity_links.jsonl \
  --synthetic-file outputs/synthetic_unanonymized.jsonl \
  --output-dir outputs/demo_500_html
```

Open `outputs/demo_500_html/index.html`. The index can be filtered by document ID, case type, document type, quality tier, or review reason; each page shows the highlighted source, synthetic reconstruction, quality, and entity audit.

## Verification

```bash
python -B -m unittest discover -s tests -v
```

Tests cover known corruption cases, atomic replacements, independent NER evidence, candidate constraints, source integrity and mocked API failures. Live Gemini accuracy requires evaluation on reviewed documents.
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
