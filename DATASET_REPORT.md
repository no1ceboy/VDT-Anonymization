# VDT entity-linking dataset: quantitative report

## Dataset scope

The dataset was streamed from `tmquan/cbba-toaan-gov-vn`, then passed through
the strict v1 reconstruction filter.

| Measure | Value |
|---|---:|
| Source records processed | 10,000 |
| Retained clean documents | 9,201 |
| Rejected documents | 799 |
| Retention rate | 92.01% |
| Rejection rate | 7.99% |
| Total source text | 33,513,497 characters |
| Median document length | 2,839 characters |
| 90th-percentile document length | 7,537 characters |
| Entity records | 51,088 |
| Replacement mentions | 189,503 |
| Mean mentions per document | 20.60 |
| Documents with repeated entity mentions | 8,932 (97.08%) |
| Unreconstructable retained entities | 0 |

The strict gate required quality score 100, no review reasons, successful
round-trip verification, and at least one replacement. Rejection reasons can
overlap because one document may trigger more than one rule:

| Rejection signal | Documents flagged |
|---|---:|
| Numeric amount-boundary collision | 728 |
| Unsupported large numeric suffix | 649 |
| Duplicate source text | 24 |
| Compound literal-number collision | 8 |
| Document-identifier collision | 8 |
| Future-issued date | 1 |

## Entity composition

| Entity label | Count | Share |
|---|---:|---:|
| PER | 25,616 | 50.14% |
| LOC | 24,570 | 48.09% |
| ADDR | 612 | 1.20% |
| ORG | 290 | 0.57% |

The retained documents contain 20,723 distinct synthetic entity values. The
dominant entity types are people and locations; organization-linking examples
are comparatively scarce.

## Document diversity

| Category | Documents | Share |
|---|---:|---:|
| Marriage & Family | 3,483 | 37.85% |
| Civil | 2,963 | 32.20% |
| Unknown | 1,849 | 20.10% |
| Criminal | 689 | 7.49% |
| Administrative | 128 | 1.39% |
| Commercial | 50 | 0.54% |
| Labor | 36 | 0.39% |
| Bankruptcy | 2 | 0.02% |
| Economic | 1 | 0.01% |

Instance-level metadata are First-instance 7,193 (78.18%), Unknown 1,850
(20.11%), Appellate 147 (1.60%), and Retrial 11 (0.12%). This is diverse in
challenge form, but not metadata-balanced: Marriage & Family and Civil account
for 70.05% of the documents, and 20.10% have unknown category metadata.

## Reconstruction challenge coverage

| Primary challenge | Documents | Share |
|---|---:|---:|
| Address marker | 6,898 | 74.97% |
| Full-name marker | 1,726 | 18.76% |
| Numbered marker | 265 | 2.88% |
| Alias language | 147 | 1.60% |
| Dotted initial | 73 | 0.79% |
| Numbered multi-letter marker | 45 | 0.49% |
| Organization marker | 37 | 0.40% |
| Person-context marker | 9 | 0.10% |
| Procedural code | 1 | 0.01% |

There are nine challenge families. Address and full-name cases dominate, while
procedural-code and person-context cases should be treated as low-count stress
tests rather than statistically reliable benchmark slices.

## Entity-linking pair dataset

Pairs were split by complete document, stratified by category, instance level,
and primary challenge. There is zero document-ID overlap and zero source-text
hash overlap between splits.

| Split | Documents | Pairs | Positive | Negative |
|---|---:|---:|---:|---:|
| Train | 7,423 | 204,079 | 105,394 | 98,685 |
| Validation | 889 | 24,780 | 12,838 | 11,942 |
| Test | 889 | 24,409 | 12,589 | 11,820 |
| Total | 9,201 | 253,268 | 130,821 | 122,447 |

Overall, 51.65% of pairs are positive and 48.35% negative. Of the negative
pairs, 109,549 (89.47%) use the same entity label, including same-surface,
same-marker, and same-marker-family conflicts. These are the deliberately hard
cases; only 12,898 negatives are cross-label controls.

The pair inputs contain original anonymized context around each mention and
never include the synthetic reconstruction value. Labels are weak supervision:
positive pairs share a rule-generated `entity_id`, while negative pairs have
different rule-generated IDs. They are not independently human-annotated GT.

## Rule baselines on the held-out test split

| Baseline | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Exact marker + label | 95.89% | 92.61% | 100.00% | 96.17% |
| Marker family + label | 94.03% | 89.63% | 100.00% | 94.53% |
| Exact surface + label | 78.36% | 91.71% | 63.81% | 75.26% |

The exact-marker rule produces 1,004 false positives on 24,409 test pairs. This
is the main quantitative reason to test a context encoder: marker equality is
strong but does not fully distinguish conflicting entities.

## Recommended report conclusion

The v1 release provides 9,201 strictly filtered documents and 253,268
document-disjoint entity-linking pairs. It has strong coverage of address,
full-name, numbered, dotted-initial, alias, and organization cases, with 89.47%
of negative pairs intentionally drawn from same-label conflicts. The dataset is
clean under deterministic structural checks, but its labels are rule-generated
weak supervision and its document metadata are imbalanced. Model performance
should therefore be reported against the exact-marker baseline and followed by
human review of a sample of hard same-label conflicts.
