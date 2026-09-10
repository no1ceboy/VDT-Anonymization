"""Extract observed PER/ORG examples from the local VLSP training Parquet.

The numeric mapping is inferred from annotated examples, not publisher metadata.
Run with Python and pyarrow; the resulting JSON requires no Parquet dependency.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def clean(text):
    return ' '.join(unicodedata.normalize('NFC', text).replace('_', ' ').split())


def extract(rows):
    inventories = {'PER': {}, 'ORG': {}}
    stats = Counter()
    seen = set()
    for row_id, row in enumerate(rows):
        tokens, tags = row['tokens'], row['ner_tags']
        stats['input_rows'] += 1
        if len(tokens) != len(tags) or any(tag not in range(9) for tag in tags):
            raise ValueError(f'Invalid tokens/tags at row {row_id}')
        signature = (tuple(tokens), tuple(tags))
        if signature in seen:
            stats['duplicate_rows_skipped'] += 1
            continue
        seen.add(signature)
        i = 0
        while i < len(tags):
            if tags[i] not in (1, 3):
                if tags[i] in (2, 4):
                    stats['orphan_inside_tokens_skipped'] += 1
                i += 1
                continue
            begin, tag = i, tags[i]
            i += 1
            while i < len(tags) and tags[i] == tag + 1:
                i += 1
            label = 'PER' if tag == 1 else 'ORG'
            name = clean(' '.join(tokens[begin:i]))
            if not name:
                continue
            entry = inventories[label].setdefault(name, {
                'name': name, 'count': 0, 'examples': [],
            })
            entry['count'] += 1
            stats[label + '_mentions'] += 1
            if len(entry['examples']) < 3:
                entry['examples'].append({
                    'row': row_id, 'token_start': begin, 'token_end': i,
                    'raw_tokens': tokens[begin:i],
                    'context': clean(' '.join(tokens[max(0, begin-8):min(len(tokens), i+8)])),
                })
    from synthetic_lexicon import FAMILY_NAMES
    families = {name.casefold() for name in FAMILY_NAMES}
    for entry in inventories['PER'].values():
        parts = entry['name'].split()
        candidate = (2 <= len(parts) <= 5 and parts[0].casefold() in families
                     and all(p.isalpha() and len(p) > 1 and p[0].isupper() for p in parts))
        entry['candidate_status'] = 'surname_and_shape_match' if candidate else 'example_only'
        if candidate:
            entry['surname_candidate'] = parts[0]
            entry['final_syllable_candidate'] = parts[-1]
    for entry in inventories['ORG'].values():
        entry['candidate_status'] = 'example_only'
        match = re.match(r'^(ngân hàng(?: thương mại cổ phần)?|công ty(?: cổ phần| tnhh| trách nhiệm hữu hạn)?)\s+(.+)$', entry['name'], re.I)
        if match:
            entry['organization_form'] = match.group(1)
            entry['observed_suffix'] = match.group(2)
    return {
        'statistics': dict(stats),
        'persons': sorted(inventories['PER'].values(), key=lambda e: (-e['count'], e['name'])),
        'organizations': sorted(inventories['ORG'].values(), key=lambda e: (-e['count'], e['name'])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--output-file', default='datasets/vlsp_vocabulary.json')
    args = parser.parse_args()
    source, target = Path(args.input_file), Path(args.output_file)
    if source.resolve() == target.resolve():
        parser.error('Output must not overwrite input')
    import pyarrow.parquet as pq
    result = extract(pq.read_table(source, columns=['tokens', 'ner_tags']).to_pylist())
    result['provenance'] = {
        'source_file': source.name,
        'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'dataset_url': 'https://huggingface.co/datasets/datnth1709/VLSP2016-NER-data',
        'split': 'train (user-supplied training file)',
        'tag_mapping': {'1': 'B-PER', '2': 'I-PER', '3': 'B-ORG', '4': 'I-ORG'},
        'mapping_status': 'inferred_from_examples; not documented by publisher',
        'notes': 'Observed names are not fictional identities. Candidate filtering is heuristic; final syllables are not annotated given-name boundaries. Frequency counts exclude duplicate sentences.',
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['statistics']))
    print(f"Unique persons: {len(result['persons'])}; organizations: {len(result['organizations'])}")
    print('Person shape candidates:', sum(e['candidate_status'] != 'example_only' for e in result['persons']))
    print('Saved:', target)


if __name__ == '__main__':
    main()
