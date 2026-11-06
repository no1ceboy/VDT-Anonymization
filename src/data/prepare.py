"""Download document shards or convert local Parquet to pipeline-compatible JSONL.

Requires pyarrow; online download additionally requires huggingface_hub.
Credentials are read by huggingface_hub from the environment/local login.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

REPO = 'tmquan/congbobanan-toaan-gov-vn'


def convert(shards, output_dir, provenance):
    import pyarrow.parquet as pq
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / 'documents.jsonl'
    partial = output_dir / 'documents.jsonl.partial'
    manifest = output_dir / 'manifest.json'
    if any(p.exists() for p in (target, partial, manifest)):
        raise ValueError('Output already exists; choose a new --output-dir to preserve earlier data.')
    tables=[]
    for path in shards:
        table=pq.ParquetFile(path)
        if not {'case_id','markdown'}.issubset(table.schema_arrow.names):
            raise ValueError(f'{path.name}: required case_id/markdown columns missing')
        tables.append((path,table))
    stats={'rows_written':0,'empty_text_skipped':0,'duplicate_case_ids_skipped':0}
    seen={}
    sources=[]
    with partial.open('x',encoding='utf-8') as out:
        for path,table in tables:
            digest=hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda:handle.read(1024*1024),b''):
                    digest.update(chunk)
            sources.append({'file':path.name,'sha256':digest.hexdigest(),'rows':table.metadata.num_rows})
            for batch in table.iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    text=row.get('markdown')
                    if not isinstance(text,str) or not text.strip():
                        stats['empty_text_skipped']+=1
                        continue
                    if row.get('case_id') is None or not str(row['case_id']).strip():
                        raise ValueError(f'{path.name}: nonempty document missing case_id')
                    row['case_id']=str(row['case_id'])
                    text_hash=hashlib.sha256(text.encode('utf-8')).hexdigest()
                    previous=seen.get(row['case_id'])
                    if previous:
                        if previous!=text_hash:
                            raise ValueError(f'Conflicting texts for case_id {row["case_id"]}; resolve before conversion')
                        stats['duplicate_case_ids_skipped']+=1
                        continue
                    seen[row['case_id']]=text_hash
                    # Keep source text and metadata; existing CLI defaults use
                    # case_id and markdown. No text cleaning or splitting here.
                    out.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    stats['rows_written']+=1
            print(f'[PREP] {path.name}: total documents={stats["rows_written"]}',flush=True)
    if not stats['rows_written']:
        raise ValueError('No documents produced; partial output retained for inspection')
    manifest.write_text(json.dumps({
        'dataset':REPO,'created_at':datetime.now(timezone.utc).isoformat(),
        'provenance':provenance,'sources':sources,'statistics':stats,
        'output':'documents.jsonl','text_field':'markdown','id_field':'case_id',
        'split':'source collection; no train/validation/test split assigned',
    },ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    partial.rename(target)
    print(f'[PREP] Ready: {target}',flush=True)
    return target,stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-parquet',nargs='+',type=Path,help='Use existing shards without downloading')
    parser.add_argument('--output-dir',type=Path,default=Path('datasets/court_documents'))
    parser.add_argument('--revision',default='main')
    parser.add_argument('--max-shards',type=int,default=0,help='0 downloads all document shards')
    args=parser.parse_args()
    if args.max_shards<0:
        parser.error('--max-shards cannot be negative')
    if any((args.output_dir/name).exists() for name in ('documents.jsonl','documents.jsonl.partial','manifest.json')):
        parser.error('Prepared output already exists; choose a new --output-dir')
    if args.local_parquet:
        shards=sorted(args.local_parquet)
        provenance={'mode':'local','revision':None,'completeness':'only supplied shards; remote revision unverified'}
    else:
        from huggingface_hub import HfApi,hf_hub_download
        info=HfApi().dataset_info(REPO,revision=args.revision,files_metadata=True)
        files=sorted(s.rfilename for s in info.siblings if re.fullmatch(r'documents-\d+-of-\d+\.parquet',s.rfilename))
        if not files:
            parser.error('No document shards found')
        available=len(files)
        if args.max_shards:
            files=files[:args.max_shards]
        print(f'[PREP] Revision {info.sha}; downloading {len(files)}/{available} document shards',flush=True)
        shards=[Path(hf_hub_download(REPO,f,repo_type='dataset',revision=info.sha,
                                   local_dir=args.output_dir/'raw')) for f in files]
        provenance={'mode':'download','revision':info.sha,'available_shards':available,'selected_shards':len(files)}
    convert(shards,args.output_dir,provenance)


if __name__=='__main__':
    main()
