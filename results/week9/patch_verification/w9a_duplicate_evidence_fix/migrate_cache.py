"""One-time audited migration for the identical-evidence duplicate-question fix.

Does not regenerate predictions or change payloads. Retains original files via
hard links before atomic replacement; rejects any unrelated identity change.
"""
import fcntl
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from src.week9.provenance import branch_guard, digest, file_hash, ROOT
from src.week9.caching import atomic_json
from src.week9.config import load_config
from src.week9.data import load_records, raw_dataset, evaluation_evidence
from src.week9.runner import build_identity

branch_guard()
base=ROOT/'results/week9/retrieval/fit'
audit=Path(__file__).parent
backup=audit/'original_cache'
if backup.exists():
    raise FileExistsError('Migration already prepared; inspect audit and original files instead of rerunning')
old_source_hash=file_hash('/tmp/w9a_data_before_duplicate_fix.py')
cfg=load_config(ROOT/'configs/week9/w9a.yaml')
cfg['_seed']=42
records,paths=load_records(cfg,'fit')
current=build_identity(cfg,'a','fit',42,paths,{})
old=json.loads((base/'w9a_retrieval_results.identity.json').read_text())
assert old['source_hashes']['src/week9/data.py']==old_source_hash
expected=json.loads(json.dumps(current))
expected['source_hashes']['src/week9/data.py']=old_source_hash
expected['prompt_version_hash']=digest(expected['source_hashes'])
assert old==expected, 'Unrelated source/config/input/model change: refusing migration'
raw=raw_dataset(str(paths['raw']))
ambiguous=set()
for rec in records:
    i=rec['inference'];paper=raw[int(i['paper_id'])]
    matches=[q for q in paper['qas']['question'] if q.strip()==i['question']]
    if len(matches)!=1:ambiguous.add(i['question_id'])
    evaluation_evidence(rec,paths['raw'])  # All FIT must resolve safely before writing.
oldstamp,newstamp=digest(old),digest(current)
chunk_ids={};locks=[];prepared=[];report=[]
try:
    for name in ['w9a_chunk_manifest','w9a_embeddings','w9a_retrieval_results']:
        lock=(base/(name+'.lock')).open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        assert json.loads((base/(name+'.identity.json')).read_text())==old
    for name in ['w9a_chunk_manifest','w9a_embeddings','w9a_retrieval_results']:
        p=base/(name+'.jsonl');tmp=p.with_suffix('.duplicate_fix.tmp')
        if tmp.exists():raise FileExistsError(tmp)
        seen=set();count=0
        with p.open() as src,tmp.open('x') as dst:
            for line in src:
                assert line.endswith('\n'), 'Partial row: inspect before migration'
                row=json.loads(line)
                assert row['row_content_hash']==digest({k:v for k,v in row.items() if k!='row_content_hash'})
                assert row['cache_fingerprint']==oldstamp
                if name=='w9a_chunk_manifest':
                    args=[str(row['question_id']),str(row['paper_id']),[],row['representation']]
                    chunk_ids[(str(row['question_id']),row['representation'])]=[c['chunk_id'] for c in row['chunks']]
                elif name=='w9a_embeddings':
                    args=['INDEX',str(row['paper_id']),[row['chunk_id']],'chunk_embedding']
                else:
                    assert str(row['question_id']) not in ambiguous, 'Unexpected cached ambiguous-question outcome'
                    operation={k:row[k] for k in ['representation','retriever','retrieval_width']}
                    args=[str(row['question_id']),str(row['paper_id']),chunk_ids[(str(row['question_id']),row['representation'])],operation]
                assert row['cache_key']==digest(args+[oldstamp])
                assert row['cache_key'] not in seen;seen.add(row['cache_key'])
                payload={k:v for k,v in row.items() if k not in ['cache_key','cache_fingerprint','row_content_hash']}
                row['cache_key']=digest(args+[newstamp]);row['cache_fingerprint']=newstamp
                row['row_content_hash']=digest({k:v for k,v in row.items() if k!='row_content_hash'})
                assert payload=={k:v for k,v in row.items() if k not in ['cache_key','cache_fingerprint','row_content_hash']}
                dst.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');count+=1
            dst.flush();os.fsync(dst.fileno())
        report.append(dict(path=str(p.relative_to(ROOT)),rows=count,old_sha256=file_hash(p),new_sha256=file_hash(tmp),payloads_unchanged=True))
        prepared.append((p,tmp))
        print(name,count,'validated and prepared',flush=True)
    backup.mkdir()
    for p,tmp in prepared:
        for original in [p,p.with_suffix('.identity.json')]:os.link(original,backup/original.name)
    atomic_json(audit/'migration.json',dict(status='PREPARED',old_identity=old,new_identity=current,identical_evidence_query_ids=sorted(ambiguous),files=report))
    for p,tmp in prepared:
        os.replace(tmp,p);atomic_json(p.with_suffix('.identity.json'),current)
    value=json.loads((audit/'migration.json').read_text());value['status']='COMPLETE';atomic_json(audit/'migration.json',value)
    print('COMPLETE: originals preserved; only cache keys/fingerprints/checksums changed.',flush=True)
finally:
    for lock in locks:lock.close()
