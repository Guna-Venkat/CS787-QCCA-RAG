"""Audited one-time, lossless W9-A compaction; no generation or metric reruns."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from src.week9.provenance import branch_guard, ROOT, digest, file_hash
from src.week9.caching import atomic_json
from src.week9.config import load_config
from src.week9.data import load_records
from src.week9.runner import build_identity
from src.week9.retrieval_cache import RetrievalCache, read_rows

branch_guard()
audit=Path(__file__).parent;base=ROOT/'results/week9/retrieval/fit'
report_path=audit/'compaction.json'
if report_path.exists() and json.loads(report_path.read_text()).get('status')=='COMPLETE':
    raise FileExistsError('Compaction already complete; do not rerun')
before=json.loads((audit/'before_sources.json').read_text())
old=json.loads((base/'w9a_retrieval_results.identity.json').read_text())
assert old['source_hashes']==before
cfg=load_config(ROOT/'configs/week9/w9a.yaml');cfg['_seed']=42
records,paths=load_records(cfg,'fit')
current=build_identity(cfg,'a','fit',42,paths,{})
expected=dict(current,source_hashes=before,prompt_version_hash=digest(before))
assert expected==old,'Unrelated config/model/input change: stop before conversion'
tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
oldstamp=digest(old);chunk_ids={};locks=[];reports=[]
metadata={'cache_key','cache_fingerprint','row_content_hash'}

def payload_hash(row):
    return digest({k:v for k,v in row.items() if k not in metadata}).encode()

try:
    for name in ['w9a_chunk_manifest','w9a_embeddings','w9a_retrieval_results']:
        lock=(base/(name+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        assert json.loads((base/(name+'.identity.json')).read_text())==old
    for name in ['w9a_chunk_manifest','w9a_embeddings','w9a_retrieval_results']:
        source=base/(name+'.jsonl');target=base/(name+'_compact.sqlite3')
        assert str(source.relative_to(ROOT)) not in tracked
        sha=hashlib.sha256();content=hashlib.sha256();n=0;partial=0;seen=set()
        with RetrievalCache(target,current,resume=True) as cache,source.open('rb') as stream:
            for line in stream:
                sha.update(line)
                if not line.endswith(b'\n'):
                    partial=len(line);break
                row=json.loads(line)
                assert row['row_content_hash']==digest({k:v for k,v in row.items() if k!='row_content_hash'})
                assert row['cache_fingerprint']==oldstamp
                if name=='w9a_chunk_manifest':
                    args=[str(row['question_id']),str(row['paper_id']),[],row['representation']]
                    chunk_ids[(str(row['question_id']),row['representation'])]=[c['chunk_id'] for c in row['chunks']]
                elif name=='w9a_embeddings':
                    args=['INDEX',str(row['paper_id']),[row['chunk_id']],'chunk_embedding']
                else:
                    operation={k:row[k] for k in ['representation','retriever','retrieval_width']}
                    args=[str(row['question_id']),str(row['paper_id']),chunk_ids[(str(row['question_id']),row['representation'])],operation]
                assert row['cache_key']==digest(args+[oldstamp]) and row['cache_key'] not in seen
                seen.add(row['cache_key']);content.update(payload_hash(row))
                key=digest(args+[cache.stamp]);saved=cache.put(key,row)
                assert payload_hash(saved)==payload_hash(row)
                n+=1
                if n%10000==0:print(name,n,'rows converted',flush=True)
            assert len(cache.rows)==n
        verified=hashlib.sha256();count=0
        for row in read_rows(target):verified.update(payload_hash(row));count+=1
        assert count==n and verified.hexdigest()==content.hexdigest()
        with sqlite3.connect(target) as db:assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        entry=dict(source=str(source.relative_to(ROOT)),target=str(target.relative_to(ROOT)),rows=n,
            source_sha256=sha.hexdigest(),target_sha256=file_hash(target),payload_chain_sha256=content.hexdigest(),
            source_bytes=source.stat().st_size,target_bytes=target.stat().st_size,ignored_incomplete_tail_bytes=partial,
            every_completed_payload_preserved=True,integrity_check='ok')
        reports.append(entry)
        atomic_json(report_path,dict(status='CONVERTING',old_identity=old,new_identity=current,files=reports))
        print(name,'VERIFIED',n,'rows;',entry['source_bytes'],'->',entry['target_bytes'],'bytes',flush=True)
    atomic_json(report_path,dict(status='VERIFIED_BEFORE_CLEANUP',old_identity=old,new_identity=current,files=reports))
    removed=[]
    for entry in reports:
        source=ROOT/entry['source']
        # Verify file did not change during conversion before deleting redundant bytes.
        assert file_hash(source)==entry['source_sha256']
        for p in [source,source.with_suffix('.identity.json')]:
            assert str(p.relative_to(ROOT)) not in tracked
            removed.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size));p.unlink()
    previous=ROOT/'results/week9/patch_verification/w9a_duplicate_evidence_fix'
    ledger=json.loads((previous/'migration.json').read_text())
    backup=previous/'original_cache'
    for entry in ledger['files']:
        p=backup/Path(entry['path']).name
        if p.exists():
            assert file_hash(p)==entry['old_sha256']
            assert str(p.relative_to(ROOT)) not in tracked
            removed.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size));p.unlink()
        side=p.with_suffix('.identity.json')
        if side.exists():
            assert json.loads(side.read_text())==ledger['old_identity']
            assert str(side.relative_to(ROOT)) not in tracked
            removed.append(dict(path=str(side.relative_to(ROOT)),bytes=side.stat().st_size));side.unlink()
    if backup.exists() and not any(backup.iterdir()):backup.rmdir()
    atomic_json(report_path,dict(status='COMPLETE',old_identity=old,new_identity=current,files=reports,
        removed_redundant_files=removed,completed_payloads_preserved=True,full_run_launched=False))
    print('COMPLETE: verified compact databases replace JSONL; redundant recovery copies removed.',flush=True)
finally:
    for lock in locks:lock.close()
