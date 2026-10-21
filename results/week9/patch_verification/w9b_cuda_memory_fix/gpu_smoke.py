"""Bounded real FIT q512 smoke; separate caches, no full experiment or freeze."""
import json,time
from pathlib import Path
import torch
from src.week9.config import load_config
from src.week9.data import load_records
from src.week9.runner import dependencies,get_pool,interventions,run_intervention
from src.week9.backend import SaraBackend
from src.week9.retrieval import LocalEncoder
from src.week9.caching import RowCache,atomic_json
from src.week9.provenance import source_hashes,branch_guard
branch_guard()
out=Path('results/week9/patch_verification/w9b_cuda_memory_fix/smoke')
cfg=load_config('configs/week9/w9b.yaml')
frozen=dependencies(cfg,'b','fit')
records,paths=load_records(cfg,'fit')
record=next(r for r in records if r['inference']['question_id']=='512')
pool=get_pool(cfg,frozen,'fit','512')
embeddings=torch.load(paths['sfr'],map_location='cpu',weights_only=False)
if 'embeddings' in embeddings: embeddings=embeddings['embeddings']
print('Loading trained SARA, explicit FP16 efficient SDPA',flush=True)
backend=SaraBackend(cfg)
encoder=LocalEncoder(cfg['encoder'],cfg['encoder_device'])
identity=dict(source_hashes=source_hashes(),role='bounded_gpu_smoke',question_id='512',model=backend.identity())
rows=[]
with RowCache(out/'lm_calls.jsonl',identity,resume=True) as calls, RowCache(out/'extraction.jsonl',identity,resume=True) as ext:
    for spec in interventions('b',cfg,frozen):
        start=time.perf_counter()
        row=run_intervention(record,pool,spec,encoder,backend,calls,embeddings,cfg,ext)
        rows.append(row)
        atomic_json(out/'interventions.json',rows)
        print(json.dumps(dict(k=spec['evidence_k'],seconds=time.perf_counter()-start,prediction=row['prediction'],calls=row.get('model_calls'))),flush=True)
    results=list(calls.rows.values())
    assert len(results)==3
    assert all(r['dtype']=='torch.float16' and r['attention_backend']=='efficient_sdpa' and r['content'] for r in results)
    assert max(r['peak_memory_allocated'] for r in results)<20_000_000_000
    atomic_json(out/'verification.json',dict(passed=True,max_new_tokens=cfg['max_new_tokens'],query_id='512',calls=[{k:r[k] for k in ('prompt_tokens','generated_tokens','peak_memory_allocated','peak_memory_reserved','dtype','attention_backend','latency','content')} for r in results]))
print('Real K0/1/2/4 smoke passed',flush=True)
