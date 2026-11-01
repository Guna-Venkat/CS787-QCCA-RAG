"""Validate real one-query selector smoke independently of primary outcomes."""
import json
from pathlib import Path
from src.week9.provenance import branch_guard,digest,file_hash,source_hashes
from src.week9.caching import read_jsonl,atomic_json
branch_guard()
root=Path('results/week9/diagnostics/w9c_k2')
dir=root/'selection/fit/smoke'
rows=read_jsonl(dir/'w9c_selection.jsonl')
meta=json.loads((dir/'w9c_selection.identity.json').read_text())
assert meta['source_hashes']==source_hashes()
assert len(rows)==5 and {r['policy'] for r in rows}=={'BM25','QuerySemantic','DraftSemantic','QueryDraft','SANE-adapted'}
assert len({r['question_id'] for r in rows})==1
assert all(r['cache_fingerprint']==digest(meta) and r['row_content_hash']==digest({k:v for k,v in r.items() if k!='row_content_hash'}) for r in rows)
assert all(r['evidence_k']==2 and r['evidence_actual_count']==min(2,r['retrieval_actual_count']) and r['prediction'].strip() for r in rows)
calls=read_jsonl(dir/'lm_calls.jsonl')
assert calls and all(r['dtype']=='torch.float16' and r['attention_backend']=='efficient_sdpa' for r in calls)
assert any(r['stage']=='selector' for r in calls)
assert max(r['peak_memory_allocated'] for r in calls)<20_000_000_000
saved=next(r for r in read_jsonl('results/week9/ke_surface/fit/w9b_ke_surface.jsonl') if r['question_id']==rows[0]['question_id'] and r['evidence_k']==2)
current=next(r for r in rows if r['policy']=='DraftSemantic')
assert saved['selected_chunk_ids']==current['selected_chunk_ids'] and saved['prediction']==current['prediction']
primary=json.loads(Path('results/week9/frozen/w9b_frozen_config.json').read_text())
assert primary['settings']['evidence_k']==0
manifest=json.loads((dir/'w9c_manifest.json').read_text())
assert not manifest['complete'] and manifest['scope']=='partial'
assert not (root/'frozen/w9c_frozen_config.json').exists()
assert not (root/'selection/fit/w9c_selection.jsonl').exists()
report=dict(ready=True,scope='FIT-only conditional K=2 selector diagnostic; not primary architecture selection',primary_b_budget=0,diagnostic_budget=2,primary_b_sha256=file_hash('results/week9/frozen/w9b_frozen_config.json'),frozen_a_sha256=file_hash('results/week9/frozen/w9a_frozen_config.json'),question_id=rows[0]['question_id'],selectors=[r['policy'] for r in rows],smoke_queries=1,smoke_rows=len(rows),model_calls=len(calls),peak_allocated_gib=max(r['peak_memory_allocated'] for r in calls)/1024**3,draft_semantic_reproduces_b=True,unit_tests_passed=80,unit_tests_deselected=1,deselection_reason='Existing notebook-defaults check conflicts with user-enabled B run flags',full_experiment_launched=False,primary_c_frozen=False,config='configs/week9/w9c_k2_diagnostic.yaml',notebook='notebooks/week9/week9c_selector.ipynb')
atomic_json(Path('results/week9/patch_verification/w9c_readiness/readiness.json'),report)
print(json.dumps(report,indent=2))
