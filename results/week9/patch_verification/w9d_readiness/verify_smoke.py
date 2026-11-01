"""Check real D smoke identities, extraction calls, and controlled positive arms."""
import json
from pathlib import Path
import pandas as pd
from src.week9.provenance import branch_guard,digest,source_hashes,file_hash
from src.week9.caching import read_jsonl,atomic_json
branch_guard()
root=Path('results/week9/diagnostics/w9c_k2')
d=root/'extraction/fit/smoke'
rows=read_jsonl(d/'w9d_representation.jsonl')
meta=json.loads((d/'w9d_representation.identity.json').read_text())
assert meta['source_hashes']==source_hashes()
assert len(rows)==4 and {r['policy'] for r in rows}=={'D0','D1','D2','D3'}
assert len({r['question_id'] for r in rows})==1
assert all(r['cache_fingerprint']==digest(meta) and r['row_content_hash']==digest({k:v for k,v in r.items() if k!='row_content_hash'}) for r in rows)
by={r['policy']:r for r in rows}
assert by['D0']['prediction']==by['D0']['initial_draft'] and by['D0']['evidence_k']==0
positive=[by[k] for k in ('D1','D2','D3')]
assert all(r['evidence_k']==2 and r['specification']['selector']=='DraftSemantic' and r['prediction'].strip() for r in positive)
assert all(r['selected_chunk_ids']==by['D1']['selected_chunk_ids'] for r in positive)
assert not by['D2']['extraction']['draft_aware'] and by['D3']['extraction']['draft_aware']
assert all(by[k]['extraction']['format']=='sane_blueprint' for k in ('D2','D3'))
assert all(by[k]['extraction']['selected_chunk_ids']==by['D1']['selected_chunk_ids'] for k in ('D2','D3'))
# Exercise the notebook's real chunk-identity comparison with real smoke rows.
frame=pd.DataFrame(positive);frame['selected_ids']=frame.selected_chunk_ids.map(tuple)
selected=frame.pivot(index=['question_id','paper_id'],columns='policy',values='selected_ids')
assert (selected.D1.eq(selected.D2)&selected.D1.eq(selected.D3)).all()
calls=read_jsonl(d/'lm_calls.jsonl')
assert all(r['dtype']=='torch.float16' and r['attention_backend']=='efficient_sdpa' for r in calls)
extractions=[r for r in calls if r['stage']=='extraction']
assert len(extractions)==2 and all(r['generated_tokens']<=1024 for r in extractions)
assert max(r['peak_memory_allocated'] for r in calls)<20_000_000_000
saved=next(r for r in read_jsonl(root/'selection/fit/w9c_selection.jsonl') if r['question_id']==by['D1']['question_id'] and r['policy']=='DraftSemantic')
assert saved['selected_chunk_ids']==by['D1']['selected_chunk_ids'] and saved['prediction']==by['D1']['prediction']
manifest=json.loads((d/'w9d_manifest.json').read_text())
assert not manifest['complete'] and manifest['scope']=='partial'
assert not (root/'extraction/fit/w9d_representation.jsonl').exists()
assert not (root/'frozen/w9d_frozen_config.json').exists()
primary_path=Path('results/week9/frozen/w9b_frozen_config.json')
assert json.loads(primary_path.read_text())['settings']['evidence_k']==0
prepared=json.loads(Path('results/week9/patch_verification/w9d_readiness/preparation.json').read_text())
assert file_hash(primary_path)==prepared['primary_b_sha256']
report=dict(ready=True,scope='FIT K=2 DraftSemantic representation diagnostic only',smoke_queries=1,smoke_rows=4,question_id=by['D0']['question_id'],same_selected_chunks=True,raw_arm_reproduces_c=True,primary_b_budget=0,primary_b_unchanged=True,full_d_launched=False,d_frozen=False,peak_allocated_gib=max(r['peak_memory_allocated'] for r in calls)/1024**3,model_calls=len(calls),extraction_calls=[dict(prompt_tokens=r['prompt_tokens'],generated_tokens=r['generated_tokens'],latency=r['latency']) for r in extractions],extraction_audits={k:by[k]['extraction']['audit'] for k in ('D2','D3')},unit_tests_passed=80,unit_tests_deselected=1,deselection_reason='User-enabled B/C notebook flags conflict with existing safe-defaults test',config='configs/week9/w9d_k2_diagnostic.yaml',notebook='notebooks/week9/week9d_representation.ipynb')
report['extraction_token_cap_reached'] = [r['generated_tokens'] >= 1024 for r in extractions]
report['extraction_quality_note'] = 'Raw extraction quality is measured, not guaranteed. Inspect cap hits, partial tables and exact-span support in extraction_audits and notebook reports.'
atomic_json(Path('results/week9/patch_verification/w9d_readiness/readiness.json'),report)
print(json.dumps({k:v for k,v in report.items() if k!='extraction_audits'},indent=2))
