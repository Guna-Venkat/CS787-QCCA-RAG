"""Read-only scientific input audit; no model generation or phase freeze."""
import json,math
from collections import Counter
from pathlib import Path
from src.week9.config import load_config
from src.week9.runner import load_phase,dependencies,interventions
from src.week9.data import load_records
from src.week9.ke_surface import surface_analysis
from src.week9.provenance import digest,source_hashes
from src.week9.evaluation import policy_summary,choose_global
cfg=load_config('configs/week9/w9b.yaml')
rows=load_phase(cfg,'b','fit')
records,_=load_records(cfg,'fit')
expected={(r['inference']['question_id'],r['inference']['paper_id'],k) for r in records for k in (0,1,2,4)}
observed=[(r['question_id'],r['paper_id'],r['evidence_k']) for r in rows]
assert len(observed)==len(set(observed))==len(expected)==6984 and set(observed)==expected
assert all(math.isfinite(r['f1']) and 0<=r['f1']<=1 for r in rows)
assert all(r['prediction']==r['initial_draft'] for r in rows if not r['evidence_k'])
assert all(r['specification']['selector']=='DraftSemantic' and not r['specification']['revision'] and r['specification']['representation']=='raw' for r in rows)
assert all(r['evidence_actual_count']==min(r['evidence_k'],r['retrieval_actual_count']) for r in rows)
assert all(c['dtype']=='torch.float16' for r in rows for c in r['model_calls'])
assert all(len(r['selected_chunk_ids'])==len(set(r['selected_chunk_ids']))==r['evidence_actual_count'] for r in rows)
matrix,oracle=surface_analysis(rows)
options=[]
for k in (0,1,2,4):
 group=[r for r in rows if r['evidence_k']==k]
 options.append(dict(policy=f'K_E={k}',evidence_k=k,complexity=(0,1,2,4).index(k),**policy_summary(group)))
winner=choose_global(options,cfg['quality_tolerance'])
positive=choose_global(options[1:],cfg['quality_tolerance'])
manifest=json.loads(Path('results/week9/ke_surface/fit/w9b_manifest.json').read_text())
assert manifest['complete'] and manifest['scope']=='full' and manifest['source_hashes']==source_hashes()
assert digest(dict(cfg,_seed=manifest['seed']))==manifest['config_hash']
try:
 dependencies(load_config('configs/week9/w9c.yaml'),'c','fit')
except FileNotFoundError as exc:
 dependency_error=str(exc)
else:
 dependency_error=None
report=dict(b_complete=True,queries=len(records),rows=len(rows),manifest_and_row_integrity_validated=True,primary_budget=winner['evidence_k'],best_positive_budget_diagnostic_only=positive['evidence_k'],f1_by_budget={r['evidence_k']:r['f1'] for r in options},frozen_b_exists=Path('results/week9/frozen/w9b_frozen_config.json').exists(),c_dependency_error=dependency_error,c_ready=False,reason='Missing frozen B; primary K=0 makes selectors identical. Positive-budget diagnostic requires explicit separate designation.',oracle=oracle)
Path('results/week9/patch_verification/w9c_readiness/audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='oracle'},indent=2))
