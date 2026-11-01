"""Audit completed diagnostic C and prepare controlled D; never replace primary K0."""
import json, math
from pathlib import Path
import yaml
from src.week9.provenance import ROOT,branch_guard,source_hashes,digest,file_hash
from src.week9.config import load_config,load_frozen
from src.week9.runner import load_phase,frozen_path,dependencies
from src.week9.data import load_records
from src.week9.evaluation import choose_global,policy_summary
from src.week9.manifests import freeze
from src.week9.caching import atomic_json
branch_guard()
proof=ROOT/'results/week9/patch_verification/w9d_readiness'
c=load_config(ROOT/'configs/week9/w9c_k2_diagnostic.yaml')
rows=load_phase(c,'c','fit')
records,_=load_records(c,'fit')
selectors=c['selectors']
expected={(r['inference']['question_id'],r['inference']['paper_id'],s) for r in records for s in selectors}
actual=[(r['question_id'],r['paper_id'],r['policy']) for r in rows]
assert len(actual)==len(set(actual))==len(expected)==8730 and set(actual)==expected
assert all(r['evidence_k']==2 and r['evidence_actual_count']==min(2,r['retrieval_actual_count']) for r in rows)
assert all(math.isfinite(r['f1']) and 0<=r['f1']<=1 for r in rows)
assert all(r['specification']['representation']=='raw' and r['specification']['revision'] is False for r in rows)
assert all(call['dtype']=='torch.float16' for r in rows for call in r['model_calls'])
summaries=[dict(policy=s,complexity=i,**policy_summary([r for r in rows if r['policy']==s])) for i,s in enumerate(selectors)]
base=next(s for s in summaries if s['policy']=='DraftSemantic')
options=[s for s in summaries if s['policy']=='DraftSemantic' or s['f1']>base['f1']+c['quality_tolerance']]
winner=choose_global(options,c['quality_tolerance'])
assert winner['policy']=='DraftSemantic'
b2={r['question_id']:r for r in load_phase(load_config(ROOT/'configs/week9/w9b.yaml'),'b','fit') if r['evidence_k']==2}
reference=[r for r in rows if r['policy']=='DraftSemantic']
assert all(r['selected_chunk_ids']==b2[r['question_id']]['selected_chunk_ids'] for r in reference)
prediction_mismatches=sum(r['prediction']!=b2[r['question_id']]['prediction'] for r in reference)
assert prediction_mismatches==0
manifest=json.loads((ROOT/c['output_root']/'selection/fit/w9c_manifest.json').read_text())
assert manifest['complete'] and manifest['scope']=='full' and manifest['split']=='fit'
assert manifest['source_hashes']==source_hashes()
assert digest(dict(c,_seed=manifest['seed']))==manifest['config_hash']
primary_path=ROOT/'results/week9/frozen/w9b_frozen_config.json'
primary_sha=file_hash(primary_path)
assert load_frozen(primary_path,'b')['settings']['evidence_k']==0
settings=dict(policy='DraftSemantic',selector='DraftSemantic')
path=frozen_path(c,'c')
if path.exists():
 value=load_frozen(path,'c')
 assert value['settings']==settings and value.get('diagnostic_only') is True
else:
 value=freeze(path,'c',settings,manifest,True,analysis_role='fit_diagnostic_fixed_k')
 value.update(diagnostic_only=True,diagnostic_label='Cost-aware selector for controlled K=2 diagnostics only; primary K=0 remains unchanged',decision_basis='Original C rule: retain DraftSemantic unless another selector beats it by more than .003 F1; SANE-adapted advantage is .001332. No primary C freeze or automatic primary D-H promotion.')
 atomic_json(path,value)
d=load_config(ROOT/'configs/week9/w9d.yaml')
d.update(output_root=c['output_root'],diagnostic_only=True,diagnostic_label='W9-D conditional representation diagnostic: K_E=2, DraftSemantic; primary W9-B remains K_E=0')
config_path=ROOT/'configs/week9/w9d_k2_diagnostic.yaml'
config_path.write_text(yaml.safe_dump(d,sort_keys=False))
frozen=dependencies(d,'d','fit')
assert frozen['b']['settings']['evidence_k']==2 and frozen['b'].get('diagnostic_only')
assert frozen['c']['settings']['selector']=='DraftSemantic' and frozen['c'].get('diagnostic_only')
assert file_hash(primary_path)==primary_sha
atomic_json(proof/'preparation.json',dict(c_complete=True,c_queries=len(records),c_rows=len(rows),c_frozen_diagnostic_only=True,selector=winner['policy'],budget=2,primary_b_budget=0,primary_b_sha256=primary_sha,draft_semantic_reproduces_b_all_queries=True,summaries=summaries,config=str(config_path.relative_to(ROOT)),output_root=d['output_root'],full_d_not_launched=True))
print('8730 C rows verified. Diagnostic selector frozen: DraftSemantic. Primary K0 unchanged. D dependencies pass.',flush=True)
