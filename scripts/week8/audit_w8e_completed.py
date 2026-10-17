"""Verify a completed frozen W8-E run and its executed notebooks without retuning."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
from transformers import AutoTokenizer
from src.week8.data import load_config
from src.week8.selective_controller import controller_sources
out=ROOT/'results/week8/w8e_controller'
s=json.loads((out/'controller_summary.json').read_text())
f=json.loads((out/'frozen_controller.json').read_text())
assert s['source_hashes']==controller_sources()
assert s['selected_config']==f['selected_config']
assert s['fit_selected_shortfall_from_grid_best']<=.003+1e-12
assert s['oracle_gate'] is False and s['frozen_before_internal_val'] is True
cfg=load_config()
tok=AutoTokenizer.from_pretrained(str(ROOT/cfg['sara']['checkpoint_path']),use_fast=True,local_files_only=True)
audit={}
for split,count in [('fit',1746),('internal_val',342)]:
 raw=[json.loads(l) for l in (ROOT/f'results/week8/w8e_gate_inputs/canonical_diagnostics_{split}.jsonl').read_text().splitlines()]
 assert len(raw)==count and len({r['question_id'] for r in raw})==count
 assert all(r['greedy_verified'] and r['diagnostic_source']=='greedy_canonical_match' for r in raw)
 mismatch=[];empty=[]
 for r in raw:
  if not r['draft_answer'].strip():empty.append(r['question_id']);continue
  ids=r['token_ids']
  if ids[-1]==tok.eos_token_id:ids=ids[:-1]
  teacher=tok(' '+r['draft_answer'],add_special_tokens=False)['input_ids']
  if ids!=teacher:mismatch.append(r['question_id'])
 assert all(tok.decode(r['token_ids'],skip_special_tokens=True).strip()==r['draft_answer'].strip() for r in raw)
 audit[split]={'queries':count,'canonical_greedy_verified':count,'nonempty_retokenized_draft_token_differences':mismatch,'decoded_canonical_text_verified':True,'masking_target_note':'W8-D scores the retokenized frozen draft text; it does not claim to score original generation token IDs. W8-E uncertainty uses actual generated token IDs, including EOS where generated.','empty_draft_queries':empty}
 for name in s['internal_val_metrics']:
  p=out/(f'{name}_internal_val_decisions.csv' if split=='internal_val' else f'{name}_fit_oof_replay.csv')
  d=pd.read_csv(p,dtype={'question_id':str,'paper_id':str})
  assert len(d)==count and d.question_id.nunique()==count
  promote=d.selected_action.eq('PROMOTE')
  keys=['candidate_passage_index','semantic_rank','bm25_rank']
  assert d.loc[promote,keys].notna().all().all()
  assert d.loc[~promote,keys].isna().all().all()
  for key in keys:assert (d.loc[promote,key]==d.loc[promote,'proposed_'+key]).all()
notebooks={}
for name in ['week8e_controller.ipynb','week8d_generator_interaction.ipynb','week8d_cached_closeout.ipynb']:
 n=json.loads((ROOT/'notebooks/week8'/name).read_text())
 code=[c for c in n['cells'] if c['cell_type']=='code' and ''.join(c['source']).strip()]
 assert all(c['execution_count'] is not None for c in code)
 assert not any(o.get('output_type')=='error' for c in code for o in c.get('outputs',[]))
 notebooks[name]={'executed_code_cells':len(code),'errors':0}
result={'source_hashes_match':True,'selected_policy':s['selected_policy'],'selected_global_quality_gap':s['fit_selected_shortfall_from_grid_best'],'token_alignment':audit,'notebooks':notebooks,'candidate_identities_verified':True}
(out/'final_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
