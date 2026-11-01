"""Read primary artifacts only; deterministic formula checks, no model execution."""
from pathlib import Path
import math,json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];p=ROOT/'results/week0'
a=pd.read_csv(p/'evidence_analysis/ablation_passage_scores.csv');t=pd.read_csv(p/'evidence_analysis/attention_passage_scores.csv');l=pd.read_csv(p/'layer_analysis/layer_similarity_long.csv');rng=np.random.RandomState(42);att=[];abl=[]
for _,g in a.groupby('question_id',sort=False):
 for layer in [0,4,8,16,24,31]:
  masses=[max(.01,math.exp(-.25*i)+(.35*layer/31 if bool(r.is_gold_evidence) else 0)+rng.normal(0,.05)) for i,r in enumerate(g.itertuples())]
  att.extend((round(m,4),round(m/sum(masses),4)) for m in masses)
 for i,r in enumerate(g.itertuples()):abl.append(round(max(0.,.05*math.exp(-.3*i)+(.45 if bool(r.is_gold_evidence) else 0)+rng.normal(0,.02)),4))
rng=np.random.RandomState(42);sim=[]
for r in l.itertuples():sim.append(round(float(np.clip(.35+.1*math.sin(r.layer/31*math.pi)+(.12*r.layer/31 if bool(r.is_gold_evidence) else 0)+rng.normal(0,.02),-1,1)),4))
checks=dict(generic_attention_rows=len(t),generic_ablation_rows=len(a),generic_layer_rows=len(l),mock_attention_all_values_match=bool(np.allclose(t[['attention_mass','attention_share']].values,att,atol=1e-12,rtol=0)),mock_ablation_all_values_match=bool(np.allclose(a.ablation_importance,abl,atol=1e-12,rtol=0)),mock_layer_all_values_match=bool(np.allclose(l.cosine_sim_with_query,sim,atol=1e-12,rtol=0)),model_execution=False)
print(json.dumps(checks,indent=2))
(ROOT/'results/weekly_reports/week_00/PrimaryRecomputationChecks.json').write_text(json.dumps(checks,indent=2)+'\n')
for sub in ['gpu_pilot','gpu_pilot/intervention_validation','final40']:
 print('\n'+sub)
 for name in ['ablation_results','replacement_results','removal_results','removal_vs_replacement','question_level_features']:
  path=p/sub/(name+'.csv')
  if not path.exists():continue
  d=pd.read_csv(path);cols=[c for c in d if 'delta' in c and pd.api.types.is_numeric_dtype(d[c])];print(name,'n=',len(d),'queries=',d.question_id.nunique() if 'question_id'in d else '')
  for c in cols:print(c,'mean',d[c].mean(),'median',d[c].median(),'positive',int((d[c]>0).sum()),'negative',int((d[c]<0).sum()))
