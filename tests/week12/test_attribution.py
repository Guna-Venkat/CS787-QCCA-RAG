import json,copy
from pathlib import Path
import pytest
from src.week12.protocol import ROOT,load
from src.week12_attribution.protocol import stratified_sample,validate_membership,study_identity
from src.week12_attribution.experiment import perturb,causal_order,retention,gold_overlap,changes,execute
from src.week12_attribution.analysis import correlation_ci,mean_ci

def test_stratification_deterministic_without_outcome_fields():
 rows=[dict(dataset='SciFact',query_id=str(i),group=str(i),stratum=['SUPPORT','REFUTE','NOT_ENOUGH_INFO'][i%3]) for i in range(177)]
 a=stratified_sample(rows);b=stratified_sample(list(reversed(rows)))
 assert a==b and len(a)==50 and len({r['query_id'] for r in a})==50
 assert {r['stratum'] for r in a}=={'SUPPORT','REFUTE','NOT_ENOUGH_INFO'}

def test_sample_paper_preference():
 rows=[dict(dataset='QASPER',query_id=str(i),group=str(i//2),stratum='qa') for i in range(160)]
 assert len({r['group'] for r in stratified_sample(rows)})==50

def test_retention_only_positive_gain():
 assert retention(.5,.5,.4) is None
 assert retention(.3,.4,.4) is None
 assert retention(.8,.4,.6)==pytest.approx(.5)

def test_perturb_source_order_and_exact_units():
 from src.week11.budgets import FixtureTokenizer
 us=[dict(unit_id=str(i),passage_id='p',document_id='d',source_text='word '+str(i),start_char=i*10,end_char=i*10+6,document_order=[0,0,i]) for i in range(3)]
 p=perturb(us,{2,0},FixtureTokenizer());assert p['units']==[us[0],us[2]]
 assert perturb(us,set(),FixtureTokenizer())['evidence_ids']==[]
 assert causal_order([0,0,0],us)==[0,1,2] # no relevance-based tie manipulation

def test_gold_mapping_unknown_is_not_false_non_gold():
 u=dict(document_id='d',start_char=3,end_char=7)
 e=dict(evaluation=dict(gold_spans=[dict(document_id='d',start_char=5,end_char=9)]))
 assert gold_overlap(u,e) is True
 e['evaluation']['gold_spans']=[dict(document_id='d',start_char=20,end_char=22,unmapped=True)]
 assert gold_overlap(u,e) is None

def test_constant_correlations_undefined_and_clustered_ci():
 rows=[dict(relevance=i,importance=0,cluster_id=str(i//2)) for i in range(8)]
 assert correlation_ci(rows,n=50)['rho'] is None
 for r in rows:r['importance']=r['relevance']
 result=correlation_ci(rows,n=100);assert result['rho']==1 and result['n_clusters']==4

def test_real_generator_requires_manual_optin():
 with pytest.raises(ValueError,match='manual opt-in'):execute(load())

def test_synthetic_end_to_end_resume(tmp_path,monkeypatch):
 # Relocate only attribution outputs; original result/cache files are never changed.
 import src.week12_attribution.protocol as p
 import src.week12_attribution.experiment as e
 import src.week12_attribution.synthetic as s
 import src.week12_attribution.analysis as a
 path=lambda cfg,smoke=False:tmp_path/'smoke'
 for mod in (p,e,s,a):monkeypatch.setattr(mod,'directory',path)
 cfg=load();e.execute(cfg,smoke=True)
 before={f.name:f.read_bytes() for f in (path(cfg)/'perturbations').glob('*.json')}
 e.execute(cfg,smoke=True)
 assert before=={f.name:f.read_bytes() for f in (path(cfg)/'perturbations').glob('*.json')}
 result=a.analyze(cfg,True)
 assert result['status']=='COMPLETE'
 assert (path(cfg)/'analysis/QUANTITATIVE_RESULTS_FREEZE.json').exists()
 assert all(r==0 for r in result['tables']['perturbation_cost_and_flips'].physical_lm_calls)
 assert len(result['tables']['query_profiles'])==12
