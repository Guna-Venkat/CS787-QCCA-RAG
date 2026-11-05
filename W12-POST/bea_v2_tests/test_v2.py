import json
from copy import deepcopy
import numpy as np
import pytest
from src.post_w12.bea_v2.protocol import FEATURES,digest,Store
from src.post_w12.bea_v2.allocator import utility,allocate,sequence
from src.post_w12.bea_v2.models import fit_binary
from src.post_w12.bea_v2.data import feature_rows

PARAM=dict(eta_cov=.5,lambda_red=.5,utility_weight='p_help_only',min_parent_sentences=2)
class Tok:
    def encode(self,text,**kw):return list(map(ord,text))

def fixture():
    p=dict(unit_id='p',passage_id='p',document_id='d',unit_type='PASSAGE',source_text='A. B. C. D.',start_char=0,end_char=11,document_order=[0,0,0,0],token_count=11)
    s=[dict(p,unit_id=str(i),unit_type='SENTENCE',source_text=t,start_char=3*i,end_char=3*i+2,token_count=2,document_order=[0,0,i,0]) for i,t in enumerate(['A.','B.','C.','D.'])]
    scores=dict(relevance=[.9,.8,.7,.6],aspect_scores=[[.9,.8,.7,.6]],pairwise=np.eye(4).tolist())
    return [p],s,scores

def test_utility_multiplication_and_harm_is_diagnostic_only():
    rel=np.array([.8]);asp=np.array([[.8]]);pair=np.array([[1.]])
    a=utility(0,set(),rel,asp,pair,[.4],[.1],PARAM)
    assert a['value']==pytest.approx(.4*1.2)
    assert utility(0,set(),rel,asp,pair,[.4],[.99],PARAM)['value']==a['value']
    assert utility(0,set(),rel,asp,pair,[0.],[.4],PARAM)['value']==0

def test_dynamic_coverage_and_redundancy():
    rel=np.array([.8,.8]);asp=np.array([[.8,.8]]);pair=np.ones((2,2))
    a=utility(1,set(),rel,asp,pair,[.5,.5],[0,0],PARAM);b=utility(1,{0},rel,asp,pair,[.5,.5],[0,0],PARAM)
    assert b['coverage_gain']==0 and b['value']<a['value']

def test_sequential_full_updates_hypothetical_zero_weight_context():
    p,s,sc=fixture();rel=np.array(sc['relevance']);asp=np.array(sc['aspect_scores']);pair=np.ones((4,4))
    seq=sequence(range(4),set(),4,rel,asp,pair,[.5,.5,.5,.5],[0,0,0,0],PARAM,s,full=True)
    assert len(seq)==4 and seq[0]['redundancy']==0 and seq[1]['redundancy']==1
    assert seq[1]['coverage_gain']==0

def test_exact_budget_source_order_and_repeatability():
    p,s,sc=fixture();prob=dict(p_help=[.8]*4,p_harm=[.1]*4)
    a=allocate(p,s,sc,prob,Tok(),16,PARAM,0);b=allocate(p,s,sc,prob,Tok(),16,PARAM,0)
    assert a['evidence_tokens']<=16 and a['trace']==b['trace']
    assert [u['start_char'] for u in a['units']]==sorted(u['start_char'] for u in a['units'])
    assert [len(t['sentences']) for t in a['trace'][:2]]==[1,1]

def test_no_evidence_on_zero_help_and_threshold_stopping():
    p,s,sc=fixture()
    a=allocate(p,s,sc,dict(p_help=[0.]*4,p_harm=[.9]*4),Tok(),1024,PARAM,0)
    assert not a['units'] and a['stop_reason']=='NO_POSITIVE_VALUE'
    a=allocate(p,s,sc,dict(p_help=[.9]*4,p_harm=[.1]*4),Tok(),1024,PARAM,100)
    assert not a['units'] and a['stop_reason']=='LOW_MARGINAL_VALUE'

def test_feature_allowlist_ignores_gold_and_loo():
    p,s,sc=fixture();x=dict(pool=p,sentences=s,scores=sc,passage_scores=[.8])
    a=feature_rows(x,256);x.update(gold='secret',prediction='secret',importance=100,category='HELPFUL')
    assert a==feature_rows(x,256) and list(a[0])==FEATURES
    assert not any(k in FEATURES for k in ['gold','importance','category','prediction','f1','correct'])

def test_model_reproducibility_and_probability_bounds():
    c=dict(model=dict(C=1,max_iter=1000),seed=42);x=np.array([[0],[1],[2],[3]],float);y=np.array([0,0,1,1])
    a=fit_binary(x,y,c).predict_proba(x);b=fit_binary(x,y,c).predict_proba(x)
    assert np.allclose(a,b) and np.all((a>=0)&(a<=1))
    with pytest.raises(ValueError,match='one-class'):fit_binary(x,np.zeros(4),c)

def test_group_split_semantics():
    from sklearn.model_selection import GroupKFold
    for groups in [['paper-a','paper-a','paper-b','paper-c','paper-d','paper-e','paper-f'],['cluster-a','cluster-a','cluster-b','cluster-c','cluster-d','cluster-e','cluster-f']]:
        for tr,te in GroupKFold(5).split(np.zeros(len(groups)),groups=groups):assert not set(groups[i] for i in tr)&set(groups[i] for i in te)

def test_predictor_hash_changes_identity_and_immutable_resume(tmp_path,monkeypatch):
    from src.post_w12.bea_v2 import protocol
    monkeypatch.setattr(protocol,'ROOT',tmp_path)
    path=tmp_path/'results/post_w12/bea_v2/test'
    a=Store(path,dict(model_hash='a',fold=1,features=digest(FEATURES)));a.put(['q'],dict(value=1));assert a.get(['q'])==dict(value=1)
    with pytest.raises(ValueError):Store(path,dict(model_hash='b')).get(['q'])
    with pytest.raises(ValueError):a.put(['q'],dict(value=2))

def test_historical_labels_and_sources_present():
    from src.post_w12.bea_v2.protocol import ROOT,read,sha
    p=ROOT/'results/week12/attribution/v1/analysis';f=read(p/'QUANTITATIVE_RESULTS_FREEZE.json')
    assert sha(p/'unit_importance.csv')==f['tables']['unit_importance']
    assert sha(ROOT/'results/week12/W13_FINAL_METHOD_FREEZE.json')=='d0188a3582d6aead0a3a31f9f15bad9b51a88a6b475b05ac79ae38e58f05366b'

def test_notebook_defaults_and_no_gpu_smoke():
    from src.post_w12.bea_v2.protocol import ROOT
    n=json.loads((ROOT/'W12-POST/BEA_QBGF_V2.ipynb').read_text());code='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
    for flag in ['RUN_PILOT = False','RUN_UNSEEN_EXPANSION = False','RUN_NEW_ATTRIBUTION = False','OVERWRITE = False','RESUME = True']:assert flag in code
    assert 'calibrate(cfg)' in code

def test_source_states_and_full_prerequisite():
    p,s,sc=fixture();prob=dict(p_help=[.8]*4,p_harm=[.1]*4)
    a=allocate(p,s,sc,prob,Tok(),100,PARAM,0)
    assert [t['kind'] for t in a['trace']]==['SENTENCES','SENTENCES','FULL']
    assert a['full_sources']==1 and a['source_positive_contributions']['p']>=2
    b=allocate(p,s,sc,prob,Tok(),100,dict(PARAM,min_parent_sentences=999),0)
    assert [len(t['sentences']) for t in b['trace']]==[1,1,2]

def test_negative_ig_cannot_create_positive_value():
    a=utility(1,{0},np.array([1.,.1]),np.array([[1.,.1]]),np.ones((2,2)),[.9,.9],[0.,0.],PARAM)
    assert a['ig']<0 and a['value']==0

def test_heldout_group_guard_fails_hard(tmp_path,monkeypatch):
    import joblib
    from src.post_w12.bea_v2 import models,protocol
    monkeypatch.setattr(protocol,'ROOT',tmp_path)
    c=dict(output_root='results/post_w12/bea_v2/test');folder=protocol.root(c);(folder/'utility_model').mkdir(parents=True)
    (folder/'CROSSFIT_FOLDS.json').write_text(json.dumps(dict(assignments={'QASPER':{'q':0}})))
    path=folder/'utility_model/fold_QASPER_0.joblib'
    joblib.dump(dict(training_groups=['paper-a'],training_queries=['q'],heldout_groups=['paper-a'],feature_names=FEATURES),path)
    monkeypatch.setattr(models,'verify_models',lambda c:dict(files={'utility_model/fold_QASPER_0.joblib':'h'}))
    with pytest.raises(ValueError,match='leaked'):models.predict(c,'QASPER','q','paper-a',[])

def test_exact_historical_label_signs():
    import pandas as pd
    from src.post_w12.bea_v2.protocol import ROOT
    data=pd.read_csv(ROOT/'results/week12/attribution/v1/analysis/unit_importance.csv')
    expected=np.where(data.importance>0,'HELPFUL',np.where(data.importance<0,'HARMFUL','NEUTRAL'))
    assert np.array_equal(expected,data.category.to_numpy())
    assert np.allclose(data.full_score-data.ablated_score,data.importance,atol=1e-12)

def test_v1_allocator_is_unchanged():
    from src.post_w12.bea_v2.protocol import ROOT,read,sha
    f=read(ROOT/'results/post_w12/bea_qbgf/v3/BEA_QBGF_V1_FREEZE.json')
    for name in ['src/post_w12/bea_qbgf/allocator.py','src/post_w12/bea_qbgf/resources.py']:
        assert sha(ROOT/name)==f['identity']['implementation'][name]

def test_full_behavior_counters_match_v2_transitions():
    from src.post_w12.bea_qbgf.allocator import behavior
    p,s,sc=fixture();a=allocate(p,s,sc,dict(p_help=[.8]*4,p_harm=[.1]*4),Tok(),100,PARAM,0)
    b=behavior(a,s,sc)
    assert b['full_eligible_query'] and b['full_chosen']==1 and b['full_considered']>=1

def test_synthetic_analysis_and_four_method_plots(tmp_path,monkeypatch):
    import pandas as pd
    from src.post_w12.bea_v2 import protocol,analysis
    monkeypatch.setattr(protocol,'ROOT',tmp_path)
    c=dict(
        output_root='results/post_w12/bea_v2/test',study='synthetic',budgets=[256,1024],
        methods=['FIXED_SENTENCE','QBGF_FULL','BEA_QBGF_V1','BEA_V2'],statistics=dict(resamples=20,seed=42),
        decision=dict(qa_margin=.003,macro_f1_margin=.01,maximum_clearly_degraded_cells=1,minimum_quality_gain=.003,uninformative_mean_auc_ceiling=.5,maximum_empty_fraction=.1,maximum_near_full_fraction=.9))
    cohort=dict(selected={d:[dict(query_id=str(i),group=str(i)) for i in range(2)] for d in protocol.DATASETS})
    f=dict(identity=dict(synthetic=True),identity_hash='fixture');monkeypatch.setattr(analysis,'freeze',lambda c:({}, {}, {}, cohort, f))
    out=protocol.root(c);(out/'utility_model').mkdir(parents=True);pd.DataFrame(dict(roc_auc=[.6],target=['HELPFUL'])).to_csv(out/'utility_model/fold_metrics.csv',index=False)
    def row(d,q,m,b):
        return dict(dataset=d,query_id=q,cluster_id=q,method=m,budget=b,f1=.5,exact_match=0,correct=1,gold='SUPPORT',prediction='SUPPORT',
                    evidence_tokens=20,unused_budget=b-20,budget_utilization=20/b,prompt_tokens=40,generated_tokens=2,stop_reason='LOW_MARGINAL_VALUE',
                    behavior=dict(selected_units=1,represented_parents=1,mean_pairwise_redundancy=None,aspect_coverage=.8,fidelity_upgrades=0,full_units=0,
                                  source_expansion=False,mean_selected_p_help=.2,mean_selected_p_harm=.1,mean_selected_D=.1,mean_transition_IG=.5,mean_transition_U=.05))
    monkeypatch.setattr(analysis,'baseline',lambda c,v1,f,d,q,m,b:row(d,q,m,b))
    for d,members in cohort['selected'].items():
        for member in members:
            for b in c['budgets']:
                key=[d,member['query_id'],'BEA_V2',b]
                protocol.Store(out/'pilot/rows',f['identity']).put(key,row(d,member['query_id'],'BEA_V2',b))
                protocol.Store(out/'pilot/selections',f['identity']).put(key,dict(selection=dict(trace=[dict(step=1,sentences=[dict(unit_id='u',p_help=.2,p_harm=.1,ig=.5,value=.05)])])))
    result=analysis.analyze(c)
    assert len(result['summary'])==24 and len(result['paired_comparisons'])==18
    assert (out/'pilot/PILOT_REPORT.md').exists() and len(list((out/'pilot/figures').glob('*.png')))==2

def test_harm_probabilities_cannot_change_evidence_or_stopping():
    p,s,sc=fixture()
    a=allocate(p,s,sc,dict(p_help=[.3,.6,.4,.2],p_harm=[0.]*4),Tok(),100,PARAM,.001)
    b=allocate(p,s,sc,dict(p_help=[.3,.6,.4,.2],p_harm=[1.,.9,.8,.7]),Tok(),100,PARAM,.001)
    assert a['evidence_ids']==b['evidence_ids'] and a['stop_reason']==b['stop_reason']
    assert [(t['parent'],t['kind'],t['value']) for t in a['trace']]==[(t['parent'],t['kind'],t['value']) for t in b['trace']]

def calibration_fixture():
    from src.post_w12.bea_v2.protocol import ROOT,read,DATASETS
    import pandas as pd
    c=read(ROOT/'W12-POST/bea_v2_config.json');rows=[]
    for tau in [0,.00025]:
        for d in DATASETS:
            for b in c['budgets']:
                for i in range(3):rows.append(dict(threshold=tau,dataset=d,budget=b,query_id=str(i),coverage=.8,units=4,utilization=.5,stop_reason='LOW_MARGINAL_VALUE' if tau else 'BUDGET_EXHAUSTED'))
    return pd.DataFrame(rows),c

def test_every_cell_must_avoid_collapse_not_just_global_mean():
    from src.post_w12.bea_v2.stages import calibration_summary
    x,c=calibration_fixture();summary,cells=calibration_summary(x,c)
    assert summary[1]['eligible'] and summary[1]['all_cells_eligible']
    mask=(x.threshold>0)&(x.dataset=='QASPER')&(x.budget==1024)
    x.loc[mask,'units']=1;x.loc[mask,'utilization']=.02
    summary,cells=calibration_summary(x,c)
    assert summary[1]['mean_utilization']>.4 and not summary[1]['eligible']
    bad=next(r for r in cells if r['dataset']=='QASPER' and r['budget']==1024 and r['threshold']>0)
    assert 'low_mean_utilization' in bad['failure_reasons'] and 'too_many_trivial_contexts' in bad['failure_reasons']

def test_any_empty_context_blocks_global_calibration():
    from src.post_w12.bea_v2.stages import calibration_summary
    x,c=calibration_fixture();x.loc[(x.threshold>0)&(x.dataset=='HotpotQA')&(x.budget==256)&(x.query_id=='0'),'units']=0
    summary,cells=calibration_summary(x,c)
    assert not summary[1]['eligible'] and summary[1]['empty_fraction']>0

def test_coverage_retention_checked_per_cell_without_quality_inputs():
    from src.post_w12.bea_v2.stages import calibration_summary
    x,c=calibration_fixture();x['f1']=12345;x['gold_label']='FORBIDDEN'
    reference=calibration_summary(x,c)
    x['f1']=-999;x['gold_label']='OTHER'
    assert calibration_summary(x,c)==reference
    x.loc[(x.threshold>0)&(x.dataset=='SciFact')&(x.budget==1024),'coverage']=.2
    summary,cells=calibration_summary(x,c)
    assert not summary[1]['eligible']
