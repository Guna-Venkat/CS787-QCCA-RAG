"""Saved-output G-PRE and action-verifier safety; no real estimator fitting."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.week9.provenance import digest,file_hash
from src.week9.action_verification_g import protocol,preflight,features,controller,runner


@pytest.fixture
def cfg():return protocol.config('configs/week9/w9g_action_verification.yaml')


def action(q,p,score,policy):
    return dict(question_id=q,paper_id=p,policy=policy,prediction=f'{policy} answer',initial_draft='D0 answer',
        f1=score,exact_match=float(score==1),final_prompt_tokens=100 if policy=='D0' else 200,
        total_query_prompt_tokens=100 if policy=='D0' else 300,total_query_generated_tokens=10 if policy=='D0' else 20,
        num_lm_calls=1 if policy=='D0' else 2,evidence_tokens=0 if policy=='D0' else 30,revision_latency=.1)


@pytest.fixture
def groups():
    scores={'a':(.5,.8,.5),'b':(.8,.2,.8),'c':(.4,.4,.7),'d':(.5,.505,.5)}
    return {p:{q:action(q,'paper-'+q,s[i],p) for q,s in scores.items()} for i,p in enumerate(protocol.ACTIONS)}


def test_alignment_rejects_missing_duplicate_and_paper_mismatch(groups):
    rows=[r for g in groups.values() for r in g.values()]
    pairs=[(q,'paper-'+q) for q in groups['D0']]
    assert len(protocol.aligned(rows,pairs)['F0'])==4
    with pytest.raises(ValueError,match='Missing'):protocol.aligned(rows[:-1],pairs)
    with pytest.raises(ValueError,match='alignment'):protocol.aligned(rows+[rows[0]],pairs)
    bad=copy.deepcopy(rows);bad[0]['paper_id']='other'
    with pytest.raises(ValueError,match='alignment'):protocol.aligned(bad,pairs)


def test_exact_epsilon_oracles_and_d0_preference(groups):
    summary,dist,choices=preflight.oracle(groups,('D0','F0'))
    c={(r['question_id'],r['oracle']):r for r in choices}
    assert c['a','exact']['action']=='F0' and c['b','exact']['action']=='D0'
    assert c['c','exact']['action']=='D0' and c['d','epsilon']['action']=='D0'
    assert c['d','exact']['action']=='F0'
    assert summary[0]['f1']==pytest.approx((.8+.8+.4+.505)/4)
    assert summary[1]['f1']==pytest.approx((.8+.8+.4+.5)/4)
    assert sum(r['count'] for r in dist if r['oracle']=='epsilon')==4


@pytest.mark.parametrize('scores,expected',[
    ({'D0':1,'F0':0,'F2-A':0},'D0 uniquely best'),
    ({'D0':0,'F0':1,'F2-A':0},'F0 uniquely best'),
    ({'D0':0,'F0':0,'F2-A':1},'F2-A uniquely best'),
    ({'D0':1,'F0':1,'F2-A':0},'tie involving D0/F0'),
    ({'D0':1,'F0':0,'F2-A':1},'tie involving D0/F2-A'),
    ({'D0':0,'F0':1,'F2-A':1},'tie involving F0/F2-A'),
    ({'D0':1,'F0':1,'F2-A':1},'three-way tie'),
])
def test_unique_best_seven_classes(scores,expected):assert preflight.unique_best(scores)==expected


def test_three_way_prefers_conservative_action_within_epsilon(groups):
    g={p:{'q':action('q','p',s,p)} for p,s in [('D0',.1),('F0',.8),('F2-A',.795)]}
    _,_,choices=preflight.oracle(g,protocol.ACTIONS)
    assert choices[0]['action']=='F0' and choices[1]['action']=='F2-A'


def test_unique_contribution_and_targets_from_saved_scores(groups,cfg,tmp_path):
    decision=preflight.analyze(groups,cfg,tmp_path)
    head=json.loads((tmp_path/'headroom_analysis.json').read_text())
    assert head['f2_unique_contribution']['f2_unique_exact_best_count']==1
    assert head['f2_unique_contribution']['exact_extra_oracle_f1']==pytest.approx(.3/4)
    assert head['action_label_balance']['counts']=={'HELP':1,'HURT':1,'NEUTRAL':2}
    assert (tmp_path/'w9g_preflight_decision.json').exists()
    assert decision['controller_training_started'] is False and decision['lm_calls']==0


def metric_fixture(a,b,extra):
    return {'PRE-A':{'epsilon':{'headroom':a}},'PRE-B':{'epsilon':{'headroom':b},'exact':{'f1':.5}},
            'PRE-C':{'exact':{'f1':.5+extra}}}


def test_go_no_go_thresholds_are_explicit(cfg):
    unique={'f2_unique_exact_best_rate':.01}
    d=preflight.screens(metric_fixture(.003,.06,.001),unique,cfg)
    assert d['G1']['decision']=='NO-GO' and d['G2']['decision']=='GO' and d['G3']['decision']=='NO-GO'
    assert d['recommended_route']=='G2'
    d=preflight.screens(metric_fixture(.006,.007,.001),unique,cfg)
    assert d['G1']['decision']=='GO' and d['G2']['decision']=='NO-GO'
    assert preflight.screens(metric_fixture(.003,.06,.01),unique,cfg)['recommended_route']=='G3'
    assert preflight.screens(metric_fixture(.001,.002,0),unique,cfg)['recommended_route']=='NO-GO'


@pytest.mark.parametrize('a,b,label,state',[(0,.02,1,'HELP'),(.02,0,0,'HURT'),(0,.01,0,'NEUTRAL'),(.01,0,0,'NEUTRAL'),(.4,.4,0,'NEUTRAL')])
def test_action_label_is_this_f0_delta(a,b,label,state):
    r=preflight.target(a,b)
    assert r['delta_action']==b-a and r['accept_f0']==label and r['action_state']==state


def feature_payload():
    return dict(question='What did the model add?',initial_answer='alpha gamma',candidate_answer='alpha beta',evidence_text='alpha beta delta',
        ranking=dict(sentence_cosines=[.8,.7,.6],mean_selected_cosine=.7,min_selected_cosine=.6,next_unselected_cosine=None,
            selected_vs_next_margin=None,evidence_tokens=20,source_chunk_ranks=[0,0,1],selected_chunk_count=2),
        generation_lengths=dict(initial_generated_tokens=4,candidate_generated_tokens=5))


def test_specific_change_features_and_missing_values():
    f=features.online_features(feature_payload())
    assert f['added_token_count']==f['removed_token_count']==1
    assert f['evidence_overlap_delta']==.5 and f['added_tokens_in_evidence_fraction']==1
    assert f['removed_tokens_absent_evidence_fraction']==1
    assert f['token_edit_distance']==1 and f['top_sentence_cosine']==.8
    assert np.isnan(f['next_unselected_cosine'])


@pytest.mark.parametrize('forbidden',['gold_answer','f1','HELP','oracle_action','paper_outcome','target_accept','paper_id'])
def test_no_outcome_or_paper_id_can_enter_inference_features(forbidden):
    item=feature_payload();item[forbidden]=1
    with pytest.raises(ValueError):features.online_features(item)
    f=features.online_features(feature_payload());f[forbidden]=1
    with pytest.raises(ValueError):features.feature_matrix([f],['V0'])


def test_feature_registry_and_ablation(cfg):
    f=features.online_features(feature_payload())
    x=features.feature_matrix([f],cfg['feature_groups'])
    assert x.shape[1]==len(sum(features.GROUPS.values(),[]))
    assert set(features.feature_matrix([f],['V0']).columns)==set(features.GROUPS['V0'])
    assert 'Unavailable' in features.AVAILABILITY['V2_confidence']


def test_five_paper_oof_folds_have_no_overlap():
    meta=pd.DataFrame([dict(question_id=str(i),paper_id=str(i//2)) for i in range(24)])
    ids,splits=controller.paper_folds(meta,5)
    assert len(set(ids))==5
    for tr,va in splits:assert not set(meta.iloc[tr].paper_id)&set(meta.iloc[va].paper_id)
    for _,g in meta.assign(fold=ids).groupby('paper_id'):assert g.fold.nunique()==1
    with pytest.raises(ValueError):controller.paper_folds(pd.concat([meta,meta.iloc[:1]]),5)


def test_post_generation_keep_still_pays_f0(groups):
    meta=pd.DataFrame([dict(question_id=q,paper_id='paper-'+q) for q in groups['D0']])
    kept=controller.applied_outcomes(groups,meta,np.zeros(len(meta),bool))
    assert all(r['num_lm_calls']==2 and r['total_query_prompt_tokens']==300 for r in kept)
    assert [r['prediction'] for r in kept]==[r['prediction'] for r in groups['D0'].values()]


def test_threshold_selection_uses_downstream_quality_not_default_half(groups,cfg):
    meta=pd.DataFrame([dict(question_id=q,paper_id='paper-'+q) for q in groups['D0']])
    selected,table=controller.choose_threshold(np.array([.35,.2,.1,.1]),groups,meta,cfg)
    assert selected['threshold']!=.5 and selected['f1']>.5
    assert len(table)==len(cfg['thresholds'])
    rows=controller.applied_outcomes(groups,meta,np.array([True,False,False,False]))
    m=controller.metrics(rows)
    assert m['harm_rejection_rate']==1 and m['help_retention_rate']==1
    assert m['accept_on_help']==1 and m['accept_on_hurt']==0


def test_training_off_fails_before_any_input_read(monkeypatch,cfg):
    monkeypatch.setattr(runner,'config',lambda _:cfg)
    monkeypatch.setattr(runner,'inputs',lambda _:pytest.fail('Training must stop before input work'))
    with pytest.raises(PermissionError):runner.main(['--config','fixture','--stage','train'])
    with pytest.raises(PermissionError):controller.train_manually(cfg,{}, {},enabled=False)


def test_preflight_cannot_generate_or_train(monkeypatch,tmp_path,groups,cfg):
    import src.week9.backend as backend
    import src.week9.retrieval as retrieval
    monkeypatch.setattr(backend.SaraBackend,'__init__',lambda *a,**kw:pytest.fail('No model loading'))
    monkeypatch.setattr(retrieval.LocalEncoder,'__init__',lambda *a,**kw:pytest.fail('No encoder loading'))
    monkeypatch.setattr(controller,'train_manually',lambda *a,**kw:pytest.fail('No controller training'))
    local=dict(cfg,output_root=str(tmp_path/'new-g'))
    bundle=dict(groups=groups,pairs=[(q,'paper-'+q) for q in groups['D0']],input_hashes={})
    monkeypatch.setattr(runner,'config',lambda _:local)
    monkeypatch.setattr(runner,'inputs',lambda _:bundle)
    monkeypatch.setattr(runner,'identity',lambda *a:dict(schema='fixture'))
    runner.main(['--config','fixture','--stage','preflight'])
    manifest=json.loads((tmp_path/'new-g/preflight/preflight_manifest.json').read_text())
    assert manifest['lm_calls']==0 and not manifest['controller_training_started']


def test_legacy_g_hierarchy_cannot_be_output(tmp_path):
    import yaml
    cfg=yaml.safe_load(Path('configs/week9/w9g_action_verification.yaml').read_text())
    cfg['output_root']='results/week9/controller'
    p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='historical'):protocol.config(p)


def test_branch_guard_requires_gagan(monkeypatch):
    import subprocess
    from src.week9.provenance import branch_guard
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:'main\n')
    with pytest.raises(RuntimeError,match='gagan'):branch_guard()


def test_frozen_input_hash_rejects_changes(tmp_path):
    p=tmp_path/'old-g-result.json';p.write_text('{}');before=(p.read_bytes(),p.stat().st_mtime_ns)
    protocol.verify_hashes({str(p):file_hash(p)})
    assert before==(p.read_bytes(),p.stat().st_mtime_ns)
    p.write_text('changed')
    with pytest.raises(ValueError):protocol.verify_hashes({str(p):digest('not file bytes')})


def test_notebook_default_preflight_only(monkeypatch):
    import subprocess
    calls=[]
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:calls.append(a[0]))
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:'gagan\n')
    path=Path('notebooks/week9/week9g_action_verification.ipynb')
    nb=json.loads(path.read_text());ns={}
    for c in nb['cells']:
        if c['cell_type']=='code':exec(compile(''.join(c['source']),'w9g_action_notebook','exec'),ns)
    assert ns['RUN_PREFLIGHT'] and not ns['RUN_CONTROLLER_TRAINING'] and not ns['RUN_FULL']
    assert len(calls)==1 and calls[0][calls[0].index('--stage')+1]=='preflight'
