"""Synthetic-only tests supplied for the user; run before allocation-only calibration."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from src.post_w12.bea_qbgf.allocator import aspects,signals,marginal,allocate,behavior
from src.post_w12.bea_qbgf.protocol import Store,digest

PARAMS=dict(lambda_red=.5,eta_cov=.5,tau_stop=.001,tau_expand=.001,tau_context=.05,min_parent_sentences=2)

class Tokenizer:
    def encode(self,text,**kwargs):return list(map(ord,text))
    def apply_chat_template(self,messages,**kwargs):return messages[0]['content']+' ANSWER:'

def fixture():
    parent=dict(unit_id='p',passage_id='p',document_id='doc',source_text='A. B. C.',start_char=0,end_char=8,document_order=[0,0,0,0],unit_type='PASSAGE')
    sentences=[dict(parent,unit_id=str(i),source_text=t,start_char=a,end_char=a+2,document_order=[0,0,i,0],unit_type='SENTENCE') for i,(t,a) in enumerate([('A.',0),('B.',3),('C.',6)])]
    scores=dict(relevance=[1,.95,.9],aspect_scores=[[0,0,0]],pairwise=np.eye(3).tolist())
    return [parent],sentences,scores

def test_query_only_aspects():
    q='Explain group structure, tournament workload, and why groups changed?'
    assert aspects(q)==aspects(q)==['Explain group structure','tournament workload','why groups changed']
    assert aspects('Why?')==['Why?']
    with pytest.raises(ValueError):aspects('')

def test_global_scores_are_cosines_not_passage_softmax():
    x=signals([[1,0],[1,0],[0,1]],[1,0],[[0,1]])
    assert x['relevance']==[1,1,0] and sum(x['relevance'])==2
    assert x['aspect_scores']==[[0,0,1]]

def test_marginal_coverage_and_redundancy_not_duplicate_novelty_reward():
    rel=np.array([.9,.89,.8]);asp=np.array([[.9,.89,0],[0,0,.9]]);pair=np.array([[1,.95,0],[.95,1,0],[0,0,1]])
    duplicate=marginal(1,{0},rel,asp,pair,PARAMS);new=marginal(2,{0},rel,asp,pair,PARAMS)
    assert duplicate['redundancy']==.95 and duplicate['novelty']==pytest.approx(.05)
    assert duplicate['coverage_gain']==0 and new['coverage_gain']==.9 and new['gain']>duplicate['gain']

def test_parent_accumulation_and_marginal_upgrade():
    pool,s,score=fixture();out=allocate(pool,s,score,Tokenizer(),20,PARAMS)
    assert [r['kind'] for r in out['trace']]==['SENTENCE','SENTENCE','FULL']
    assert out['choices'][0]['state']=='FULL' and len(out['choices'][0]['accumulated'])==2
    assert out['fidelity_upgrades']==1 and out['text']=='A. B. C.' and out['evidence_tokens']==10
    assert out['trace'][-1]['dc']==2 and out['trace'][-1]['gain']==pytest.approx(.9)

def test_stopping_unused_budget_and_no_gold_inputs():
    p,s,score=fixture();params=dict(PARAMS,tau_stop=1,tau_expand=1)
    out=allocate(p,s,score,Tokenizer(),256,params)
    assert not out['units'] and out['unused_budget']==256 and out['stop_reason']=='LOW_MARGINAL_VALUE'
    with pytest.raises(TypeError):allocate(p,s,score,Tokenizer(),256,params,gold_answer='leak')

def test_exact_accounting_source_order_and_deterministic_rerun():
    from src.week12.algorithm import serialize
    p,s,score=fixture();score['relevance']=[.8,.9,1.]
    a=allocate(p,s,score,Tokenizer(),8,PARAMS);b=allocate(p,s,score,Tokenizer(),8,PARAMS)
    assert a['evidence_ids']==b['evidence_ids'] and a['trace']==b['trace']
    assert a['evidence_tokens']<=8 and serialize(a['units'],Tokenizer())['evidence_ids']==a['evidence_ids']
    assert [u['start_char'] for u in a['units']]==sorted(u['start_char'] for u in a['units'])

def test_behavior_full_covers_underlying_sentences():
    p,s,score=fixture();out=allocate(p,s,score,Tokenizer(),20,PARAMS);b=behavior(out,s,score)
    assert b['full_units']==1 and b['selected_units']==1 and b['represented_parents']==1
    assert b['mean_pairwise_redundancy']==0

def test_immutable_cache_resume_and_wrong_identity(tmp_path,monkeypatch):
    from src.post_w12.bea_qbgf import protocol
    monkeypatch.setattr(protocol,'ROOT',tmp_path)
    st=Store(tmp_path/'results/post_w12/bea_qbgf/test/rows',dict(v=1))
    assert st.get('q') is None
    st.put('q',dict(answer='a'));st.put('q',dict(answer='a'));assert st.get('q')['answer']=='a'
    with pytest.raises(ValueError):st.put('q',dict(answer='changed'))
    with pytest.raises(ValueError):Store(st.path,dict(v=2)).get('q')
    with pytest.raises(ValueError):protocol.save(tmp_path/'results/week12/forbidden.json',dict(x=1))

def test_cache_missing_regeneration_and_resume(tmp_path,monkeypatch):
    from src.post_w12.bea_qbgf import runner,protocol
    from src.week12.reader import prompt_ids
    import transformers,src.week12.reader,torch
    monkeypatch.setattr(protocol,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    tok=Tokenizer();c=dict(output_root='results/post_w12/bea_qbgf/test',budgets=[256],cache=dict(reuse_compatible=True),minimum_free_disk_gib=0)
    parent=dict(models=dict(Qwen=dict(id='fake',revision='fake')),labels=['SUPPORT','REFUTE','NOT_ENOUGH_INFO'])
    cohort=dict(selected=dict(HotpotQA=[dict(query_id='q')]))
    frozen=dict(identity=dict(test=True),identity_hash=digest(dict(test=True)))
    e=dict(dataset='HotpotQA',query_id='q',paper_id='p',cluster_id='p',split='dev',task_type='multihop_qa',question='Q?',inference=dict(source='fixture'),evaluation=dict(gold_answer=['answer']))
    monkeypatch.setattr(runner,'freeze',lambda c:(parent,{},cohort,frozen));monkeypatch.setattr(runner,'load_val',lambda *a:[e])
    monkeypatch.setattr(runner,'read',lambda p:{})
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:tok)
    calls=[]
    class Reader:
        def __init__(self,*a):pass
        def generate(self,e,sel):
            calls.append(sel['text'])
            return dict(prediction='answer',reader='BASE_FP16',model_hash=digest([parent['models']['Qwen'],'BASE_FP16',None]),prompt_hash=digest(prompt_ids(tok,e['question'],e['task_type'],sel['evidence_ids'],parent['labels'])),prompt_tokens=100,generated_tokens=1)
        def close(self):pass
    monkeypatch.setattr(src.week12.reader,'Reader',Reader)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True);monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(100*1024**3,100*1024**3))
    monkeypatch.setattr(runner,'compatible_prediction',lambda *a:(None,None,['rejected: incompatible old cache']))
    _,sel,results,gen=runner.stores(c,frozen)
    for m in runner.METHODS:
        selection=dict(method=m,budget=256,text='fixture',evidence_ids=[1],evidence_tokens=1,evidence_hash='h',selected_unit_ids=['u'],lineage=dict(candidate_ids=['p'],candidate_content_hashes=['hash']))
        sel.put(['HotpotQA','q',m,256],dict(selection=selection,source_hash=digest(e['inference']),behavior={}))
    first=runner.generate(c);assert len(first)==3 and len(calls)==1 # same exact prompt reuses generation, conceptual rows remain three
    second=runner.generate(c);assert second==first and len(calls)==1
    manifest=(protocol.output(c)/'pilot/cache_manifest.csv').read_text()
    assert 'POST-W12 local completed row' in manifest

def test_frozen_dependencies_unchanged():
    from src.post_w12.bea_qbgf.protocol import ROOT,sha,FREEZE_SHA
    assert sha(ROOT/'results/week12/W13_FINAL_METHOD_FREEZE.json')==FREEZE_SHA
    assert sha(ROOT/'src/week12/algorithm.py')=='433222886e997116cea673df24d32e2a3f81210663cfabcb5b689720f5bb5132'
    assert sha(ROOT/'src/week12/reader.py')=='ade75ad9d2630c135ef6e12866a4d104fba54adfc947479f7dcf7c89f544b94c'

def test_notebook_does_not_enable_expansion_or_attribution():
    from src.post_w12.bea_qbgf.protocol import ROOT
    nb=json.loads((ROOT/'W12-POST/BEA_QBGF.ipynb').read_text())
    code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    assert 'AUTO_RUN_MISSING = True' in code
    assert 'RUN_EXPANDED = False' in code and 'RUN_ATTRIBUTION = False' in code and 'RUN_TESTS = True' in code

def test_full_recomputes_duplicate_redundancy_sequentially():
    from src.post_w12.bea_qbgf.allocator import sequential_full
    rel=np.array([.8,.8,.8]);asp=np.array([[.9,.9,.9]]);pair=np.ones((3,3))
    gain,steps=sequential_full([0,1,2],set(),rel,asp,pair,PARAMS)
    assert steps[0]['coverage_gain']==.9 and steps[1]['coverage_gain']==0
    assert steps[0]['redundancy']==0 and steps[1]['redundancy']==1
    independent=sum(marginal(i,set(),rel,asp,pair,PARAMS)['gain'] for i in range(3))
    assert gain<independent and gain==pytest.approx(1.25+.3+.3)

def test_full_diagnostics_cost_and_eligibility():
    p,s,score=fixture();out=allocate(p,s,score,Tokenizer(),8,PARAMS)
    assert out['stop_reason']=='BUDGET_EXHAUSTED'
    b=behavior(out,s,score)
    assert b['full_eligible_query'] and b['full_rejected_cost']>0
    assert b['full_blocked_one_sentence']>0 and b['full_chosen']==0
    assert len(out['step_scores'])==len(out['trace'])+1

def test_global_threshold_rule_is_outcome_free():
    from src.post_w12.bea_qbgf.calibration import choose_threshold
    c=dict(calibration=dict(tau_stop_grid=[.001,.01],minimum_threshold_stop_fraction=.2,
                           maximum_near_full_fraction=.8,maximum_empty_fraction=.1,minimum_coverage_retention=.85))
    rows=[dict(dataset='d',query_id='q',budget=256,tau=t,coverage=1,utilization=u,tokens=100,stop_reason=reason)
          for t,u,reason in [(.001,1.,'BUDGET_EXHAUSTED'),(.01,.6,'LOW_MARGINAL_VALUE')]]
    selected,summary=choose_threshold(rows,c)
    assert selected==.01 and summary[0]['eligible'] is False
    rows[1]['coverage']=.5
    assert choose_threshold(rows,c)[0] is None

def test_final_freeze_requires_ready_calibration(tmp_path,monkeypatch):
    from src.post_w12.bea_qbgf import protocol
    monkeypatch.setattr(protocol,'ROOT',tmp_path)
    monkeypatch.setattr(protocol,'preregister',lambda c:({}, {}, {}, dict(identity_hash='p',identity={})))
    c=dict(output_root='results/post_w12/bea_qbgf/test')
    with pytest.raises(ValueError,match='calibration required'):protocol.freeze(c)
    protocol.save(protocol.output(c)/'calibration/CALIBRATION.json',dict(protocol_hash='p',status='NEEDS_REVIEW'))
    with pytest.raises(ValueError,match='needs review'):protocol.freeze(c)
