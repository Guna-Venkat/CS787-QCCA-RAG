"""Post-hoc safety/resume tests. Provided for manual execution; no real models."""
import ast
import json
from copy import deepcopy
from pathlib import Path
import pytest
from src.post_w12.bea_v2_posthoc import protocol as p
from src.post_w12.bea_v2_posthoc import runner
from src.post_w12.bea_v2_posthoc.analysis import represented,jaccard

def test_fixed_registry_and_posthoc_label():
    c=p.load()
    assert c['threshold']==.00008 and c['methods']==p.METHODS
    assert c['budgets']==[256,1024] and c['label']==p.LABEL
    assert not c['automatic_expansion'] and not c['retrain_predictors']

@pytest.mark.parametrize('field,value',[
    ('threshold',.00009),('budgets',[256]),('methods',['BEA_V2']),
    ('automatic_expansion',True),('retrain_predictors',True),('overwrite',True),('resume',False)])
def test_registry_rejects_changes(tmp_path,monkeypatch,field,value):
    c=p.load();c[field]=value;path=tmp_path/'config.json'
    path.write_text(json.dumps(c));monkeypatch.setattr(p,'CONFIG',path)
    with pytest.raises(ValueError):p.load()

def test_original_termination_is_pinned_and_not_overridden():
    c=p.load();name='results/post_w12/bea_v2/v3/CALIBRATION_TERMINATED.json'
    assert p.sha(p.ROOT/name)==c['protected_inputs'][name]
    assert p.read(p.ROOT/name)['status']=='CALIBRATION_FAILED_TERMINATED'

def test_exact_1200_logical_keys_and_no_expansion():
    cohort={'selected':{d:[dict(query_id=str(i)) for i in range(50)] for d in p.DATASETS}}
    keys=p.keys(cohort)
    assert len(keys)==len({tuple(k) for k in keys})==1200
    assert sum(k[2]=='BEA_V2' for k in keys)==300

def test_store_resumes_and_rejects_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(p.original,'ROOT',tmp_path)
    st=p.Store(tmp_path/'results/post_w12/bea_v2/posthoc_tau8e5/rows',{'x':1})
    key=['HotpotQA','q','BEA_V2',256];row=dict(label=p.LABEL,answer='saved')
    st.put(key,row);st.put(key,row)
    assert st.get(key)==row
    assert p.read(st.file(key))['label']==p.LABEL
    with pytest.raises(ValueError):st.put(key,dict(row,answer='changed'))
    with pytest.raises(ValueError):p.Store(st.path,{'x':2}).get(key)

def test_no_generation_without_explicit_unlock(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('Must fail before any inputs are opened')
    monkeypatch.setattr(runner,'context',forbidden)
    with pytest.raises(ValueError,match='RUN_FULL'):runner.generate(explicit=False)

def test_source_model_and_group_guard():
    parent={'models':{'Qwen':{'id':'same','revision':'same'}}}
    row=dict(dataset='HotpotQA',query_id='q',method='BEA_V2',budget=256,
        cluster_id='g',source_hash='source',evidence_tokens=100,reader='BASE_FP16',
        model_hash=p.digest([parent['models']['Qwen'],'BASE_FP16',None]))
    args=('HotpotQA','q','BEA_V2',256,{'group':'g'},parent,{'source_hash':'source'})
    runner.check_row(row,*args)
    for key,value in [('source_hash','different'),('model_hash','wrong'),('cluster_id','leak'),('evidence_tokens',257)]:
        with pytest.raises(ValueError):runner.check_row(dict(row,**{key:value}),*args)

def test_full_expands_underlying_sentence_overlap():
    problem=dict(pool=[dict(unit_id='p',passage_id='p',unit_type='PASSAGE')],
        sentences=[dict(unit_id='s1',passage_id='p',unit_type='SENTENCE'),
                   dict(unit_id='s2',passage_id='p',unit_type='SENTENCE')])
    parents,sent=represented({'selected_unit_ids':['p']},problem)
    assert parents=={'p'} and sent=={'s1','s2'}
    assert jaccard(sent,{'s1'})==.5
    with pytest.raises(ValueError):represented({'selected_unit_ids':['unknown']},problem)

def test_no_calibration_training_or_unseen_entrypoints():
    for filename in ['protocol.py','runner.py','analysis.py']:
        tree=ast.parse((Path(p.__file__).parent/filename).read_text())
        calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        assert not calls & {'calibrate','train','audit','plan_unseen','freeze'}
    cli=(p.ROOT/'W12-POST/run_bea_v2_posthoc.py').read_text()
    assert "choices=['status','prepare','generate','analyze','all']" in cli

def test_notebook_defaults_do_not_generate_and_can_resume():
    nb=p.read(p.ROOT/'W12-POST/BEA_QBGF_V2_POSTHOC_TAU8E5.ipynb')
    code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    assert 'RUN_FULL = False' in code and 'RUN_ANALYSIS = RUN_FULL' in code
    assert 'RESUME = True' in code and 'OVERWRITE = False' in code
    assert 'if RUN_FULL:' in code and 'explicit=True' in code
    assert all(c.get('execution_count') is None and not c.get('outputs') for c in nb['cells'])

def test_complete_record_required_before_analysis(tmp_path,monkeypatch):
    from src.post_w12.bea_v2_posthoc import analysis
    c={'output_root':'results/post_w12/bea_v2/posthoc_tau8e5'}
    monkeypatch.setattr(p,'ROOT',tmp_path)
    monkeypatch.setattr(analysis,'context',lambda c=None:(c or {},{},{},{},{},{},{'identity_hash':'x'}))
    # No COMPLETE.json: analysis must stop before loading predictions or computing metrics.
    with pytest.raises(FileNotFoundError):analysis.analyze(c)
