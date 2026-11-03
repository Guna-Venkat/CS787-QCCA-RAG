import json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from demo import data_loader as dl
from demo import result_registry as rr
from demo.constants import METHODS,BUDGETS,DATASETS,FREEZE_SHA

def test_registry_exact_core():
    assert METHODS==('RAW_SKIP','FIXED_SENTENCE','H3','QBGF_FULL','NONE')
    assert BUDGETS==(256,1024)

def test_primary_load_and_count():
    assert len(dl.summaries('validation'))==51
    assert len(dl.summaries('confirmation'))==20

def test_finding_values_come_from_primary():
    f=rr.findings()
    for row in f['confirmation']:
        assert row==rr.lookup(dl.load('confirmation'),row['dataset'],row['method'],row['budget'])

def test_metric_dataset_mismatch():
    with pytest.raises(dl.DataError):dl.validate_rows([dict(dataset='SciFact',method='NONE',budget=256,quality=.2,metric='f1')])

def test_missing_quality():
    with pytest.raises(dl.DataError):dl.validate_rows([dict(dataset='QASPER',method='NONE',budget=256,quality=None,metric='f1')])

def test_duplicate_condition():
    r=dict(dataset='QASPER',method='NONE',budget=256,quality=.2,metric='f1')
    with pytest.raises(dl.DataError):dl.validate_rows([r,r])

def test_unknown_budget():
    with pytest.raises(dl.DataError):dl.validate_rows([dict(dataset='QASPER',method='NONE',budget=512,quality=.2,metric='f1')])

def test_missing_result_not_zero():
    with pytest.raises(dl.DataError):rr.lookup([], 'QASPER','NONE',256)

def test_invalid_query():
    with pytest.raises(dl.DataError):rr.get_query('SciFact','not-a-real-query',{})

def test_curation_is_deterministic_and_val_only():
    for ds in DATASETS:
        assert rr.curated_ids(ds)==rr.curated_ids(ds)
        assert set(rr.curated_ids(ds)) <= {r['query_id'] for r in rr.profiles() if r['dataset']==ds}

def test_cache_fallback(monkeypatch,tmp_path):
    monkeypatch.setattr(rr,'ROOT',tmp_path)
    assert rr.replay_index()=={}

def test_missing_row_no_generation():
    q=rr.get_query('SciFact',rr.curated_ids('SciFact')[0],{})
    assert rr.method_replay(q,'RAW_SKIP',256) is None

def test_saved_text_fallback_without_index():
    q=rr.get_query('SciFact',rr.curated_ids('SciFact')[0],{})
    r=rr.method_replay(q,'QBGF_FULL',256)
    assert r['answer'] is not None and any(u['text'] for u in r['units'])

def test_unit_category_signs():
    for u in dl.load('units'):
        assert u['category']==('HELPFUL' if u['importance']>0 else 'HARMFUL' if u['importance']<0 else 'NEUTRAL')

def test_exact_freeze():assert dl.freeze_check()==FREEZE_SHA

def test_provenance_is_real_file():
    p=dl.provenance('validation')
    assert p['sha256']==dl.sha(ROOT/p['path'])
    assert not p['path'].startswith('/')

def test_markdown_not_numerical_source():
    with pytest.raises(dl.DataError):dl.load('results/FINAL_PROJECT_REPORT.md')

def test_escape_path_blocked():
    with pytest.raises(dl.DataError):dl.path_for('../outside.csv')

def test_live_missing_models(monkeypatch,tmp_path):
    import demo.live_pipeline as lp
    monkeypatch.setattr(lp.Path,'home',lambda:tmp_path)
    monkeypatch.setattr(lp,'ROOT',tmp_path)
    def missing(*a,**k):raise FileNotFoundError
    monkeypatch.setattr(lp.subprocess,'run',missing)
    r=lp.readiness()
    assert not r['enabled'] and not any(r['models'].values()) and r['gpu_free_mib'] is None

def test_imports_do_not_load_model_modules():
    import subprocess
    run=subprocess.run([sys.executable,'-c',"from demo.result_registry import findings; import sys; findings(); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],cwd=ROOT,capture_output=True,text=True)
    assert run.returncode==0,run.stderr

def test_recovered_index_exact_unit_tokens():
    index=rr.replay_index()
    if not index:pytest.skip('Optional CPU recovery index not built')
    assert len(index)==36
    for ds in DATASETS:
        q=rr.get_query(ds,rr.curated_ids(ds)[0],index)
        for p in q['profiles']:
            r=rr.method_replay(q,p['method'],p['budget'])
            assert all(u['source'] and u['text']==u['source']['source_text'] and u['tokens']==u['source']['token_count'] for u in r['units'])

def test_recovered_index_detects_stale_export(monkeypatch,tmp_path):
    p=tmp_path/'demo/cache';p.mkdir(parents=True)
    (p/'replay_index.json').write_text(json.dumps(dict(sources={'units':{'sha256':'wrong'}},queries={})))
    monkeypatch.setattr(rr,'ROOT',tmp_path)
    with pytest.raises(dl.DataError):rr.replay_index()

def test_provenance_manifest_matches():assert dl.verify_manifest()

def test_all_curated_methods_recover_consistently():
    ix=rr.replay_index()
    if not ix:pytest.skip('CPU index not built')
    for key in ix:
        ds,qid=key.split(':',1);q=rr.get_query(ds,qid,ix)
        for p in q['profiles']:
            r=rr.method_replay(q,p['method'],p['budget'])
            assert len(r['units'])==p['units']
            assert all(u['text'] is not None for u in r['units'])
