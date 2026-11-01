"""Additive extension tests: frozen identity, selection parity, resume and safety."""
import copy,json
from pathlib import Path
import pytest
from src.week12.protocol import ROOT,load,code_hash,Store,digest
from src.week12_extension import baselines as b

def test_existing_experiment_hash_excludes_extension_and_notebooks():
    source=(ROOT/'src/week12/protocol.py').read_text()
    assert "(ROOT/'src/week12').glob('*.py')" in source
    assert 'src/week12_extension' not in source

def test_extension_protocol():
    cfg=load();i=b.extension_identity(cfg,True)
    assert i['budgets']==[256,1024]
    assert i['reader']=='BASE_FP16'
    assert set(b.METHODS.values())=={'BM25','SFR'}

def test_synthetic_selection_and_resume(tmp_path):
    cfg=copy.deepcopy(load());cfg['output_root']=str(tmp_path/'out');cfg['cache_root']=str(tmp_path/'cache')
    rows=b.prepare(cfg,True)
    assert len(rows)==12
    assert {r['selection']['method'] for r in rows}==set(b.METHODS)
    assert all(r['selection']['evidence_tokens']<=r['selection']['budget'] for r in rows)
    assert rows==b.prepare(cfg,True)
    generated=b.generate(cfg,True)
    assert len(generated)==12
    assert generated==b.generate(cfg,True)
    assert all(r['reader']=='BASE_FP16' for r in generated)
    assert len({(r['dataset'],r['query_id']) for r in generated})==3

def test_specialist_qualification_required(tmp_path):
    from src.week12_extension.specialists import require_qualified,config
    cfg=copy.deepcopy(load());cfg['output_root']=str((ROOT/'results/week12/test_missing_extension').relative_to(ROOT))
    with pytest.raises(ValueError,match='Qualify'):require_qualified(cfg,'SARA_SFR_K5_N10')
    with pytest.raises(ValueError):config(cfg,'HISTORICAL_SFR_K6')

def test_safe_notebook_defaults_and_freeze_gate():
    import subprocess
    n=json.loads(subprocess.check_output(['git','show',':notebooks/week12/week12_C_EXT_baseline_extension.ipynb'],cwd=ROOT,text=True));text='\n'.join(''.join(c['source']) for c in n['cells'])
    assert 'RUN_C_EXT_FULL=False' in text and 'RUN_SPECIALIST_QUALIFICATION=False' in text and 'RUN_SFR_SARA=False' in text
    for name in ['H_validation_analysis','I_week13_freeze']:
        n=json.loads((ROOT/f'notebooks/week12/week12_{name}.ipynb').read_text());s=str(n)
        assert 'analyze_with_extension' in s and 'c_extension' in s
    assert 'require_complete(cfg)' in s

def test_plain_sfr_matches_c_raw_skip(tmp_path):
    from src.week12.runner import fixture_examples
    from src.week12.laboratory import Lab
    from src.week12.algorithm import choose
    from src.week11.budgets import FixtureTokenizer
    cfg=copy.deepcopy(load());cfg['output_root']=str(tmp_path/'out');cfg['cache_root']=str(tmp_path/'cache')
    rows=b.prepare(cfg,True);tok=FixtureTokenizer();lab=Lab(cfg,tok,True)
    try:
        for e in fixture_examples():
            pool,sent,props,ps,ss,_=lab.prepare(e,'SFR')
            for budget in [256,1024]:
                ref=choose(pool,sent,props,ps,ss,tok,budget,'RAW_SKIP',{'Qwen':tok})
                got=next(r['selection'] for r in rows if r['dataset']==e['dataset'] and r['selection']['method']=='SFR_PLAIN_RAG' and r['selection']['budget']==budget)
                assert got['evidence_ids']==ref['evidence_ids']
                assert got['selected_unit_ids']==ref['selected_unit_ids']
    finally:lab.close()
