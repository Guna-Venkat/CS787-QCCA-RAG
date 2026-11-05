"""Read-only checks of the completed final allocation-only resolution."""
import ast
import json
from pathlib import Path
import pandas as pd
from src.post_w12.bea_v2.protocol import ROOT, read, sha, load
from src.post_w12.bea_v2.stages import calibration_summary

OUT = ROOT/'results/post_w12/bea_v2/v3'
RES = OUT/'calibration/resolution'

def test_amendment_and_original_inputs_preserved():
    a=read(RES/'AMENDMENT.json'); r=read(RES/'RESOLUTION_RESULT.json')
    assert r['amendment_sha']==sha(RES/'AMENDMENT.json')
    assert a['thresholds']==[.00006,.00007,.00008,.00009]
    assert a['cohort']==read(OUT/'calibration/COHORT.json')
    assert a['original_rules']==load()['calibration']
    for p,h in a['input_hashes'].items():
        assert sha(ROOT/p)==h

def test_exact_nine_queries_four_thresholds_two_budgets():
    a=read(RES/'AMENDMENT.json')
    df=pd.read_csv(RES/'allocations.csv',dtype={'query_id':str})
    expected={(d,m['query_id'],b,t) for d,ms in a['cohort'].items()
              for m in ms for b in [256,1024] for t in a['thresholds']}
    assert len(df)==len(expected)==72
    assert set(df[['dataset','query_id','budget','threshold']].itertuples(index=False,name=None))==expected

def test_unchanged_rules_reject_every_resolution_threshold():
    old=pd.read_csv(OUT/'calibration/allocations.csv',dtype={'query_id':str})
    new=pd.read_csv(RES/'allocations.csv',dtype={'query_id':str})
    summary,cells=calibration_summary(pd.concat([old[old.threshold==0],new]),load())
    candidates=[r for r in summary if r['threshold']!=0]
    assert len(candidates)==4 and not any(r['eligible'] for r in candidates)
    r=read(RES/'RESOLUTION_RESULT.json')
    assert r['status']=='CALIBRATION_FAILED_TERMINATED'
    assert r['threshold'] is None and not r['reader_pilot_permitted']
    assert not r['further_calibration_permitted']
    assert read(OUT/'CALIBRATION_TERMINATED.json')==r

def test_resolution_has_no_reader_training_or_quality_calls():
    tree=ast.parse((ROOT/'W12-POST/run_bea_v2_resolution.py').read_text())
    names={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    assert not names & {'Reader','generate','evaluate','analyze','train','audit','load_val','freeze'}
    r=read(RES/'RESOLUTION_RESULT.json')
    assert r['answer_generations']==0
    assert not r['answer_outcomes_accessed'] and not r['heldout_LOO_labels_accessed']
    assert not (OUT/'BEA_V2_FREEZE.json').exists()
    assert not any((OUT/'pilot/rows').glob('*.json'))

def test_notebook_stops_before_audit_or_reader_when_terminated():
    nb=json.loads((ROOT/'W12-POST/BEA_QBGF_V2.ipynb').read_text())
    code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    assert 'RUN_PILOT = False' in code[0]
    assert 'CALIBRATION_TERMINATED.json' in code[0] and 'raise RuntimeError' in code[0]
    assert all('audit(cfg)' not in s and 'generate(cfg)' not in s for s in code[:1])
