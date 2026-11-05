"""Synthetic-only diagnostic tests, supplied for manual execution."""
from copy import deepcopy
import ast
from pathlib import Path
import pytest
from src.post_w12.final_case_study.gold import merge,retention,retrieval_status
from src.post_w12.final_case_study.selection import choose,METHODS,PREFIXES
from src.post_w12.final_case_study import analysis

class Tokenizer:
    is_fast=True
    def __call__(self,text,**kwargs):return dict(offset_mapping=[(i,i+1) for i in range(len(text))])

def example():
    return dict(inference=dict(documents=[dict(document_id='d',text='Alpha. Beta.')]),
        evaluation=dict(gold_spans=[dict(document_id='d',evidence_id='g',text='Alpha.',start_char=0,end_char=6)]))

def test_exact_offset_gold_retention_and_document_recall():
    e=example();u=[dict(document_id='d',start_char=0,end_char=6)]
    r=retention(e,u,Tokenizer())
    assert r['gold_evidence_recall']==1 and r['gold_evidence_token_coverage']==1
    assert retrieval_status(r)=='RETRIEVAL_SUFFICIENT'
    miss=retention(e,[dict(document_id='d',start_char=7,end_char=12)],Tokenizer())
    assert miss['gold_parent_recall']==1 and miss['gold_evidence_recall']==0
    assert retrieval_status(miss)=='RETRIEVAL_MISS'

def test_partial_gold_and_unmapped_annotations_are_not_success():
    e=example();e['evaluation']['gold_spans'].append(dict(document_id='missing',evidence_id='x',text='?',start_char=None,end_char=None))
    r=retention(e,[dict(document_id='d',start_char=0,end_char=3)],Tokenizer())
    assert r['gold_evidence_token_coverage']==.5 and r['unmapped_annotations']==1
    assert retrieval_status(r)=='RETRIEVAL_PARTIAL'

def test_absent_nei_support_is_unavailable_not_zero():
    e=example();e['evaluation']['gold_spans']=[]
    r=retention(e,[],Tokenizer())
    assert r['gold_evidence_recall'] is None and r['gold_parent_recall'] is None
    assert retrieval_status(r)=='ANNOTATIONS_UNAVAILABLE'

def test_qasper_alternative_occurrence_and_deduplication():
    e=example();e['inference']['documents'][0]['text']='Alpha. Alpha.'
    g=e['evaluation']['gold_spans'][0];g['alternatives']=[[0,6],[7,13]]
    e['evaluation']['gold_spans'].append(deepcopy(g))
    r=retention(e,[dict(document_id='d',start_char=7,end_char=13)],Tokenizer())
    assert r['annotation_count']==1 and r['gold_evidence_recall']==1

def test_interval_union_does_not_double_count():
    assert merge([(0,4),(2,6),(8,10)])==[[0,6],[8,10]]

def test_inter_sentence_whitespace_gaps_are_not_evidence_loss():
    e=example();e['evaluation']['gold_spans'][0].update(text='Alpha. Beta.',end_char=12)
    r=retention(e,[dict(document_id='d',start_char=0,end_char=6),dict(document_id='d',start_char=7,end_char=12)],Tokenizer())
    assert r['gold_evidence_recall']==1 and r['gold_evidence_token_coverage']==1

def record(q,b,scores):
    return dict(dataset='SciFact',query_id=q,budget=b,
        **{p+'_score':v for p,v in zip(PREFIXES,scores)},score_spread=max(scores)-min(scores),
        max_score=max(scores),meaningfully_different_allocation=True,allocation_difference=.8)

def test_deterministic_selection_and_scifact_correctness_patterns():
    rs=[record('a',256,[1,0,0,0]),record('a',1024,[0,0,1,0]),
        record('b',256,[0,0,0,1]),record('b',1024,[1,1,1,1]),
        record('c',256,[1,1,1,1]),record('c',1024,[1,1,1,1])]
    selected,rev=choose(rs,['SciFact'])
    assert choose(list(reversed(rs)),['SciFact'])==(selected,rev)
    assert selected[0]['query_id']=='a' and selected[3]['query_id']=='a'
    assert any(r['case_type']=='BUDGET_REVERSAL' for r in selected)
    assert rev[0]['strict_inversions']>0

def test_missing_rescue_tie_or_reversal_never_fabricated():
    rs=[record('a',b,[0,0,0,0]) for b in [256,1024]]
    selected,rev=choose(rs,['SciFact'])
    assert all(not r['qualifies'] for r in selected)
    assert not rev[0]['qualifies'] and rev[0]['query_id'] is None

def test_default_analysis_does_not_open_inputs(monkeypatch):
    def forbidden():raise AssertionError('Inputs must not be read before explicit flag')
    monkeypatch.setattr(analysis,'load_inputs',forbidden)
    with pytest.raises(ValueError,match='RUN_ANALYSIS'):analysis.analyze()

def test_analysis_source_does_not_call_allocation_or_models():
    for p in Path(analysis.__file__).parent.glob('*.py'):
        tree=ast.parse(p.read_text());calls=set()
        for n in ast.walk(tree):
            if isinstance(n,ast.Call):
                if isinstance(n.func,ast.Name):calls.add(n.func.id)
                elif isinstance(n.func,ast.Attribute):calls.add(n.func.attr)
        assert not calls & {'Reader','generate','allocate','predict','train','fit','calibrate','encode'}

def test_writes_cannot_touch_source_results(tmp_path,monkeypatch):
    monkeypatch.setattr(analysis,'OUT',tmp_path/'case_study')
    with pytest.raises(ValueError):analysis.save(tmp_path/'original_results.json',{})
    analysis.save(analysis.OUT/'identity.json',{'x':1},immutable=True)
    with pytest.raises(ValueError):analysis.save(analysis.OUT/'identity.json',{'x':2},immutable=True)
