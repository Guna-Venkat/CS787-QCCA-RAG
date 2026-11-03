import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from demo.evidence_registry import (retrieval_slice,reader_slice,matched_retriever_downstream,sara_case,
 model_registry,trace_comparison,retriever_scope,packing_facts,RETRIEVERS)
from demo.result_registry import get_query,replay_index,curated_ids,method_replay
from demo.data_loader import load,DataError

@pytest.mark.parametrize('ds',['QASPER','HotpotQA','SciFact'])
def test_three_retriever_slices_match(ds):
    for g in ('PASSAGE','SENTENCE','PROPOSITION'):
        rs=retrieval_slice(ds,g,10)
        assert {r['retriever'] for r in rs}==set(RETRIEVERS)
        assert len({r['n_queries'] for r in rs})==1
        assert all(r['passage_representation']=='P0' and r['dataset']==ds for r in rs)

@pytest.mark.parametrize('ds',['QASPER','HotpotQA','SciFact'])
def test_reader_slice_separates_primary_metrics(ds):
    for b in (256,1024):
        rs=reader_slice(ds,b);assert len(rs)==6
        assert all(r['quality']==r['macro_f1' if ds=='SciFact' else 'f1'] for r in rs)
        assert all(r['reader_variant']=='PUBLIC_TEXT_READER' for r in rs)

@pytest.mark.parametrize('ds',['QASPER','HotpotQA','SciFact'])
def test_downstream_retrieval_holds_reader_budget_method(ds):
    rs=matched_retriever_downstream(ds,256)
    assert {r['retriever'] for r in rs}=={'BM25','SFR'}
    assert len({(r['reader'],r['budget'],r['n_queries']) for r in rs})==1


def test_sara_case_is_actual_paired_specialist():
    rs=sara_case();raw=load('specialists')
    for r in rs:
        name='SARA_SFR_K5_N10' if r['retriever']=='SFR' else 'SARA_CANONICAL_BM25_K5_N10'
        vals=[x for x in raw if x['method']==name]
        assert r['quality']==sum(x['f1'] for x in vals)/len(vals) or abs(r['quality']-sum(x['f1'] for x in vals)/len(vals))<1e-12
        assert r['n_queries']==len(vals)
    assert len({r['checkpoint_hash'] for r in rs})==1


def test_exact_model_ids_from_frozen_protocol():
    registry=model_registry();raw=load('w10_protocol')['models']
    assert registry['SFR']==raw['SFR'] and registry['MiniLM']==raw['MiniLM']
    assert registry['BM25']['revision'] is None


def test_missing_minilm_trace_cannot_produce_answer():
    q=get_query('SciFact',curated_ids('SciFact')[0],replay_index())
    assert trace_comparison(q)['MiniLM']==[]
    scoped=retriever_scope(q,'MiniLM')
    assert scoped['profiles']==[] and method_replay(scoped,'QBGF_FULL',256) is None


def test_bm25_and_sfr_outputs_are_not_substituted():
    q=get_query('SciFact','1132',replay_index())
    bm=retriever_scope(q,'BM25');sfr=retriever_scope(q,'SFR')
    assert all(r['method']=='BM25_PLAIN_RAG' for r in bm['profiles'])
    assert all(r['method']!='BM25_PLAIN_RAG' for r in sfr['profiles'])
    assert method_replay(bm,'QBGF_FULL',256) is None


def test_recovered_trace_is_not_a_full_pool_by_assumption():
    q=get_query('SciFact','1132',replay_index());groups=trace_comparison(q)
    assert len(groups['SFR'])<10 and len(groups['BM25'])<10
    for rt in ('SFR','BM25'):
        assert len({r['rank'] for r in groups[rt]})==len(groups[rt])
        assert all(r['retriever']==rt for r in groups[rt])


def test_packing_uses_exact_source_block_cost_and_missing_is_unknown():
    assert packing_facts({},256)==dict(available=False,cost=None,budget=256,fits=None)
    ix=replay_index();q=get_query('SciFact','1132',ix)
    parents=trace_comparison(q)['SFR'];p=next(p for p in parents if p['serialized_block_tokens']>256)
    assert not packing_facts(p,256)['fits']
    assert packing_facts(p,1024)['fits']


def test_invalid_retriever_rejected():
    q=get_query('SciFact','1132',replay_index())
    with pytest.raises(DataError):retriever_scope(q,'invented')


def test_reader_interaction_caveat_remains_true():
    rs=load('reader_interactions')
    assert rs and all(r['ci_low']<=0<=r['ci_high'] for r in rs)


def test_final_freeze_not_reader_or_retriever_search():
    f=load('freeze');assert f['method']=='FIXED_SENTENCE' and not f['retain_lora']
