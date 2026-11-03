"""Controlled comparisons and scientific identity guards, without real weights."""
from pathlib import Path
import json,hashlib
import numpy as np
import pytest
from demo.live.engine import run,input_key
from demo.live.protocol import CONTROLLED,NATIVE,assert_controlled_pair,archive_obsolete,validity
from demo.live.segmentation import ADAPTED,FROZEN,documents,annotate
from demo.tests.test_live import FakeResources,CharTokenizer,PASSAGES

@pytest.mark.parametrize('method',['RAW_SKIP','FIXED_SENTENCE','QBGF_FULL'])
def test_controlled_pair_same_downstream_protocol(method):
 r=FakeResources();cache={}
 a=run('beta',PASSAGES,method,256,'BM25',r,True,prepared_cache=cache,scoring_mode=CONTROLLED)
 b=run('beta',PASSAGES,method,256,'SFR',r,True,prepared_cache=cache,scoring_mode=CONTROLLED)
 assert assert_controlled_pair(a,b)
 assert a['scorer_identity']==b['scorer_identity']
 assert a['invariants']==b['invariants']
 for result in (a,b):
  for u,score in zip(result['pool'],result['allocation_passage_scores']):
   q=r.embed(['beta'],'query')[0];v=r.embed([u['source_text']],'document')[0]
   assert np.isclose(score,np.dot(q,v)/(np.linalg.norm(q)*np.linalg.norm(v)))
  assert all(-1.000001<=x<=1.000001 for x in result['allocation_passage_scores'])
  assert result['result_type']=='LIVE_CUSTOM'


def test_native_bm25_qbgf_is_blocked():
 with pytest.raises(ValueError,match='BM25-native QBGF is disabled'):
  run('beta',PASSAGES,'QBGF_FULL',256,'BM25',FakeResources(),True)


def test_controlled_scoring_does_not_use_bm25_allocation(monkeypatch):
 import src.week9.retrieval as retrieval
 original=retrieval.bm25;calls=[]
 def bm25(q,texts):calls.append(list(texts));return original(q,texts)
 monkeypatch.setattr(retrieval,'bm25',bm25)
 result=run('beta',PASSAGES,'QBGF_FULL',256,'BM25',FakeResources(),True,scoring_mode=CONTROLLED)
 assert len(calls)==1 # Candidate retrieval only; unit allocation uses common embeddings.
 assert result['qbgf_options']
 assert all(o['label'].startswith(('DROP','FULL','1 selected','2 selected','4 selected','8 selected')) for o in result['qbgf_options'])


def test_keys_include_scorer_and_segmentation():
 args=('q',PASSAGES,256,'BM25')
 assert input_key(*args,ADAPTED,NATIVE)!=input_key(*args,ADAPTED,CONTROLLED)
 assert input_key(*args,ADAPTED,CONTROLLED)!=input_key(*args,FROZEN,CONTROLLED)


def test_changed_controlled_invariant_rejected():
 r=FakeResources();a=run('beta',PASSAGES,'RAW_SKIP',256,'BM25',r,True,scoring_mode=CONTROLLED)
 b=run('beta',PASSAGES,'RAW_SKIP',1024,'SFR',r,True,scoring_mode=CONTROLLED)
 with pytest.raises(ValueError,match='invariants differ'):assert_controlled_pair(a,b)


def test_old_runs_retained_but_invalid_for_reuse():
 records={('old',):dict(method='FIXED_SENTENCE',segmentation=FROZEN),('raw',):dict(method='RAW_SKIP')};archive={}
 archive_obsolete(records,archive)
 assert len(records)==len(archive)==2
 assert records[('old',)]['invalid_for_granularity']
 assert records[('raw',)]['invalid_for_granularity'] is False
 assert validity(records[('old',)])


@pytest.mark.parametrize('passage,n',[(6,3),(9,3),(7,4)])
def test_exact_world_cup_critical_passages(passage,n):
 from demo.live.engine import parse_passages
 from src.week11.hierarchy import build
 from demo.live.preset import load_preset
 ps=load_preset()['passages']
 units=annotate(build(documents(ps),CharTokenizer(),{'max_passage_tokens':256}),ADAPTED)
 ss=[u for u in units if u['unit_type']=='SENTENCE' and u['document_id']==f'P{passage}']
 assert len(ss)==n
 assert all(u['display_kind']=='SENTENCE' for u in ss)
 if passage==6:assert [s['source_text'].split()[0] for s in ss]==['Starting','As','It']
 if passage==7:assert '39 days' in ss[1]['source_text'] and '56 days' in ss[3]['source_text']


def test_world_cup_original_p6_real_token_count():
 from demo.live.engine import parse_passages
 from demo.live.resources import LiveResources
 from demo.live.preset import load_preset
 ps=load_preset()['passages']
 assert len(LiveResources().tokenizer().encode(ps[5],add_special_tokens=False))==117


def test_protected_frozen_hashes_unchanged():
 root=Path(__file__).resolve().parents[2]
 baseline=json.loads((root/'demo/audit/correctness_protected_before.json').read_text())
 for rel in ['src/week9/senttopk_e3/sentences.py','src/week12/algorithm.py','results/week12/W13_FINAL_METHOD_FREEZE.json','demo/replay_data/v1/bundle.json']:
  assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==baseline[rel]

def test_controlled_ui_end_to_end_and_frozen_mode_visibility(tmp_path):
 from streamlit.testing.v1 import AppTest
 script=tmp_path/'controlled_host.py'
 script.write_text('''import streamlit as st
from demo.live import ui
from demo.tests.test_live import FakeResources,PASSAGES
ui.resources=lambda:FakeResources()
ui.available=lambda:dict(sfr_local=True,qwen_local=True,gpu_free_mib=None)
if 'live_passage_ids' not in st.session_state:
    st.session_state['live_passage_ids']=list(range(5));st.session_state['live_next_id']=5
    for i,p in enumerate(PASSAGES):st.session_state[f'live_passage_{i}']=p
    st.session_state['live_query']='Which beta sensor?'
    st.session_state['live_method']='QBGF_FULL'
    st.session_state['live_models_enabled']=True
    st.session_state['live_scoring_mode']='controlled_sfr'
ui.render()
''')
 app=AppTest.from_file(str(script)).run(timeout=20)
 seg=next(s for s in app.selectbox if s.label=='Source segmentation')
 assert len(seg.options)==1 and 'Citation/Markdown-aware Live segmentation' in seg.options
 next(b for b in app.button if b.label=='Run Evidence Pipeline').click().run(timeout=20)
 assert not app.exception
 next(b for b in app.button if b.label=='Compare Retrievers End-to-End').click().run(timeout=20)
 assert not app.exception,[e.message for e in app.exception]
 assert any('Only the candidate retriever differs' in i.value for i in app.info)
 assert any(e.label=='Pipeline invariants' for e in app.expander)
 assert len(app.session_state['live_e2e'])==2

def test_demo_only_method_labels():
 from demo.curated.ui import method_label
 assert 'DEMO-ONLY REPRESENTATION' in method_label('PROPOSITION')
 assert 'clause-level' not in method_label('PROPOSITION')
 assert 'DEMO-ONLY REPRESENTATION' in method_label('MULTI_SENTENCE_2')
 assert 'DEMO-ONLY REPRESENTATION' not in method_label('FIXED_SENTENCE')
