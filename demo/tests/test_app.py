"""No model loading: execute actual Streamlit screens and controls."""
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

def start(historical=True):
    at=AppTest.from_file(str(ROOT/'demo/app.py'),default_timeout=40).run()
    assert not at.exception,[e.message for e in at.exception]
    if historical and any(r.label=='Replay collection' for r in at.radio)==False:
        next(t for t in at.toggle if t.label=='Researcher Mode').set_value(True).run()
    if historical and any(r.label=='Replay collection' for r in at.radio):
        next(r for r in at.radio if r.label=='Replay collection').set_value('Historical exports').run()
    return at

def group(at,label):return next(w for w in at.get('button_group') if w.label==label)
def select(at,label):return next(w for w in at.selectbox if w.label==label)

def test_default_model_free_lab():
    at=start()
    assert select(at,'Dataset').value=='SciFact'
    assert any(m.label=='Correctness' for m in at.metric)
    assert not at.error

def test_all_dataset_lab_screens():
    at=start()
    for ds in ['HotpotQA','QASPER','SciFact']:
        select(at,'Dataset').set_value(ds).run()
        assert not at.exception,[e.message for e in at.exception]
        assert not at.error

def test_budget_and_missing_method():
    at=start();group(at,'Evidence-token budget').set_value(1024).run()
    assert not at.exception
    select(at,'Left method').set_value('RAW_SKIP').run()
    assert not at.exception
    assert any('No compatible saved result' in i.value for i in at.info)

def test_findings_all_datasets_and_technical():
    at=start();group(at,'Workspace').set_value('Research Findings').run()
    assert not at.exception,[e.message for e in at.exception]
    for ds in ['QASPER','HotpotQA','SciFact']:
        group(at,'Result dataset').set_value(ds).run()
        assert not at.exception,[e.message for e in at.exception]
    next(r for r in at.radio if r.label=='Detail').set_value('Technical View').run()
    assert not at.exception

def test_journey():
    at=start();group(at,'Workspace').set_value('Research Journey').run()
    assert not at.exception
    assert any('What we established' in x.value for x in at.markdown)

def test_invalid_query_ui():
    at=start();at.text_input[0].set_value('invalid').run()
    assert not at.exception
    assert any('not present' in w.value for w in at.warning)

def test_representation_options_are_grounded():
    at=start()
    group(at,'Explore representation').set_value('Adaptive / QBGF').run()
    assert not at.exception
    assert any('Original per-passage action' in x.value for x in at.caption)

def test_researcher_and_presentation():
    at=start()
    next(t for t in at.toggle if t.label=='Researcher Mode').set_value(True).run()
    next(t for t in at.toggle if t.label=='Presentation Mode').set_value(False).run()
    assert not at.exception

def test_tour_changes_budget():
    at=start();next(t for t in at.toggle if t.label=='Start 90-second tour').set_value(True).run()
    at.select_slider[0].set_value('5 · Larger budget').run()
    assert group(at,'Evidence-token budget').value==1024
    assert not at.exception

def test_genuinely_stored_proposition_view():
    at=start();select(at,'Dataset').set_value('HotpotQA').run()
    select(at,'Curated validation example').set_value('5a84bc5b5542991dd0999dc3').run()
    source=select(at,'Same source, different granularity')
    i=next(i for i,x in enumerate(source.options) if 'SFR rank 4 ' in x)
    source.select_index(i).run()
    assert 'Fine-grained source spans' in group(at,'Explore representation').options
    group(at,'Explore representation').set_value('Fine-grained source spans').run()
    assert not at.exception

def test_missing_retriever_is_retrieval_only():
    at=start();group(at,'Retriever').set_value('MiniLM').run()
    assert not at.exception
    assert any('Retrieval-only exploration' in x.value for x in at.info)
    assert not any(m.label=='Correctness' for m in at.metric)


def test_retriever_reader_and_protocol_findings_render():
    at=start();group(at,'Workspace').set_value('Research Findings').run()
    assert not at.exception,[e.message for e in at.exception]
    assert any('Retriever quality changes the evidence ceiling' in m.value for m in at.markdown)
    assert any('Same evidence protocol, different reader' in m.value for m in at.markdown)
    assert any('Does more adaptive mean better?' in m.value for m in at.markdown)

def test_return_from_unavailable_retriever_restores_method_state():
    at=start()
    group(at,'Retriever').set_value('MiniLM').run()
    group(at,'Retriever').set_value('SFR').run()
    assert not at.exception,[e.message for e in at.exception]
    assert select(at,'Left method').value is not None
    assert any(m.label=='Correctness' for m in at.metric)


def test_clean_curated_default_and_live_navigation():
    at=start(historical=False)
    assert any(s.value=='Curated Research Examples' for s in at.subheader)
    assert not any('Complete curated collection' in x.value or 'not original benchmark' in x.value for x in at.markdown)
    assert 'LiveDemo' in group(at,'Workspace').options
    group(at,'Workspace').set_value('LiveDemo').run()
    assert not at.exception
    assert any(b.label=='Run Evidence Pipeline' for b in at.button)
