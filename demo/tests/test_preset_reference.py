"""Demo-only reference diagnostics; no real inference or research result writes."""
from copy import deepcopy
import pytest
from demo.live.preset import load_preset
from demo.live.reference import evaluate,metrics_text,checklist
from demo.live.history import remember,rows
from demo.live.engine import run
from demo.tests.test_live import FakeResources


def reference_result():
    p=load_preset()
    return dict(query=p['query'],passages=p['passages'],answer=p['reference']['answer'])


def test_exact_reference_and_all_required_facts():
    r=reference_result();m=evaluate(r)
    assert m['precision']==m['recall']==m['f1']==m['em']==1
    assert m['matched_facts']==m['total_facts']==9
    assert all(x['Matched'] for x in m['checklist'])
    assert len(load_preset()['reference']['required_facts'])==9
    assert load_preset()['reference']['label']=='Preset Reference Answer'


@pytest.mark.parametrize('field',['query','passages'])
def test_edited_preset_cannot_score(field):
    r=deepcopy(reference_result())
    if field=='query':r[field]+=' What else?'
    else:r[field][0]+=' Added custom information.'
    assert evaluate(r) is None
    assert metrics_text(r)=='No gold-reference evaluation available'


def test_empty_missing_and_partial_answers():
    r=reference_result();r['answer']=''
    m=evaluate(r)
    assert m['f1']==m['em']==m['precision']==m['recall']==0
    assert m['matched_facts']==0
    r['answer']=None
    assert evaluate(r) is None and 'no generated answer' in metrics_text(r)
    r['answer']='48 teams'
    m=evaluate(r)
    assert m['precision']==1 and 0<m['f1']<1 and m['em']==0
    assert m['matched_facts']==1


def test_checklist_requires_linked_cues_and_risk_qualification():
    assert not any(f['Matched'] for f in checklist('48 spectators. 12 groups of 3. 64 teams. 104 goals.'))
    facts=checklist('Penalty shootouts would completely remove collusion risk.')
    assert not facts[-1]['Matched']
    assert checklist('Penalty shootouts would not fully solve collusion risk.')[-1]['Matched']


def test_history_reference_is_not_a_run_and_no_prompt_leakage():
    p=load_preset();resource=FakeResources()
    r=run(p['query'],p['passages'],'NONE',256,'SFR',resource,True,scoring_mode='controlled_sfr')
    assert resource.calls==[(p['query'],'')]
    assert 'reference' not in r
    r['answer']=p['reference']['answer'];history=[];remember(history,r)
    table=rows(history)
    assert [x['Run'] for x in table]==[0,1]
    assert len(history)==1 and history[0]['number']==1
    assert table[0]['Method']=='Preset Reference Answer'
    assert table[0]['Budget'] is None and 'not a model run' in table[0]['Metrics']
    assert 'F1 1.000' in table[1]['Metrics'] and '9/9' in table[1]['Metrics']


def test_preset_reference_ui(tmp_path):
    from streamlit.testing.v1 import AppTest
    script=tmp_path/'reference_host.py'
    script.write_text('''from demo.live.ui import render_history,evidence_card
from demo.live.preset import load_preset
from demo.live.history import remember
from demo.live.engine import run
from demo.tests.test_live import FakeResources
p=load_preset()
r=run(p['query'],p['passages'],'NONE',256,'SFR',FakeResources(),True,scoring_mode='controlled_sfr')
r['answer']=p['reference']['answer']
history=[];remember(history,r)
render_history(history)
evidence_card(r)
''')
    app=AppTest.from_file(str(script)).run(timeout=20)
    assert not app.exception,[e.message for e in app.exception]
    table=app.dataframe[0].value
    assert table['Run'].tolist()==[0,1]
    assert table['Metrics'][1].endswith('(100%)')
    assert any('Preset Reference Answer' in e.label for e in app.expander)
