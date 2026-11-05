"""History tests use synthetic inputs and fake readers only."""
from copy import deepcopy
from demo.live.engine import run
from demo.live.history import remember,rows,differences
from demo.tests.test_live import FakeResources,PASSAGES


def result(method='RAW_SKIP',budget=256):
    return run('Which beta sensor?',PASSAGES,method,budget,'SFR',FakeResources(),True,
               scoring_mode='controlled_sfr')


def test_history_deduplicates_and_retains_results():
    history=[];a=result();remember(history,a);remember(history,a)
    assert len(history)==1 and rows(history)[0]['Run']==1
    cache={'a':a};cache.clear()
    assert history[0]['result']['answer']=='A concise answer.'
    assert rows(history)[0]['Answer']=='A concise answer.'
    updated=deepcopy(a);updated['answer']='Updated';remember(history,updated)
    assert len(history)==1 and history[0]['result']['answer']=='Updated'


def test_invalid_history_and_context_only():
    a=result();a['protocol_version']='old';history=[];remember(history,a)
    assert history==[]
    a=result();a['answer']=None;remember(history,a)
    assert rows(history)[0]['Reader']=='Context only'
    assert rows(history)[0]['Answer']=='No answer — reader unavailable or disabled'


def test_history_bound_and_changed_settings():
    a=result();history=[]
    for i in range(35):
        b=deepcopy(a);b['input_key']=str(i);remember(history,b)
    assert len(history)==30 and history[0]['number']==6 and history[-1]['number']==35
    b=deepcopy(a);b['budget']=1024;b['query']='Other question'
    assert differences(a,b)==['Query','Budget']


def test_history_ui_comparison_never_generates(tmp_path):
    from streamlit.testing.v1 import AppTest
    script=tmp_path/'history_host.py'
    script.write_text('''import streamlit as st
from demo.live.ui import render_history
from demo.live.history import remember
from demo.tests.test_live import FakeResources,PASSAGES
from demo.live.engine import run
if 'history' not in st.session_state:
    resource=FakeResources()
    history=[]
    for method in ('RAW_SKIP','FIXED_SENTENCE','QBGF_FULL'):
        remember(history,run('Which beta sensor?',PASSAGES,method,256,'SFR',resource,True,scoring_mode='controlled_sfr'))
    st.session_state['history']=history
    st.session_state['resource']=resource
render_history(st.session_state['history'],technical=True)
''')
    app=AppTest.from_file(str(script)).run(timeout=20)
    assert not app.exception
    assert app.dataframe[0].value['Answer'].tolist()==['A concise answer.']*3
    calls=len(app.session_state['resource'].calls)
    app.multiselect[0].set_value([1,3]).run()
    next(b for b in app.button if b.label=='Compare Selected Runs').click().run()
    assert not app.exception,[e.message for e in app.exception]
    assert len(app.session_state['resource'].calls)==calls==3
    assert any(x.value=='Changed: Method.' for x in app.caption)
    assert sum(x.label=='Evidence units' for x in app.metric)==2
    app.multiselect[0].set_value([2,3]).run()
    next(b for b in app.button if b.label=='Compare Selected Runs').click().run()
    assert not app.exception and len(app.session_state['resource'].calls)==calls
