"""Synthetic custom inputs; no model weights, GPU or research output writes."""
import threading
import numpy as np
import pytest
from demo.live.engine import parse_passages,validate,run,explanation,safe_answer,STAGES,input_key

class CharTokenizer:
    eos_token_id=0
    def encode(self,text,**kwargs):return [ord(c) for c in text]
    def decode(self,ids,**kwargs):return ''.join(chr(i) for i in ids)
    def apply_chat_template(self,messages,**kwargs):return '[USER]'+messages[0]['content']+'[ANSWER]'
class FakeResources:
    def __init__(self,missing=False):self.device="cpu";self.lock=threading.RLock();self.tok=CharTokenizer();self.missing=missing;self.calls=[];self.embeds=[];self.released=False;self.encoder=None;self.reader=None
    def available_memory_mib(self):return None
    def tokenizer(self):return self.tok
    def embed(self,texts,role):
        self.embeds.append((texts,role));return np.array([[float('alpha' in t.lower()),float('beta' in t.lower()),1.] for t in texts])
    def generate(self,query,selection):
        if self.missing:raise RuntimeError('missing local reader')
        from src.week12.reader import prompt_ids
        self.calls.append((query,selection['text']))
        return dict(prediction='A concise answer.',prompt_tokens=len(prompt_ids(self.tok,query,'freeform_qa',selection['evidence_ids'])),generated_tokens=4,generation_seconds=.01)
    def release_gpu(self):self.released=True

PASSAGES=['Alpha sensors recorded a signal. The sampling interval was one second.','Beta sensors were selected for remote operation. Their batteries lasted forty-eight hours.','The expedition lasted thirty-six hours. Remote operation required a battery lasting at least that long.','The enclosures had different colors. Color was a secondary requirement.','The report compared measurement intervals and battery duration.']

def test_parse_multiple_headers():
    assert parse_passages('Passage 1:\nAlpha.\n\n---\nPassage 2:\nBeta.')==['Alpha.','Beta.']
    assert parse_passages('One passage.')==['One passage.']
@pytest.mark.parametrize('query,passages',[('',PASSAGES),('  ',PASSAGES),('Q',[]),('Q',['']),('Q',['ok',' ']),('Q',['x']*21)])
def test_validation_errors(query,passages):
    with pytest.raises(ValueError):validate(query,passages,'RAW_SKIP',256,'BM25')

def test_none_skips_and_query_only():
    rsrc=FakeResources();r=run('Which sensor?',PASSAGES[:3],'NONE',256,'SFR',rsrc,enable_models=True)
    assert r['selection']['evidence_tokens']==0 and r['selection']['units']==[]
    assert r['stages']['Retriever']['state']==r['stages']['Granularity']['state']=='skipped'
    assert rsrc.calls[0][1]=='' and not rsrc.embeds
    assert not {'f1','gold','exact_match','correctness'} & set(r)

def test_raw_skip_trace_and_remaining_budget():
    r=run('unmatchedquery',['A'*180,'B'*310,'C'*60,'D'*300,'E'*300],'RAW_SKIP',256,'BM25',FakeResources())
    decisions=r['allocation_trace'];assert decisions[0]['decision']=='INCLUDED'
    assert decisions[1]['decision']=='DOES NOT FIT — SKIPPED'
    assert decisions[2]['decision']=='INCLUDED'
    assert decisions[0]['remaining_budget']==74 and decisions[2]['remaining_budget']==12
    assert r['selection']['evidence_tokens']==244

@pytest.mark.parametrize('method',['RAW_SKIP','FIXED_SENTENCE','QBGF_FULL'])
@pytest.mark.parametrize('budget',[256,1024])
def test_all_budgeted_methods_exact_context(method,budget):
    res=FakeResources();r=run('Which beta sensor?',PASSAGES,method,budget,'BM25',res,True,scoring_mode='controlled_sfr' if method=='QBGF_FULL' else 'pipeline_native')
    from src.week12.algorithm import serialize
    selected=r['selection'];assert selected['evidence_tokens']<=budget
    rebuilt=serialize(selected['units'],res.tok);assert rebuilt['evidence_ids']==selected['evidence_ids']
    assert res.calls[0][1]==selected['text']
    for u in selected['units']:assert u['source_text']==PASSAGES[int(u['document_id'][1:])-1][u['start_char']:u['end_char']]
    assert [u['document_order'] for u in selected['units']]==sorted(u['document_order'] for u in selected['units'])
    if method=='QBGF_FULL':assert all(x['option_id'] in ['DROP','SENT_1','SENT_2','SENT_4','SENT_8','FULL'] for x in r['allocation_trace'])

def test_retriever_ranking_and_dense_comparison_cache():
    res=FakeResources();cache={}
    a=run('beta',PASSAGES,'FIXED_SENTENCE',256,'SFR',res,True,prepared_cache=cache)
    assert a['ranking'][0]['unit']['document_id']=='P2'
    before=len(res.embeds)
    b=run('beta',PASSAGES,'RAW_SKIP',256,'SFR',res,True,prepared_cache=cache)
    assert len(res.embeds)==before+1 # query lookup only; candidate embeddings/ranking are reused
    assert a['input_key']==b['input_key'] and a['ranking']==b['ranking']

def test_missing_reader_retains_context():
    r=run('beta',PASSAGES,'FIXED_SENTENCE',256,'BM25',FakeResources(missing=True),True)
    assert r['answer'] is None and r['selection']['units'] and r['prompt_tokens']>0
    assert r['stages']['Reader']['state']=='failed' and r['stages']['Answer']['state']=='completed'
    assert 'unavailable' in r['explanation'][-1]

def test_model_disabled_cpu_context_only_and_notifications():
    events=[];res=FakeResources();r=run('beta',PASSAGES,'FIXED_SENTENCE',256,'BM25',res,False,lambda *event:events.append(event))
    assert not res.calls and r['answer'] is None and r['selection']['evidence_tokens']>0
    assert all(r['stages'][s]['state'] in ('completed','skipped') for s in STAGES)
    assert any(s=='Retriever' and state=='completed' and states[s].get('preview') for s,state,detail,states in events)
    assert len(r['explanation'])>=5

def test_no_reader_call_during_retriever_only_compare():
    res=FakeResources();r=run('beta',PASSAGES,'RAW_SKIP',256,'SFR',res,True,generate_answer=False)
    assert res.embeds and not res.calls and r['answer'] is None

def test_no_hidden_thinking():
    assert safe_answer('<think>secret</think>Visible answer.')=='Visible answer.'
    assert safe_answer('<think>secret') is None
    assert safe_answer('Hidden prefix</think>Answer only')=='Answer only'

def test_method_comparison_input_signature():
    a=input_key('q',PASSAGES,256,'BM25');assert a==input_key('q',PASSAGES,256,'BM25')
    assert a!=input_key('q',PASSAGES,1024,'BM25')

def test_progress_html_escapes_details():
    from demo.live.ui import pipeline_html
    states={s:dict(state='pending',detail='<script>') for s in STAGES}
    assert '<script>' not in pipeline_html(states)

@pytest.mark.parametrize('method,budget',[('NONE',256),('RAW_SKIP',256),('FIXED_SENTENCE',256),('FIXED_SENTENCE',1024),('QBGF_FULL',256)])
def test_live_ui_cases(tmp_path,method,budget):
    from streamlit.testing.v1 import AppTest
    # Inject a fake resource explicitly into a tiny test-only host. No live model loads.
    script=tmp_path/'live_host.py';script.write_text('''import streamlit as st
from demo.live import ui
from demo.tests.test_live import FakeResources,PASSAGES
ui.resources=lambda:FakeResources(missing=True)
ui.available=lambda:dict(sfr_local=True,qwen_local=False,gpu_free_mib=None)
if 'live_passage_ids' not in st.session_state:
    st.session_state['live_passage_ids']=list(range(5));st.session_state['live_next_id']=5
    for i,p in enumerate(PASSAGES):st.session_state[f'live_passage_{i}']=p
    st.session_state['live_query']='Which beta sensor?'
    st.session_state['live_method']='''+repr(method)+'''
    st.session_state['live_budget']='''+repr(budget)+'''
    st.session_state['live_models_enabled']=True
    if st.session_state['live_method']=='QBGF_FULL':
        st.session_state['live_models_enabled']=True
        st.session_state['live_scoring_mode']='controlled_sfr'
ui.render()
''')
    app=AppTest.from_file(str(script)).run(timeout=20);assert not app.exception
    if method=='FIXED_SENTENCE' and budget==1024:
        next(t for t in app.toggle if t.label=='Enable Live Models').set_value(True).run()
    next(b for b in app.button if b.label=='Run Evidence Pipeline').click().run(timeout=20)
    assert not app.exception,[e.message for e in app.exception]
    assert any('Reader generation unavailable' in i.value for i in app.info)
    assert not any(m.label in ('F1','Answer F1','Correctness','EM') for m in app.metric)
    assert any(m.value=='What happened in this run?' for m in app.subheader)
    next(b for b in app.button if b.label=='Run comparison').click().run(timeout=20)
    assert not app.exception

def test_gpu_guard_counts_same_process_reusable_cache(monkeypatch):
    import torch
    from demo.live.resources import LiveResources,ModelUnavailable
    r=LiveResources()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(8*2**30,24*2**30))
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda:15*2**30)
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda:0)
    r._memory(18000)
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda:15*2**30)
    with pytest.raises(ModelUnavailable):r._memory(18000)

def test_cpu_tokenizer_uses_bundled_assets_without_reader_cache(monkeypatch):
    from demo.live.resources import LiveResources
    from demo.constants import ROOT
    from transformers import AutoTokenizer
    original=AutoTokenizer.from_pretrained;paths=[]
    def local_only(path,*args,**kwargs):
        paths.append(str(path));assert str(path)==str(ROOT/'demo/assets/live_tokenizer')
        assert kwargs['local_files_only'] is True
        return original(path,*args,**kwargs)
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',local_only)
    tok=LiveResources().tokenizer()
    assert tok.encode('Exact source sentence.') and len(paths)==1

def test_cpu_reader_sequential_handoff_and_cleanup(monkeypatch):
    import demo.live.resources as mod
    from types import SimpleNamespace
    r=mod.LiveResources('cpu');r.tok=CharTokenizer();events=[]
    r.encoder=SimpleNamespace()
    original_release=r.release_gpu
    def release():
        events.append(('release',r.encoder is not None,r.reader is not None));original_release()
    class TinyReader:
        def __init__(self,cfg,tok):
            assert r.encoder is None
            self.model=None;events.append(('reader_load',))
        def generate(self,*args):return {'prediction':'answer'}
    monkeypatch.setattr(r,'release_gpu',release)
    monkeypatch.setattr(r,'_cpu_memory',lambda:None)
    monkeypatch.setattr(mod,'local_model',lambda spec:True)
    monkeypatch.setattr(mod,'CPUReader',TinyReader)
    assert r.generate('q',{})['prediction']=='answer'
    assert r.encoder is None and r.reader is None
    assert events==[('release',True,False),('reader_load',),('release',False,True)]

def test_cpu_reader_cleanup_when_generation_fails(monkeypatch):
    import demo.live.resources as mod
    r=mod.LiveResources('cpu');r.tok=CharTokenizer()
    class BrokenReader:
        def __init__(self,*args):self.model=None
        def generate(self,*args):raise RuntimeError('fixture failure')
    monkeypatch.setattr(r,'_cpu_memory',lambda:None)
    monkeypatch.setattr(mod,'local_model',lambda spec:True)
    monkeypatch.setattr(mod,'CPUReader',BrokenReader)
    with pytest.raises(RuntimeError,match='fixture failure'):r.generate('q',{})
    assert r.encoder is None and r.reader is None

def test_cpu_low_memory_fails_without_loading(monkeypatch):
    import demo.live.resources as mod
    monkeypatch.setattr(mod,'cpu_available_mib',lambda:2000)
    with pytest.raises(mod.ModelUnavailable,match='available RAM'):mod.LiveResources('cpu')._cpu_memory()

def test_cpu_mode_never_needs_cuda_guard(monkeypatch):
    from demo.live.resources import LiveResources
    r=LiveResources('cpu')
    monkeypatch.setattr(r,'_memory',lambda amount:pytest.fail('CUDA guard used'))
    assert r.device=='cpu'
    with pytest.raises(ValueError):r.set_device('invalid')

def test_cpu_context_only_unloads_encoder():
    res=FakeResources();res.device='cpu'
    out=run('beta',PASSAGES,'RAW_SKIP',256,'SFR',res,True,generate_answer=False)
    assert out['device']=='cpu' and res.released

def test_cpu_reader_loader_pins_device_dtype_and_revision(monkeypatch):
    import torch
    from transformers import AutoModelForCausalLM
    from demo.live.resources import LiveResources,CPUReader
    calls=[]
    class TinyModel:
        def eval(self):return self
    def load(name,**kwargs):calls.append((name,kwargs));return TinyModel()
    monkeypatch.setattr(AutoModelForCausalLM,'from_pretrained',load)
    cfg=LiveResources().cfg
    reader=CPUReader(cfg,CharTokenizer())
    name,opts=calls[0]
    assert name==cfg['models']['Qwen']['id']
    assert opts['revision']==cfg['models']['Qwen']['revision']
    assert opts['device_map']=={'':'cpu'} and opts['torch_dtype']==torch.float16
    assert opts['local_files_only'] and opts['attn_implementation']=='sdpa'


def test_cpu_reader_prompt_and_generation_match_frozen_config(monkeypatch):
    import torch
    from demo.live.resources import CPUReader,LiveResources
    from src.week12.reader import prompt_ids
    cfg=LiveResources().cfg;tok=CharTokenizer();reader=CPUReader.__new__(CPUReader)
    reader.cfg=cfg;reader.tok=tok
    class TinyModel:
        def generate(self,**kw):
            assert kw['input_ids'].device.type=='cpu'
            assert kw['input_ids'][0].tolist()==prompt_ids(tok,'q','freeform_qa',[65],cfg['labels'])
            assert kw['do_sample'] is False
            assert kw['max_new_tokens']==cfg['generation']['max_new_tokens_qa']
            return torch.cat([kw['input_ids'],torch.tensor([[79,75]])],dim=1)
    reader.model=TinyModel()
    result=reader.generate(dict(question='q',task_type='freeform_qa'),{'evidence_ids':[65]})
    assert result['prediction']=='OK' and result['generated_tokens']==2
    assert result['device']=='cpu' and result['dtype']=='float16'

def test_cpu_ui_offers_sfr_without_gpu(tmp_path):
    from streamlit.testing.v1 import AppTest
    script=tmp_path/'cpu_live_host.py'
    script.write_text('''import streamlit as st
from demo.live import ui
from demo.tests.test_live import FakeResources,PASSAGES
ui.resources=lambda:FakeResources(missing=True)
ui.available=lambda:dict(sfr_local=True,qwen_local=True,gpu_free_mib=None)
if 'live_passage_ids' not in st.session_state:
    st.session_state['live_passage_ids']=[0,1]
    st.session_state['live_next_id']=2
    for i,p in enumerate(PASSAGES[:2]):st.session_state[f'live_passage_{i}']=p
    st.session_state['live_query']='Which beta sensor?'
ui.render()
''')
    app=AppTest.from_file(str(script)).run(timeout=20)
    assert not app.exception
    next(t for t in app.toggle if t.label=='Enable Live Models').set_value(True).run()
    retriever=next(s for s in app.selectbox if s.label=='Retriever')
    assert 'SFR' in retriever.options
    retriever.select('SFR').run()
    next(b for b in app.button if b.label=='Run Evidence Pipeline').click().run(timeout=20)
    assert not app.exception
    assert any(h.value=='Retrieved Candidate Pool' for h in app.subheader)
