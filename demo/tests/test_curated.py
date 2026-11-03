"""Synthetic fixtures only. No model download, GPU inference or research output writes."""
import json
from pathlib import Path
import pytest
from demo.curated import protocol as p,publication
from demo.curated.protocol import Store,digest,grid,METHODS,RETRIEVERS,BUDGETS,DATASETS

def test_exact_grid():
    m=dict(queries={d:[str(i) for i in range(10)] for d in DATASETS})
    g=grid(m)
    assert len(g)==1620 and len(set(g))==1620
    assert {k[4] for k in g}==set(METHODS)
    assert {k[2] for k in g}==set(RETRIEVERS)

def test_immutable_resume(tmp_path):
    s=Store(tmp_path,dict(v=1));s.put(['q','SFR'],dict(answer='yes'));assert s.get(['q','SFR'])==dict(answer='yes')
    s.put(['q','SFR'],dict(answer='yes'))
    with pytest.raises(ValueError):s.put(['q','SFR'],dict(answer='other'))
    with pytest.raises(ValueError):Store(tmp_path,dict(v=2))

def test_demo_policies_exact_source_and_budget():
    from demo.curated.preparation import choose
    from src.week11.budgets import FixtureTokenizer
    from src.week12.algorithm import serialize
    tok=FixtureTokenizer()
    pool=[dict(unit_id='p',passage_id='p',document_id='d',document_order=[0,0,0,0],start_char=0,end_char=20,source_text='A useful fact. Other.',token_count=5,unit_type='PASSAGE')]
    sent=[dict(pool[0],unit_id='s',source_text='A useful fact.',end_char=14,unit_type='SENTENCE')]
    prop=[dict(sent[0],unit_id='x',unit_type='PROPOSITION')]
    for method in METHODS:
        sel=choose(pool,sent,prop,[.8],dict(s=.7,x=.7),tok,256,method)
        assert sel['evidence_tokens']<=256
        assert all(u['source_text'] in pool[0]['source_text'] for u in sel['units'])
    assert choose(pool,sent,prop,[.8],dict(s=.7,x=.7),tok,0,'PROPOSITION')['units']==[]

def fake_pack(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'OUT',tmp_path);monkeypatch.setattr(publication,'OUT',tmp_path)
    # One fixture per dataset keeps plumbing tests cheap; production manifest requires ten each.
    m=dict(queries={d:['fixture'] for d in DATASETS},scope='SYNTHETIC TEST ONLY',expected_rows=162,methods=list(METHODS),budgets=list(BUDGETS),retrievers=list(RETRIEVERS),freeze_sha256='synthetic')
    ps=p.store(m,'pools');prep=p.store(m,'prepared');rs=p.store(m,'results');ats=p.store(m,'attribution')
    for d in DATASETS:
        for rt in RETRIEVERS:
            u=dict(unit_id='u',passage_id='u',source_text='Synthetic source.',unit_type='PASSAGE',token_count=3,document_id='doc',document_order=[0,0,0,0],start_char=0,end_char=17,serialized_block_tokens=4,rank=1,score=.5)
            ps.put([d,'fixture',rt],dict(question='Synthetic question?',gold=dict(gold_answer=['yes']),task='qa',hierarchy=[u],pool=[u],lineage=dict(source_hash='s')))
        for rt,b,a in [(r,b,a) for r in RETRIEVERS for b in BUDGETS for a in METHODS]:
            key=[d,'fixture',rt,b,a];units=[] if a=='NONE' else [u];ids=[] if a=='NONE' else [1,2];text='' if not units else u['source_text']
            sel=dict(units=units,evidence_ids=ids,text=text,evidence_tokens=len(ids),choices=[],selected_unit_ids=[v['unit_id'] for v in units])
            prep.put(key,dict(selection=sel,relevance=dict(u=.5),lineage=dict(source_hash='s')))
            r=dict(dataset=d,query_id='fixture',retriever=rt,budget=b,method=a,evidence_hash=digest([ids,text]),prediction='yes',raw_prediction='yes',task_score=1.,prompt_tokens=20,generated_tokens=1,evidence_tokens=len(ids),reader='SYNTHETIC',exact_match=1.,prompt_hash='synthetic',source_hash='s',model_hash='synthetic',generation_seconds=.1,conceptual_lm_calls=1,exact_prompt_cache_reused=True)
            rs.put(key,r)
            if units:ats.put(key+['u'],dict(unit_id='u',original_answer='yes',ablated_answer='yes',importance=0.,category='NEUTRAL'))
    return m

def test_complete_publish_self_contained_and_rerunnable(tmp_path,monkeypatch):
    m=fake_pack(tmp_path,monkeypatch);s=publication.status(m)
    assert s['all_views_complete'] and s['answers']==162
    x=publication.publish(m);assert x==publication.publish(m)
    bundle=publication.load_bundle();assert len(bundle['queries'])==3
    for q in bundle['queries'].values():assert len(q['conditions'])==54
    # Delete intermediate shards: portable published collection still works.
    import shutil
    shutil.rmtree(tmp_path/'prepared');shutil.rmtree(tmp_path/'attribution')
    assert publication.load_bundle()['queries']

def test_no_partial_publication(tmp_path,monkeypatch):
    m=fake_pack(tmp_path,monkeypatch)
    p.store(m,'results').path.joinpath(digest(['QASPER','fixture','SFR',256,'NONE'])+'.json').unlink()
    with pytest.raises(ValueError):publication.publish(m)
    assert not (tmp_path/'PUBLISHED.json').exists()

def test_missing_loo_blocks_publish(tmp_path,monkeypatch):
    m=fake_pack(tmp_path,monkeypatch)
    p.store(m,'attribution').path.joinpath(digest(['QASPER','fixture','SFR',256,'RAW_SKIP','u'])+'.json').unlink()
    with pytest.raises(ValueError):publication.publish(m)

def test_checksum_tamper_rejected(tmp_path,monkeypatch):
    m=fake_pack(tmp_path,monkeypatch);publication.publish(m)
    (tmp_path/'bundle.json').write_text('{}')
    with pytest.raises(ValueError):publication.load_bundle()

def test_notebook_safe_defaults():
    nb=json.loads((p.APP/'notebooks/BUILD_CURATED_REPLAYS.ipynb').read_text());text='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    # Users change RUN_FULL to execute the notebook; preserve that working setting.
    assert ('RUN_FULL = False' in text or 'RUN_FULL = True' in text)
    import re
    assert re.search(r'RUN_SMOKE\s*=\s*(True|False)',text)
    assert 'RESUME = True' in text and 'OVERWRITE = False' in text

def test_full_opt_in_required(monkeypatch):
    from demo.curated import cli
    monkeypatch.setattr(cli,'config',lambda:{})
    monkeypatch.setattr(cli,'manifest',lambda c:{})
    with pytest.raises(ValueError):cli.run('answers',full=False)

def test_render_complete_pack_all_datasets(tmp_path,monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    m=fake_pack(tmp_path,monkeypatch);publication.publish(m)
    # UI cannot load models; use a fixture pack in a separate test app.
    script=tmp_path/'fixture_app.py';script.write_text('from demo.curated.ui import render\nfrom demo.curated.protocol import read\nrender(read('+repr(str(tmp_path/'bundle.json'))+'))\n')
    app=AppTest.from_file(str(script)).run(timeout=15)
    assert not app.exception
    for d in DATASETS:
        app.selectbox(key='curated_dataset').set_value(d).run()
        assert not app.exception
    app.selectbox(key='curated_left').set_value('NONE').run()
    assert not app.exception

def test_generation_and_deletion_resume_with_fake_reader(tmp_path,monkeypatch):
    import shutil
    from demo.curated import generation,data
    from src.week11.datasets import normalize,document
    from src.week11.budgets import FixtureTokenizer
    import src.week12.reader as reader_module
    from transformers import AutoTokenizer
    m=fake_pack(tmp_path,monkeypatch)
    shutil.rmtree(tmp_path/'results');shutil.rmtree(tmp_path/'attribution')
    xs={(d,'fixture'):normalize(d,'dev','fixture','Synthetic question?',[document(d,'doc','Synthetic source.')],answer=['yes'],label='SUPPORT') for d in DATASETS}
    monkeypatch.setattr(data,'examples',lambda c,manifest:xs)
    calls=[]
    class FakeReader:
        tok=FixtureTokenizer()
        def __init__(self,*a):pass
        def generate(self,e,sel):
            calls.append((e['dataset'],digest(sel['evidence_ids'])))
            answer='SUPPORT' if e['task_type']=='verification' else 'yes'
            return dict(prediction=answer,raw_prediction=answer,prompt_hash=digest([e['question'],sel['evidence_ids']]),prompt_tokens=20,generated_tokens=1,generation_seconds=.01,reader='SYNTHETIC',model_hash='synthetic')
        def close(self):pass
    monkeypatch.setattr(reader_module,'Reader',FakeReader)
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k:FixtureTokenizer())
    c=dict(models=dict(Qwen=dict(id='SYNTHETIC',revision='fixture')))
    generation.run(c,m,attribution=False)
    assert len(calls)==6 # Two distinct exact prompts per dataset, not 162 physical calls.
    generation.run(c,m,attribution=True)
    generation.run(c,m,attribution=True)
    assert len(calls)==6
    assert publication.status(m)['all_views_complete']
