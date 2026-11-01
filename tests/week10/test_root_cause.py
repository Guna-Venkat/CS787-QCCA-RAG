import json
from pathlib import Path
import pytest
from src.week10.root_cause.storage import Store,atomic
from src.week10.root_cause.analysis import oracle,capture,stability,retrieval_metrics
from src.week10.root_cause.encoding import rank,Encoder
from src.week10.root_cause.protocol import guard,load
from src.week10.root_cause.smoke import run


def test_resume_no_duplicate_calls(tmp_path):
    assert run(tmp_path)['new_fake_calls']==4
    assert run(tmp_path)['new_fake_calls']==0


def test_metadata_does_not_count_as_answer(tmp_path):
    s=Store(tmp_path,{'version':1});s.put('q',{'answer':'x'});atomic(tmp_path/'complete.json',{'done':True})
    assert s.rows()==[{'answer':'x'}]


def test_changed_provenance_rejected(tmp_path):
    Store(tmp_path,{'v':1})
    with pytest.raises(ValueError):Store(tmp_path,{'v':2})


def test_corrupt_checkpoint_rejected(tmp_path):
    s=Store(tmp_path,{'v':1});s.put('q',{'a':1});p=s.filename('q');o=json.loads(p.read_text());o['row']['a']=2;p.write_text(json.dumps(o))
    with pytest.raises(ValueError):s.get('q')


def test_no_overwriting_answers(tmp_path):
    s=Store(tmp_path,{});s.put('q',{'a':1})
    with pytest.raises(ValueError):s.put('q',{'a':2})


@pytest.mark.parametrize('split',['test','TEST','confirmation'])
def test_test_closed(split):
    with pytest.raises(ValueError):guard(split)


def test_bm25_ranking_ignores_gold():
    inf=dict(question='cats',chunks=[dict(chunk_id='b',order=1,text='dogs'),dict(chunk_id='a',order=0,text='cats')])
    ranked,_=rank(inf,'BM25');assert ranked[0]['chunk_id']=='a'


def test_dense_ties_source_order():
    class E:
        def encode(self,texts,role='document'):return [[1.,0.] for _ in texts],dict(cost=0)
    inf=dict(question='q',chunks=[dict(chunk_id='z',order=1,text='a'),dict(chunk_id='y',order=0,text='b')])
    assert rank(inf,'MiniLM',E())[0][0]['chunk_id']=='y'


def test_cache_identity_text_role_revision(tmp_path):
    e=Encoder(dict(id='x',revision='a'),tmp_path)
    assert e.key('q','query')!=e.key('q','document')!=e.key('other','document')
    f=Encoder(dict(id='x',revision='b'),tmp_path)
    assert e.key('q','query')!=f.key('q','query')
    e.db.close();f.db.close()


def test_oracle_epsilon_smallest_k():
    rows=[dict(question_id='q',paper_id='p',action='k'+str(k),k=k,f1=f) for k,f in [(0,.4),(1,.595),(2,.6)]]
    o=oracle(rows);assert o['epsilon_action_distribution']=={'k1':1};assert o['exact_action_distribution']=={'k2':1}
    assert capture(.4,.4,.4) is None


def test_incomplete_oracle_rejected():
    with pytest.raises(ValueError):oracle([dict(question_id='q',action='a',f1=1),dict(question_id='r',action='b',f1=1)])


def test_seed_stability():
    r=stability([dict(question_id='q',action='x',delta=d) for d in [.2,-.2,.1]])
    assert r[0]['agreement']==pytest.approx(2/3);assert not r[0]['sign_stable']


def test_no_gold_evidence_is_undefined():
    row=retrieval_metrics([],dict(gold_chunk_sets=[]),len)[0]
    assert row['any_recall'] is None and row['all_recall'] is None


def test_notebook_runall_controls():
    n=json.loads(Path('notebooks/week10/week10_root_cause.ipynb').read_text())
    source='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
    for flag in ['RUN_RETRIEVER_FULL','RUN_COMPRESSION_FULL','RUN_READER_FULL','RUN_DATASET_ANALYSIS_FULL','RUN_POLICY_EXPERIMENT']:
        assert flag+' = False' in source
    assert 'SMOKE_ONLY = False' in source
    assert 'run_pending(' in source and 'phase_approved(' in source


def test_actual_reader_runner_resumes_interrupted_call(tmp_path,monkeypatch):
    import src.week10.root_cause.runner as r
    import src.week10.root_cause.runtime as runtime
    cfg=dict(output_root=str(tmp_path),k_values=[2,4,6,8])
    records=[dict(inference=dict(question_id='q',paper_id='p',question='Q'),evaluation={})]
    retrieved={'q':dict(ranked=[dict(chunk_id=str(i),text=str(i)) for i in range(10)])}
    monkeypatch.setattr(r,'prepare',lambda cfg:records)
    monkeypatch.setattr(r,'retrieval_rows',lambda *args:retrieved)
    monkeypatch.setattr(r,'identity',lambda *args:{'synthetic':True})
    monkeypatch.setattr(r,'score',lambda *args:dict(token_f1=.5,exact_match=0))
    calls=[];loads=[];fail=[True]
    class Fake:
        def __init__(self,cfg):loads.append(1)
        def generate(self,q,top,k,v):
            if k==4 and fail[0]:raise RuntimeError('simulated interruption')
            calls.append(k);return dict(content='x',prompt_tokens=10)
        def close(self):pass
    monkeypatch.setattr(runtime,'SaraReader',Fake)
    with pytest.raises(RuntimeError):r.run_r(cfg,'BM25','full')
    assert calls==[2]
    fail[0]=False;r.run_r(cfg,'BM25','full')
    assert calls==[2,4,6,8]
    r.run_r(cfg,'BM25','full')
    assert len(loads)==2  # completed rerun loads NO model


def test_generation_requires_manual_execute(monkeypatch):
    import sys
    import src.week10.root_cause.runner as r
    monkeypatch.setattr(r,'load',lambda p:{})
    monkeypatch.setattr(sys,'argv',['runner','--config','unused','--phase','R'])
    with pytest.raises(RuntimeError,match='Manual'):r.main()


def test_policy_is_disabled(monkeypatch):
    import sys
    import src.week10.root_cause.runner as r
    monkeypatch.setattr(r,'load',lambda p:{})
    monkeypatch.setattr(sys,'argv',['runner','--config','unused','--phase','P','--execute'])
    with pytest.raises(RuntimeError,match='Policy research disabled'):r.main()


def test_downstream_phase_needs_upstream_freeze(tmp_path):
    from src.week10.root_cause.runner import phase_gate
    with pytest.raises(RuntimeError,match='Phase R'):phase_gate({'output_root':str(tmp_path)},'C')


def test_source_sentences_and_order():
    from src.week10.root_cause.runner import compact
    class E:
        def encode(self,texts,role='document'):
            if role=='query':return [[1.,0.]],{}
            return [[1.,i*.01] for i,_ in enumerate(texts)],{}
    class T:
        def encode(self,text,**kw):return text.split()
    chunks=[dict(chunk_id='a',text='Dr. Smith wrote a paper. It cites Fig. 2.'),dict(chunk_id='b',text='A second exact sentence.')]
    text,selected=compact('Q',chunks,E(),T(),100)
    assert len(selected)>=3
    assert all(s['text'] in chunks[s['chunk_order']]['text'] for s in selected)
    assert selected==sorted(selected,key=lambda s:(s['chunk_order'],s['start']))


def test_report_pairs_and_oracle(tmp_path,monkeypatch):
    import src.week10.root_cause.report as report
    from src.week10.root_cause.storage import digest
    monkeypatch.setattr(report,'freeze',lambda cfg:{'frozen':'synthetic'})
    s=Store(tmp_path/'runs/C-Mistral',{'synthetic':True})
    for q in ['a','b']:
        for a,f in [('no_evidence',.4),('qcs256',.6)]:
            s.put([q,a],dict(question_id=q,paper_id='p'+q,action=a,k=0 if a=='no_evidence' else 3,
                f1=f,exact_match=0,prediction=a,cumulative_prompt_tokens=10,generation=dict(prompt_tokens=10)))
    atomic(s.path/'complete.json',{'done':True})
    report.report({'output_root':str(tmp_path)})
    import csv
    rows=list(csv.DictReader((tmp_path/'summary.csv').open()))
    assert next(r for r in rows if r['action']=='qcs256')['HELP']=='1.0'
    assert (tmp_path/'bootstrap.csv').exists()
    assert json.loads((tmp_path/'decision_R.json').read_text())['GO_NEXT'] is False


def test_synthetic_smoke_resumes_partial_fixture(tmp_path):
    s=Store(tmp_path,dict(kind='synthetic',version=1))
    s.put(['a',1],dict(question_id='a',paper_id='papera',action='k1',k=1,f1=.5,cumulative_prompt_tokens=10))
    assert run(tmp_path)['new_fake_calls']==3


def test_gold_span_union_coverage_not_lexical_guess():
    from src.week10.root_cause.data import evidence_occurrences
    from src.week10.root_cause.analysis import retrieval_metrics
    text='Alpha  beta gamma. Another line.'
    spans=evidence_occurrences(text,['Alpha beta gamma.'])
    chunks=[dict(chunk_id='a',text='Alpha  beta ',char_start=0,char_end=12),dict(chunk_id='b',text='gamma.',char_start=12,char_end=18)]
    metrics=retrieval_metrics(chunks,dict(gold_occurrences=spans),len)
    assert metrics[0]['any_recall']==0
    assert metrics[1]['all_recall']==1
    assert metrics[1]['coverage']==1


def test_repeated_gold_occurrences_are_alternatives():
    from src.week10.root_cause.data import evidence_occurrences
    from src.week10.root_cause.analysis import retrieval_metrics
    text='Same evidence. Other. Same evidence.'
    occurrences=evidence_occurrences(text,['Same evidence.','Not in document'])
    start=text.rindex('Same')
    chunks=[dict(chunk_id='a',text=text[start:],char_start=start,char_end=len(text))]
    m=retrieval_metrics(chunks,dict(gold_occurrences=occurrences),len)[0]
    assert m['any_recall']==1 and m['all_recall']==0 and m['unmapped_gold_spans']==1


def test_branch_main_rejected(monkeypatch):
    import src.week10.root_cause.protocol as p
    monkeypatch.setattr(p.subprocess,'check_output',lambda *a,**kw:'main\n')
    with pytest.raises(RuntimeError,match='only on gagan'):p.guard('dev')


def test_old_week_output_root_rejected(tmp_path):
    from src.week10.root_cause.protocol import load
    p=tmp_path/'config.json';p.write_text(json.dumps(dict(split='dev',output_root='results/week9/not_allowed')))
    with pytest.raises(ValueError,match='Week-10 output required'):load(p)


def test_notebook_hash_ignores_outputs_counts_metadata_and_run_flags(tmp_path):
    from src.week10.root_cause.preflight import notebook_source_hash
    p=tmp_path/'notebook.ipynb'
    nb=dict(cells=[dict(cell_type='code',source=['RUN_RETRIEVER_FULL = False\n','answer = 42\n'],outputs=[],execution_count=None)],metadata={})
    p.write_text(json.dumps(nb));expected=notebook_source_hash(p)
    nb['cells'][0].update(outputs=[dict(output_type='stream',text='saved output')],execution_count=8,metadata={'collapsed':True})
    nb['metadata']={'kernel':'other'}
    nb['cells'][0]['source'][0]='RUN_RETRIEVER_FULL = True\n'
    p.write_text(json.dumps(nb));assert notebook_source_hash(p)==expected
    nb['cells'][0]['source'][1]='answer = 43\n'
    p.write_text(json.dumps(nb));assert notebook_source_hash(p)!=expected


def test_notebook_hash_keeps_markdown_and_config_changes(tmp_path):
    from src.week10.root_cause.preflight import notebook_source_hash
    p=tmp_path/'notebook.ipynb'
    nb=dict(cells=[dict(cell_type='markdown',source='Frozen protocol'),dict(cell_type='code',source="CONFIG = 'v1.json'\n")])
    p.write_text(json.dumps(nb));expected=notebook_source_hash(p)
    nb['cells'][1]['source']="CONFIG = 'other.json'\n"
    p.write_text(json.dumps(nb));assert notebook_source_hash(p)!=expected
    nb['cells'][1]['source']="CONFIG = 'v1.json'\n";nb['cells'][0]['source']='Changed protocol'
    p.write_text(json.dumps(nb));assert notebook_source_hash(p)!=expected
