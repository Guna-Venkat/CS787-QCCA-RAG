"""Run All scheduling tests; all generation/subprocess entry points are mocked."""
import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from src.week10.root_cause.storage import Store,atomic,digest

NOTEBOOK=Path(__file__).resolve().parents[2]/'notebooks/week10/week10_root_cause.ipynb'


def make_workflow(tmp_path,monkeypatch,partial_c=False):
    import src.week10.root_cause.protocol as protocol
    monkeypatch.setattr(protocol,'source_hashes',lambda:{})
    dataset=tmp_path/'dev.jsonl';dataset.write_text(json.dumps(dict(id='q',example_id='p'))+'\n')
    cfg=dict(dataset=str(dataset),k_values=[2,4,6,8],output_root='results/week10/root_cause/v1',reader_panel=['Mistral','Qwen','Llama'])
    root=tmp_path/cfg['output_root'];root.mkdir(parents=True)
    frozen=dict(config=cfg,sources={});atomic(root/'execution_freeze.json',frozen)
    atomic(root/'freeze_R.json',dict(phase='R',GO_NEXT=True,retriever='SFR',reason='already reviewed',execution_freeze_hash=digest(frozen)))
    tiny=tmp_path/'results/week10/root_cause/gpu_smoke_v1'
    def surface(target,name,keys):
        store=Store(target/'runs'/name,dict(freeze_hash=digest(frozen),phase=name))
        for key in keys:store.put(key,{'saved':True})
        atomic(store.path/'complete.json',dict(complete=True))
    for method in ('BM25','MiniLM','SFR'):
        surface(root,'R0-'+method,['q']);surface(tiny,'R0-'+method,['q'])
        for interface in ('full','sara'):
            surface(root,f'R-{method}-{interface}',[['q',k] for k in cfg['k_values']])
            surface(tiny,f'R-{method}-{interface}',[['q',k] for k in (2,8)])
    ckeys=[['q',v] for v in ('no_evidence','full','sara_k4','lead256','qcs256')]
    surface(root,'C-Mistral',ckeys[:2] if partial_c else ckeys)
    if partial_c:(root/'runs/C-Mistral/complete.json').unlink()
    calls=[]; generated=[]
    required=('summary.csv','bootstrap.csv','oracle_diagnostics.json','retrieval_summary.csv','decision_R.json')
    def stage(phase,*args,execute=False):
        calls.append((phase,args,execute))
        if phase=='RCA':
            for name in required:(root/name).write_text('mock report\n')
        elif phase=='C':
            store=Store(root/'runs/C-Mistral',dict(freeze_hash=digest(frozen),phase='C-Mistral'))
            for key in ckeys:
                if store.get(key) is None:store.put(key,{'saved':True});generated.append(key)
            atomic(store.path/'complete.json',dict(complete=True))
        elif phase not in ('audit','status','smoke'):
            raise AssertionError('Unexpected generation phase: '+phase)
    def no_subprocess(*a,**kw):raise AssertionError('No actual subprocess permitted in scheduling tests')
    ns=dict(ROOT=tmp_path,RESULT_ROOT=root,cfg=cfg,PYTHON=tmp_path/'python',CONFIG=tmp_path/'config.json',
        SMOKE_ONLY=False,ACTIVE_READERS=('Mistral','Qwen'),Path=Path,json=json,subprocess=SimpleNamespace(run=no_subprocess),stage=stage)
    n=json.loads(NOTEBOOK.read_text())
    code_cells=[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    # Skip notebook import/setup; execute the actual helper and stage cells.
    with contextlib.redirect_stdout(io.StringIO()):exec(code_cells[1],ns)
    # The GPU/download setup is separately tested; default scheduler fixtures leave it pending.
    ns['prepare_reader_panel']=lambda:False
    def partial(stage):
        if stage=='D':return ns['stage']('D',execute=True)
        if stage=='complete-G':atomic(root/'partial_panel_status.json',dict(full_panel_complete=False,available_panel_complete=True,pending_readers=['Llama']))
    ns['partial_stage']=partial
    def run_all():
        with contextlib.redirect_stdout(io.StringIO()):
            for source in code_cells[2:]:exec(source,ns)
    return run_all,calls,generated,ns


def test_runall_skips_completed_r_c_and_stops_at_unreviewed_g(tmp_path,monkeypatch):
    run_all,calls,generated,ns=make_workflow(tmp_path,monkeypatch)
    run_all();run_all()
    assert generated==[]
    assert [c[0] for c in calls].count('RCA')==1
    assert not any(c[0] in ('R0','R','C','G','D') for c in calls)
    assert ns['RUN_READER_FULL'] is False and ns['RUN_DATASET_ANALYSIS_FULL'] is False


def test_runall_resumes_partial_c_without_repeating_saved_answers(tmp_path,monkeypatch):
    run_all,calls,generated,ns=make_workflow(tmp_path,monkeypatch,partial_c=True)
    run_all();run_all()
    assert len(generated)==3
    assert [c[0] for c in calls].count('C')==1
    assert not any(c[0] in ('R0','R','G','D') for c in calls)


def test_preview_starts_no_generation_or_reports(tmp_path,monkeypatch):
    run_all,calls,generated,ns=make_workflow(tmp_path,monkeypatch)
    ns['SMOKE_ONLY']=True;run_all()
    assert generated==[]
    assert all(not execute for _,_,execute in calls)
    assert all(phase in ('audit','smoke','status') for phase,_,_ in calls)


def test_all_notebook_code_cells_compile():
    n=json.loads(NOTEBOOK.read_text())
    for cell in n['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))


def test_authorized_remaining_workflow_runs_g_d_once(tmp_path,monkeypatch):
    run_all,calls,generated,ns=make_workflow(tmp_path,monkeypatch)
    ns['prepare_reader_panel']=lambda:True
    base_stage=ns['stage'];extra=[]
    def stage(phase,*args,execute=False):
        if phase=='RCA' and '--review-phase' in args:
            review=args[args.index('--review-phase')+1]
            atomic(ns['RESULT_ROOT']/f'freeze_{review}.json',dict(phase=review,GO_NEXT=True,retriever='SFR',
                reason='mock reviewed completion',execution_freeze_hash=ns['workflow_identity']()))
            return base_stage(phase,*args,execute=execute)
        if phase in ('G','D'):
            name='G-'+args[args.index('--reader')+1] if phase=='G' else 'D-stability'
            store=Store(ns['RESULT_ROOT']/'runs'/name,dict(freeze_hash=ns['workflow_identity'](),phase=name))
            for key in ns['expected_keys'](phase):
                if store.get(key) is None:store.put(key,{'saved':True});extra.append((name,key))
            atomic(store.path/'complete.json',dict(complete=True))
            calls.append((phase,args,execute));return
        return base_stage(phase,*args,execute=execute)
    ns['stage']=stage
    run_all();run_all()
    assert [phase for phase,_,_ in calls].count('G')==2
    assert [phase for phase,_,_ in calls].count('D')==1
    assert len(extra)==12  # 2 available readers x3 arms +3 paired seeds x2 actions
    assert ns['phase_approved']('C') and not ns['phase_approved']('G')
    assert (ns['RESULT_ROOT']/'partial_panel_status.json').exists()
    assert ns['RUN_POLICY_EXPERIMENT'] is False


def test_reader_setup_failure_starts_no_g_or_review(tmp_path,monkeypatch):
    import pytest
    run_all,calls,generated,ns=make_workflow(tmp_path,monkeypatch)
    def blocked():raise RuntimeError('Insufficient disk')
    ns['prepare_reader_panel']=blocked
    with pytest.raises(RuntimeError,match='Insufficient disk'):run_all()
    assert not (ns['RESULT_ROOT']/'freeze_C.json').exists()
    assert not any(phase in ('G','D') for phase,_,_ in calls)
