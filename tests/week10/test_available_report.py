import csv,importlib.util,json
from pathlib import Path
from src.week10.root_cause.storage import Store,atomic
P=Path(__file__).resolve().parents[2]/'scripts/week10/report_available.py'
s=importlib.util.spec_from_file_location('available_report',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_deferred_llama_has_blank_scores_not_zero(tmp_path,monkeypatch):
    cfg=dict(output_root=str(tmp_path),reader_panel=['Mistral','Qwen','Llama'])
    atomic(tmp_path/'partial_panel_status.json',dict(available_panel_complete=True))
    atomic(tmp_path/'runs/D-stability/complete.json',dict(done=True))
    atomic(tmp_path/'dataset_diagnostics.json',dict(stability=[dict(agreement=1.,sign_stable=True,delta_variance=0.)],
        em_stability=[dict(agreement=1.)],oracle_by_seed={}))
    for reader in ['Mistral','Qwen']:
        store=Store(tmp_path/'runs'/('G-'+reader),{'synthetic':True})
        for action,score in [('no_evidence',.2),('full',.4),('qcs256',.3)]:
            store.put(['q',action],dict(question_id='q',paper_id='p',action=action,f1=score,exact_match=0.,generation=dict(prompt_tokens=100)))
    monkeypatch.setattr(m,'paired',lambda *a,**kw:dict(point_estimate=.1,ci95=[0,.2]))
    m.report(cfg)
    rows=list(csv.DictReader((tmp_path/'reader_panel_summary.csv').open()))
    pending=[r for r in rows if r['reader']=='Llama']
    assert len(pending)==3 and all(r['f1']=='' and r['HELP']=='' and r['status']=='PENDING_ACCESS' for r in pending)
    assert not json.loads((tmp_path/'available_study_summary.json').read_text())['full_reader_panel_complete']
