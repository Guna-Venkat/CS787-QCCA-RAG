import importlib.util,json
from pathlib import Path
import pytest
from src.week10.root_cause.storage import Store,atomic,digest
P=Path(__file__).resolve().parents[2]/'scripts/week10/available_panel.py'
s=importlib.util.spec_from_file_location('available_panel',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def fixture(tmp_path,monkeypatch,complete=True):
    data=tmp_path/'dev.jsonl';data.write_text(json.dumps(dict(id='q'))+'\n')
    cfg=dict(output_root=str(tmp_path/'out'),dataset=str(data),reader_panel=['Mistral','Qwen','Llama'],models={'Llama':{'id':'gated','revision':'sha'}})
    root=Path(cfg['output_root']);frozen=dict(config=cfg,sources={});atomic(root/'execution_freeze.json',frozen)
    monkeypatch.setattr(m,'source_hashes',lambda:{})
    for reader in m.ACTIVE:
        store=Store(root/'runs'/('G-'+reader),dict(freeze_hash=digest(frozen)))
        for action in ('no_evidence','full','qcs256'):
            if complete or action!='full':store.put(['q',action],{'saved':True})
        atomic(store.path/'complete.json',{'done':True})
    return cfg,root

def test_partial_status_never_fakes_llama_or_full_freeze(tmp_path,monkeypatch):
    cfg,root=fixture(tmp_path,monkeypatch)
    obj=m.record(cfg,True)
    assert obj['available_panel_complete'] and not obj['full_panel_complete']
    assert obj['pending_readers']['Llama']['status']=='PENDING_ACCESS'
    assert not (root/'freeze_G.json').exists() and not (root/'runs/G-Llama').exists()

def test_incomplete_available_panel_blocks_d(tmp_path,monkeypatch):
    cfg,root=fixture(tmp_path,monkeypatch,False)
    with pytest.raises(RuntimeError,match='not complete'):m.record(cfg,True)

def test_d_waiver_is_scoped_and_documented(tmp_path,monkeypatch):
    import src.week10.root_cause.runner as runner
    import src.week10.root_cause.dataset_audit as dataset
    cfg,root=fixture(tmp_path,monkeypatch)
    def gate(actual,phase):
        assert phase=='G';return dict(retriever='SFR',GO_NEXT=True)
    monkeypatch.setattr(runner,'phase_gate',gate)
    def run(actual):
        prior=runner.phase_gate(actual,'D')
        assert prior['partial_panel'] and prior['deferred_readers']==['Llama']
        atomic(root/'dataset_diagnostics.json',{'synthetic':True})
    monkeypatch.setattr(dataset,'run',run)
    m.run_d(cfg)
    assert runner.phase_gate is gate
    assert (root/'D_partial_panel_provenance.json').exists()
    assert not (root/'freeze_G.json').exists()
