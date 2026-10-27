"""Explicit user-authorized partial panel; never mark Llama/full G complete.

Scientific runner/model/config stay frozen. The sole gate exception permits
prespecified Mistral-only D after available G readers, with an auditable waiver.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.week10.root_cause.storage import Store,atomic,digest,file_hash
from src.week10.root_cause.protocol import load,source_hashes

ACTIVE=('Mistral','Qwen')
DEFERRED='Llama'
AUTHORIZATION='User explicitly authorized executing the notebook without Llama, marking it pending until access is granted; complete the available study first.'


def state(cfg,require_complete=False):
    root=ROOT/cfg['output_root'];frozen=json.loads((root/'execution_freeze.json').read_text())
    if frozen['config']!=cfg or frozen['sources']!=source_hashes():raise ValueError('Frozen method changed')
    if tuple(cfg['reader_panel'])!=('Mistral','Qwen','Llama'):raise ValueError('Unexpected frozen panel')
    qids=[str(json.loads(line)['id']) for line in Path(cfg['dataset']).read_text().splitlines() if line.strip()]
    if cfg.get('query_limit'):qids=qids[:cfg['query_limit']]
    coverage={};completed=[]
    for reader in ACTIVE:
        path=root/'runs'/('G-'+reader);p=path/'identity.json'
        count=0
        if p.exists():
            identity=json.loads(p.read_text())
            if identity['freeze_hash']!=digest(frozen):raise ValueError('Wrong G identity')
            store=Store(path,identity)
            count=sum(store.get([q,a]) is not None for q in qids for a in ('no_evidence','full','qcs256'))
        good=count==len(qids)*3 and (path/'complete.json').exists()
        coverage[reader]=dict(saved=count,expected=len(qids)*3,complete=good)
        if good:completed.append(reader)
    if require_complete and len(completed)!=len(ACTIVE):raise RuntimeError('Available G panel is not complete')
    return dict(execution_freeze_hash=digest(frozen),full_panel_complete=False,
        available_panel_complete=len(completed)==len(ACTIVE),active_readers=list(ACTIVE),coverage=coverage,
        pending_readers={DEFERRED:dict(status='PENDING_ACCESS',model=cfg['models'][DEFERRED],
            reason='GatedRepoError; explicitly deferred by user. No placeholder predictions or scores.')},
        user_authorization=AUTHORIZATION,policy_training_authorized=False,qasper_test_used=False,
        gate_exception='D may run after available G panel; original full-panel freeze_G.json is not created or replaced.')


def record(cfg,complete=False):
    obj=state(cfg,complete);root=ROOT/cfg['output_root']
    atomic(root/'partial_panel_status.json',obj)
    return obj


def run_d(cfg):
    obj=record(cfg,True)
    import src.week10.root_cause.runner as runner
    from src.week10.root_cause.dataset_audit import run
    original=runner.phase_gate
    def authorized_gate(actual,phase):
        if actual!=cfg or phase!='D':return original(actual,phase)
        state(cfg,True)
        prior=original(actual,'G')  # C must remain reviewed and frozen.
        return dict(prior,partial_panel=True,deferred_readers=[DEFERRED])
    # Scope this exception to the prespecified D invocation, never the full study.
    with patch.object(runner,'phase_gate',authorized_gate):run(cfg)
    atomic(ROOT/cfg['output_root']/'D_partial_panel_provenance.json',dict(obj,
        dataset_audit_sha256=file_hash(ROOT/'src/week10/root_cause/dataset_audit.py'),
        diagnostics_sha256=file_hash(ROOT/cfg['output_root']/'dataset_diagnostics.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=['status','complete-G','D'],required=True);a=p.parse_args();cfg=load(a.config)
    if a.stage=='D':return run_d(cfg)
    print(json.dumps(record(cfg,a.stage=='complete-G'),indent=2))

if __name__=='__main__':main()
