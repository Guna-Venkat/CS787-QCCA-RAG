"""Durable authorized continuation: notebooks, mechanical audits, recorded reviews.

No automatic finalist selection: F uses the five manually reviewed saved C IDs.
Stops on errors, identity changes or insufficient disk. Never modifies old rows.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.week11.protocol import load, output, review, gate, code_identity, scientific_config
from src.week11.cache import atomic, digest, file_hash
from src.week11.planning import plan, pending_reason
from operations.week11.watch_progress import writer_active
from operations.week11.review_generation import audit

MONITOR = ROOT/'results/week11/monitoring'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def owner_active(phase):
    state = read(MONITOR/f'cell_state_{phase}.json')
    if state.get('status') != 'RUNNING':return False
    try:
        command=Path(f"/proc/{state['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()
        return 'operations/week11/run_phase.py' in command and f'--phase {phase}' in command
    except (OSError, KeyError):return False


def complete(cfg, phase):
    root=output(cfg,phase)
    return (read(root/'status.json').get('executable_complete',False)
        and (root/'REPORT.md').exists() and (root/'figures.json').exists()
        and not writer_active(root/'run.lock') and not owner_active(phase))


def progress(cfg,phase,state):
    path=ROOT/'results/week11/progress.json';obj=read(path)
    obj.update(updated_at=utc(),running_phase=phase,supervisor_state=state,
        supervisor='operations/week11/continue_study.py',monitor_interval_seconds=900)
    done=[]
    for row in obj.get('phases',[]):
        p=row['phase'];status=read(output(cfg,p)/'status.json')
        frozen=(ROOT/cfg['output_root']/f'freeze_{p}.json').exists()
        row.update(freeze_exists=frozen,number_completed=status.get('completed',0),number_expected=status.get('expected'))
        if status.get('executable_complete'):
            row['full_status']='COMPLETE_AVAILABLE' if status.get('pending') else 'COMPLETE'
            row['next_action']='Audited/frozen' if frozen else 'Mechanical audit and recorded continuation review'
        elif p==phase:row.update(full_status=state,next_action='Durable continuation; no duplicate launch needed')
        if frozen:done.append(p)
    obj['completed_phases']=done
    atomic(path,obj)
    lines=['# Week 11 progress',f'Updated {utc()}. Branch `gagan`. Frozen pilot: 50 development queries per dataset.',
        f'Current phase: **{phase or "all available phases complete"}**. State: **{state}**.',
        'A durable operational supervisor now runs the authorized notebook sequence. Every generation phase receives checksum/cohort/budget/baseline audits before a recorded mechanical continuation review. F uses the five manually reviewed actual C actions; the supervisor never selects new winners. Errors stop advancement. LLMLingua conditions remain explicitly pending.',
        '| Phase | State | Completed | Expected | Frozen |','|---|---|---:|---:|---|']
    lines += [f"| {r['phase']} | {r['full_status']} | {r['number_completed']} | {r.get('number_expected') or '—'} | {r['freeze_exists']} |" for r in obj.get('phases',[])]
    lines += ['Reports and row flags are in each phase `INTERPRETATION.md`, `readiness_audit.json` and `flagged_queries.csv`. Saved outputs and manually edited original notebooks are preserved. Executed copies remain local. The passive monitor records every 15 minutes; it is separate from this notebook supervisor. Detailed statistical conclusions still require reading the reported intervals and flags; retrospective best policies and oracles are not independent confirmation or deployable controllers.']
    (ROOT/'results/week11/PROGRESS.md').write_text('\n\n'.join(lines)+'\n')


def analysis_audit(cfg,phase):
    from src.week11.reporting import saved_rows
    rows=[r for r in saved_rows(ROOT/cfg['output_root']) if 'task_score' in r]
    root=output(cfg,phase);status=read(root/'status.json')
    assert status['generation_started'] is False
    assert digest(rows)==status['source_result_hash']
    assert len(rows)==status['completed']==status['expected']
    names=['oracle_summary_df.csv','oracle_action_distribution_df.csv']
    if phase=='I':names+=['master_summary_df.csv','best_by_dataset_df.csv']
    for name in names:assert (root/name).exists()
    result=dict(phase=phase,timestamp_utc=utc(),checks_passed=True,rows_verified=len(rows),
        generation_started=False,source_result_hash=status['source_result_hash'],
        audit_driver_sha256=file_hash(Path(__file__)),status_sha256=file_hash(root/'status.json'))
    atomic(root/'readiness_audit.json',result)
    import pandas as pd
    oracle=pd.read_csv(root/'oracle_summary_df.csv')
    text=[f'# Week 11 {phase} — saved-result analysis',
        f'Verified {len(rows):,} source rows by their immutable store checksums and aggregate status hash. No generation started in this phase.',
        '## Retrospective oracle ceilings',oracle.to_markdown(index=False,floatfmt='.6f'),
        'These ceilings select actions using observed outcomes on the same development pilot. They do not train or evaluate a deployable controller. Exact/epsilon choice is cost-aware only within the retrospective action surface. SciFact uses exact classification utility; QA uses epsilon 0.01.',
        '## Interpretation limits',
        'Use the C/D/E/F/G phase interpretations for controlled comparisons and flagged rows. Do not pool raw QA F1 and verification accuracy. ALL_READERS includes heterogeneous checkpoints and is a retrospective ceiling; SARA-specialist results stay separate from portable public-reader conclusions. C and D share cached Qwen observations; duplicate action/query records are deduplicated in aggregate analyses and are not independent replications. Marginal factorial summaries have unequal exposure and are descriptive, not causal effects. All point-estimate best policies were selected retrospectively on 50 development queries per dataset.',
        'LLMLingua token conditions remain pending and are not implied by completion of the available text study. Historical trajectory records remain unspecified rather than inventing comparable scores across protocols. A larger preregistered cohort/new output version is needed for confirmation.']
    if phase=='H':
        contrasts=pd.read_csv(root/'paired_interaction_contrasts_df.csv')
        text+=['## Explicit exploratory interaction contrasts',contrasts.to_markdown(index=False,floatfmt='.6f'),
            'The action order for each named contrast is recorded in the configuration. Twelve difference-in-differences contrasts were specified before D/E/F finished. Intervals are nominal, not corrected for multiplicity.']
    if phase=='I':
        best=pd.read_csv(root/'best_by_dataset_df.csv')
        text+=['## Retrospective best by dataset',best[['dataset','reader','reader_variant','retriever','evidence_granularity','selector','budget','compressor','task_score','prompt_tokens']].to_markdown(index=False,floatfmt='.6f'),
            '## Next research step',
            'Review the matched evidence-budget and reader/compressor contrasts before choosing a prospective policy. Freeze any candidate in a new confirmation protocol, retain no-evidence/raw-passage controls, and evaluate a larger untouched cohort. The present pilot cannot establish general RAG superiority or validate an adaptive controller.']
    (root/'INTERPRETATION.md').write_text('\n\n'.join(text)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--plan',action='store_true')
    args=parser.parse_args();cfg=load()
    if not args.execute:
        if not args.plan:raise ValueError('Use --execute for the authorized continuation or --plan for read-only inspection')
        for phase in ['D','E','F','G']:
            counts={d:sum(not pending_reason(a,cfg) for a in plan(cfg,phase,d)) for d in cfg['datasets']}
            print(phase,counts,'plus explicit no-evidence baselines where absent')
        print('Five manual F actions:',cfg['finalist_run_ids']);return
    MONITOR.mkdir(parents=True,exist_ok=True)
    with (MONITOR/'continuation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen_code=digest(code_identity())
        frozen_config={p:digest(scientific_config(cfg,p)) for p in ['D','E','F','G','H','I']}
        state=dict(pid=os.getpid(),status='RUNNING',started_utc=utc(),driver_sha256=file_hash(Path(__file__)),
            review_driver_sha256=file_hash(ROOT/'operations/week11/review_generation.py'),
            operational_dependencies={name:file_hash(ROOT/'operations/week11'/name) for name in ['run_phase.py','isolated_readers.py','label_compliance.py']},
            mode='User-authorized notebook sequence; mechanical audits; no automatic finalist selection',phases={})
        state_path=MONITOR/'continuation_state.json'
        def save(**changes):
            state.update(changes,updated_utc=utc());atomic(state_path,state)
        save()
        try:
            for phase in ['D','E','F','G','H','I']:
                cfg=load();assert digest(code_identity())==frozen_code
                assert digest(scientific_config(cfg,phase))==frozen_config[phase]
                assert file_hash(ROOT/'operations/week11/review_generation.py')==state['review_driver_sha256']
                assert all(file_hash(ROOT/'operations/week11'/name)==sha for name,sha in state['operational_dependencies'].items())
                save(current_phase=phase);progress(cfg,phase,'RUNNING')
                # Adopt the already-running authorized D job; never duplicate a writer.
                while owner_active(phase) or writer_active(output(cfg,phase)/'run.lock'):
                    time.sleep(30)
                if not complete(cfg,phase):
                    previous=read(MONITOR/f'cell_state_{phase}.json')
                    if previous.get('status')=='ERROR':raise RuntimeError(f'Phase {phase} notebook error requires inspection: '+str(previous.get('error_type')))
                    if shutil.disk_usage(ROOT).free < 10*2**30:raise RuntimeError('Less than 10 GiB free; no next phase launched')
                    for other in ['C','D','E','F','G','H','I']:
                        if other!=phase and writer_active(output(cfg,other)/'run.lock'):
                            raise RuntimeError('Another phase has an active writer: '+other)
                    gate(cfg,phase)
                    with (MONITOR/f'phase_{phase}.log').open('a') as log:
                        job=subprocess.Popen([str(ROOT/'sara_env/bin/python'),str(ROOT/'operations/week11/run_phase.py'),'--phase',phase,'--execute'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                        save(child_pid=job.pid)
                        while job.poll() is None:time.sleep(30)
                        if job.returncode:raise RuntimeError(f'Phase {phase} notebook failed with exit {job.returncode}; saved rows preserved')
                assert complete(cfg,phase)
                result=audit(phase) if phase in 'DEFG' else analysis_audit(cfg,phase)
                selected=dict(waive_pending=read(output(cfg,phase)/'status.json').get('pending',[]),
                    available_surface_only=True,mechanical_audit='readiness_audit.json',promote_single_winner=False,
                    continuation_mode='Predeclared operational review under user delegation; not an independent scientific confirmation')
                if phase=='D':selected.update(reuse_C_selection_verified=True,next_phase='E',reader_panel=cfg['readers'])
                if phase=='E':selected.update(next_phase='F',finalist_run_ids=cfg['finalist_run_ids'],manual_selection_receipt='results/week11/F_MANUAL_FINALIST_REVIEW.json')
                if phase=='F':selected.update(next_phase='G',retriever='SFR',reader='Qwen',reason='Preserve primary controls for prescribed hierarchy comparison; no outcome-driven retriever change')
                review(cfg,phase,selected,'Authorized durable continuation after mechanical audit passed: complete frozen cohort/action surface, checksums, identities, budgets and baseline pairs (or saved-source aggregate hash for H/I). Preserve row flags and exploratory statistical caveats in INTERPRETATION.md. Explicitly waive only pending optional components; no automatic winner selection or deployability claim.')
                state['phases'][phase]=dict(status='AUDITED_FROZEN',completed_utc=utc(),audit=result)
                save(child_pid=None);progress(cfg,phase,'AUDITED_FROZEN')
                print(utc(),phase,'AUDITED_FROZEN',flush=True)
            save(status='COMPLETE_AVAILABLE_STUDY',current_phase=None)
            progress(cfg,None,'COMPLETE_AVAILABLE_STUDY')
            print('All available W11 phases complete; optional token conditions remain pending.',flush=True)
        except BaseException as exc:
            save(status='STOPPED_ERROR',error_type=type(exc).__name__,error=str(exc))
            progress(cfg,state.get('current_phase'),'STOPPED_ERROR')
            raise


if __name__=='__main__':main()
