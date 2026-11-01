"""Report every planned reader, with genuinely empty scores for deferred Llama."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.week10.root_cause.storage import Store,atomic
from src.week10.root_cause.protocol import load
from src.week10.root_cause.analysis import paired


def rows_for(root,reader):
    path=root/'runs'/('G-'+reader)
    if not (path/'identity.json').exists():return []
    return Store(path,json.loads((path/'identity.json').read_text())).rows()


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def report(cfg):
    root=ROOT/cfg['output_root'];status=json.loads((root/'partial_panel_status.json').read_text())
    if not status['available_panel_complete'] or not (root/'runs/D-stability/complete.json').exists():
        raise RuntimeError('Finish the available reader surfaces and D first')
    table=[];groups={};comparisons=[]
    for reader in cfg['reader_panel']:
        rows=rows_for(root,reader);by={}
        for row in rows:by.setdefault(row['action'],[]).append(row)
        base={r['question_id']:r for r in by.get('no_evidence',[])}
        for action in ('no_evidence','full','qcs256'):
            current=by.get(action,[]);groups[reader,action]=current
            table.append(dict(reader=reader,action=action,status='COMPLETE' if current else 'PENDING_ACCESS',
                n_queries=len(current),f1=float(np.mean([r['f1'] for r in current])) if current else None,
                exact_match=float(np.mean([r['exact_match'] for r in current])) if current else None,
                net_evidence_utility=float(np.mean([r['f1']-base[r['question_id']]['f1'] for r in current])) if current else None,
                HELP=float(np.mean([r['f1']-base[r['question_id']]['f1']>.01 for r in current])) if current else None,
                HURT=float(np.mean([r['f1']-base[r['question_id']]['f1']<-.01 for r in current])) if current else None,
                final_prompt_tokens=float(np.mean([r['generation']['prompt_tokens'] for r in current])) if current else None))
    for reader in ('Mistral','Qwen'):
        for action in ('full','qcs256'):
            result=paired(groups[reader,action],groups[reader,'no_evidence'])
            comparisons.append(dict(comparison=f'{reader}:{action}-no_evidence',point=result['point_estimate'],
                ci_low=result['ci95'][0],ci_high=result['ci95'][1]))
    for action in ('no_evidence','full','qcs256'):
        result=paired(groups['Qwen',action],groups['Mistral',action])
        comparisons.append(dict(comparison=f'Qwen-Mistral:{action}',point=result['point_estimate'],
            ci_low=result['ci95'][0],ci_high=result['ci95'][1]))
    for action in ('full','qcs256'):
        deltas=[]
        for reader in ('Qwen','Mistral'):
            base={r['question_id']:r['f1'] for r in groups[reader,'no_evidence']}
            deltas.append([dict(question_id=r['question_id'],paper_id=r['paper_id'],f1=r['f1']-base[r['question_id']]) for r in groups[reader,action]])
        result=paired(*deltas)
        comparisons.append(dict(comparison=f'(Qwen evidence utility)-(Mistral evidence utility):{action}',point=result['point_estimate'],
            ci_low=result['ci95'][0],ci_high=result['ci95'][1]))
    diagnostics=json.loads((root/'dataset_diagnostics.json').read_text())
    d=dict(n_queries=len(diagnostics['stability']),
        mean_label_agreement=float(np.mean([r['agreement'] for r in diagnostics['stability']])),
        all_seeds_same_label_fraction=float(np.mean([r['sign_stable'] for r in diagnostics['stability']])),
        mean_delta_variance=float(np.mean([r['delta_variance'] for r in diagnostics['stability']])),
        mean_EM_label_agreement=float(np.mean([r['agreement'] for r in diagnostics['em_stability']])),
        oracle_by_seed=diagnostics['oracle_by_seed'],note='Mistral-only sampling diagnostic; primary systems use greedy decoding')
    write_csv(root/'reader_panel_summary.csv',table);write_csv(root/'available_reader_bootstrap.csv',comparisons)
    atomic(root/'available_study_summary.json',dict(available_phases_complete=True,full_reader_panel_complete=False,
        pending_readers=['Llama'],reader_results=table,comparisons=comparisons,dataset_stability=d,
        root_cause_scope='Development diagnostics on available Mistral/Qwen readers only; no full-panel or confirmation claim',
        policy_training_started=False,qasper_test_used=False))
    lines=['# Week 10 — Available-reader results','',
        'Available Mistral/Qwen reader experiments and the prespecified Mistral seed diagnostic are complete. Llama remains pending access; its scores below are blank, not zero. The original full-panel study is not complete.','',
        '| Reader | Evidence | Status | F1 | Net evidence utility | HELP | HURT |',
        '|---|---|---|---:|---:|---:|---:|']
    fmt=lambda x:'' if x is None else f'{x:.5f}'
    for r in table:lines.append(f"| {r['reader']} | {r['action']} | {r['status']} | {fmt(r['f1'])} | {fmt(r['net_evidence_utility'])} | {fmt(r['HELP'])} | {fmt(r['HURT'])} |")
    lines+=['','## Paired paper-cluster bootstrap','', '| Comparison | Delta F1 | 95% CI |','|---|---:|---|']
    for r in comparisons:lines.append(f"| {r['comparison']} | {r['point']:.5f} | [{r['ci_low']:.5f}, {r['ci_high']:.5f}] |")
    lines+=['','## Dataset/action-label stability','',
        f"Mistral subset: {d['n_queries']} queries, seeds42/43/44. Mean modal HELP/NEUTRAL/HURT agreement: {d['mean_label_agreement']:.3f}; fraction identical across all seeds: {d['all_seeds_same_label_fraction']:.3f}.",
        '', 'No LLM judge or semantic-correctness estimate was invented. These comparisons are development diagnostics with unadjusted exploratory CIs. Stronger-reader effects must be assessed on identical evidence within G, not by directly comparing public G Mistral to the separately adapted C SARA checkpoint.',
        '', '## Remaining work','', 'Llama access, qualification and three G arms remain pending. Faithful SeleCom, a learned textual compressor and fresh external confirmation are separate uncompleted extensions. No policy/controller or TEST run was started.']
    (root/'W10_AVAILABLE_PANEL_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(reader_results=table,dataset_stability=d),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();report(load(a.config))
