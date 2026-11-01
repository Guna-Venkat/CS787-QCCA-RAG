import json
from pathlib import Path
import pandas as pd
from .protocol import ROOT,output,DIRECTORIES
from .catalog import NOTEBOOKS
from .planning import plan,pending_reason


def resources(cfg):
    result=[];n=cfg['characterization_queries_per_dataset']
    for phase in DIRECTORIES:
        for dataset in cfg['datasets']:
            actions=plan(cfg,phase,dataset)
            pending=sum(bool(pending_reason(a,cfg)) for a in actions)
            result.append(dict(phase=phase,dataset=dataset,configured_queries=n,
                planned_configurations=len(actions),executable_configurations=len(actions)-pending,pending_configurations=pending,
                answer_rows_upper_bound=n*len(actions) if phase in 'CDEFG' else 0,
                executable_answer_rows_upper_bound=n*(len(actions)-pending) if phase in 'CDEFG' else 0,
                new_lm_calls='<= answer rows; identical evidence + reader outputs reused; D reuses primary C reader',
                actual_time_estimate='NOT ESTIMATED; no full W11 timings measured'))
    return pd.DataFrame(result)


def status(cfg):
    rows=[]
    for phase,name in NOTEBOOKS.items():
        root=output(cfg,phase);p=root/'status.json';obj=json.loads(p.read_text()) if p.exists() else {}
        rows.append(dict(phase=phase,notebook=str(ROOT/'notebooks/week11'/name),
            full_status='COMPLETE_ELIGIBLE_PENDING_OPTIONAL' if obj.get('executable_complete') and obj.get('pending') else 'COMPLETE' if obj.get('executable_complete') else 'NOT_STARTED',
            number_expected=obj.get('expected'),number_completed=obj.get('completed',0),result_path=str(root),
            next_action='Review/freeze manually' if obj.get('executable_complete') else 'Open notebook; inspect safe smoke, then explicitly opt in',
            freeze_exists=(ROOT/cfg['output_root']/f'freeze_{phase}.json').exists()))
    return pd.DataFrame(rows)
