"""H/I consume SAVED synthetic rows only. No model calls or new data."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.week11.protocol import load
from src.week11.runner import execute
from src.week11.cache import atomic
from src.week11.catalog import PLOTS


def main():
    cfg=load()
    cfg['history_trajectory_records']=[dict(week=w,protocol='FABRICATED '+w+' separate protocol',metric='synthetic_fixture_score',value=.1+i*.05)
        for i,w in enumerate(['W3','W6','W8','W9','W10','W11'])]
    results=[]
    for phase in ['H','I']:
        tables,figures=execute(phase,cfg,smoke=True,limit=3)
        names={f['name'] for f in figures};expected={f[0] for f in PLOTS[phase]}
        missing=sorted(expected-names)
        if missing:raise ValueError('Missing fabricated figure coverage: '+str(missing))
        results.append(dict(phase=phase,synthetic=True,figures=len(figures),figure_types=len(names),tables=len(tables),
                            all_figures_present=True,model_calls=0,full=False))
    atomic(ROOT/cfg['output_root']/'smoke/analysis_plot_validation.json',results)
    print(results)


if __name__=='__main__':main()
