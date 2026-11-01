"""Execute ONLY safe notebook defaults, with synthetic fixtures or saved views."""
import argparse
import ast
import json
from pathlib import Path
import sys
import time
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.week11.catalog import NOTEBOOKS
from src.week11.cache import atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--phases',nargs='*');a=p.parse_args()
    items=list(NOTEBOOKS.items())+[('MASTER','week11_MASTER.ipynb')]
    if a.phases:items=[(k,v) for k,v in items if k in a.phases]
    records=[];start=time.perf_counter()
    for phase,name in items:
        path=ROOT/'notebooks/week11'/name;notebook=nbformat.read(path,as_version=4)
        flags=next(c for c in notebook.cells if c.cell_type=='code')
        assignments={node.targets[0].id:node.value.value for node in ast.parse(flags.source).body if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant)}
        if assignments.get('RUN_FULL') is not False or assignments.get('REVIEW_PHASE') is not None or assignments.get('OVERWRITE') is not False:
            raise RuntimeError('Refuse notebook with full/overwrite/review enabled: '+name)
        before=time.perf_counter();print('SAFE NOTEBOOK',phase,'START',flush=True)
        try:
            NotebookClient(notebook,timeout=600,kernel_name='sara_env',resources={'metadata':{'path':str(ROOT)}}).execute()
            nbformat.write(notebook,path)
            records.append(dict(phase=phase,status='PASS',seconds=time.perf_counter()-before,full=False))
        except Exception as exc:
            nbformat.write(notebook,path)
            records.append(dict(phase=phase,status='FAIL',error_type=type(exc).__name__,seconds=time.perf_counter()-before,full=False))
            atomic(ROOT/'results/week11/multigranular/v1/smoke/notebook_validation.json',records)
            raise
        print('SAFE NOTEBOOK',phase,'PASS',flush=True)
    path=ROOT/'results/week11/multigranular/v1/smoke/notebook_validation.json'
    if a.phases and path.exists():
        old=json.loads(path.read_text());phases={r['phase'] for r in records}
        records=[r for r in old if r['phase'] not in phases]+records
    atomic(path,records)
    print('Only safe/synthetic notebooks executed; seconds=',round(time.perf_counter()-start),flush=True)


if __name__=='__main__':main()
