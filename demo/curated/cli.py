import argparse,os,subprocess,sys,json
from pathlib import Path
from .protocol import APP,ROOT,OUT,config,manifest,read

def run(stage,full=False):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    if stage=='smoke':
        python=APP/'.venv/bin/python'
        subprocess.run([str(python) if python.exists() else sys.executable,'-m','pytest',str(APP/'tests/test_curated.py'),'-q'],cwd=ROOT,check=True);return
    c=config();m=manifest(c)
    if stage=='plan':
        from .publication import status
        out=status(m);out.update(queries_per_dataset=10,model=c['models']['Qwen'],methods=m['methods'],retrievers=m['retrievers'],budgets=m['budgets'],scope=m['scope'])
        print(json.dumps(out,indent=2));return
    if not full:raise ValueError('Real demo preparation/generation requires explicit --full')
    from src.week12.protocol import lock
    with lock(OUT/'run.lock'):
        if stage=='prepare':
            from .preparation import run as prepare
            prepare(c,m)
        elif stage in ('answers','attribution'):
            from .generation import run as generate
            generate(c,m,attribution=stage=='attribution')
        elif stage=='publish':
            from .publication import publish
            print(publish(m))
        else:raise ValueError('Unknown stage')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','smoke','prepare','answers','attribution','publish']);parser.add_argument('--full',action='store_true');a=parser.parse_args();run(a.stage,a.full)
if __name__=='__main__':main()
