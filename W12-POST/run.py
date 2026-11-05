"""Manual stages. This module does nothing on import."""
from pathlib import Path
import sys,argparse,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['audit','calibrate','prepare','generate','analyze','pilot','expanded','attribution'])
    parser.add_argument('--execute',action='store_true',help='Explicit opt-in to real preparation/generation')
    parser.add_argument('--config',type=Path,default=ROOT/'W12-POST/config.json')
    args=parser.parse_args()
    from src.post_w12.bea_qbgf.protocol import load,freeze,expanded,read,output
    from src.post_w12.bea_qbgf.runner import plan,prepare,generate
    from src.post_w12.bea_qbgf.analysis import analyze
    c=load(args.config)
    if args.stage=='audit':print(json.dumps(plan(c),indent=2));return
    if args.stage=='analyze':analyze(c);return
    if not args.execute:parser.error('Real work requires explicit --execute; only audit is read-only/model-free')
    if args.stage=='calibrate':
        from src.post_w12.bea_qbgf.calibration import run
        print(json.dumps(run(c),indent=2));return
    if args.stage=='pilot':
        from src.post_w12.bea_qbgf.calibration import run
        run(c)
    if args.stage=='attribution':
        from src.post_w12.bea_qbgf.attribution import run
        run(c);return
    stage='expanded' if args.stage=='expanded' else 'pilot'
    if stage=='expanded':
        # Recompute decision from every paired pilot row before allowing Stage 2.
        result=analyze(c);parent,s,cohort,f=freeze(c);expanded(c,s,cohort,dict(result['decision']))
    p=plan(c,stage);run_missing=p['automatic_run_missing'] and c['cache']['run_missing']
    if args.stage in ['prepare','pilot','expanded']:prepare(c,stage,run_missing)
    if args.stage in ['generate','pilot','expanded']:generate(c,stage,run_missing)
    if args.stage in ['pilot','expanded']:analyze(c,stage)

if __name__=='__main__':main()
