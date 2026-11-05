"""Explicit manual V2 stages; importing launches nothing."""
from pathlib import Path
import sys,argparse,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['audit','train','smoke','freeze','prepare','generate','controls','analyze','pilot','unseen-plan']);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    from src.post_w12.bea_v2.protocol import load,freeze
    from src.post_w12.bea_v2.data import audit
    c=load()
    if a.stage=='audit':audit(c);return
    if not a.execute:p.error('Use --execute for this stage; the agent must not launch pilot/prepare/generate')
    from src.post_w12.bea_v2.models import train
    from src.post_w12.bea_v2.stages import calibrate,prepare,generate
    from src.post_w12.bea_v2.baselines import restore_controls
    from src.post_w12.bea_v2.analysis import analyze
    if a.stage in ['train','smoke','pilot']:
        subprocess.run([str(ROOT/'sara_env/bin/python'),'-m','pytest','W12-POST/bea_v2_tests','-q'],cwd=ROOT,check=True)
        train(c)
    if a.stage in ['smoke','pilot']:print(json.dumps(calibrate(c),indent=2))
    if a.stage=='freeze':freeze(c)
    if a.stage in ['prepare','pilot']:prepare(c)
    if a.stage in ['generate','pilot']:generate(c)
    if a.stage in ['controls','pilot']:restore_controls(c)
    if a.stage in ['analyze','pilot']:analyze(c)
    if a.stage=='unseen-plan':
        from src.post_w12.bea_v2.unseen import plan
        print(json.dumps(plan(c),indent=2))

if __name__=='__main__':main()
