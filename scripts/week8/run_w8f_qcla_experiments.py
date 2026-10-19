"""Run measured QCLA experiments; no synthetic results or zero feature fallback."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.week8.qcla_experiment import run_experiment,DEFAULT_OUT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true',help='Verify data, canonical routing, local checkpoint and split integrity')
    mode.add_argument('--prepare',action='store_true',help='Measure and cache SFR query/draft embeddings only')
    mode.add_argument('--smoke',action='store_true',help='Real 2 FIT / 2 validation query training and generation for all three variants')
    mode.add_argument('--full',action='store_true',help='Full fixed-budget FIT training and development confirmation')
    p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    a=p.parse_args()
    run_experiment(a.output_dir,smoke=a.smoke,device=a.device,epochs=a.epochs,prepare_only=a.prepare,check=a.check)
if __name__=='__main__':main()
