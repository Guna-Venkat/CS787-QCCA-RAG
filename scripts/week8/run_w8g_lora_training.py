"""Measured W8-G uniform/HC/OW LoRA with no mocks or validation-selected gamma."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.week8.outcome_weighted_training import run_w8g,OUT

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--smoke',action='store_true',help='Real A/B/C FIT cases and STOP/PROMOTE validation cases')
    mode.add_argument('--full',action='store_true')
    parser.add_argument('--device',default='cuda');parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    run_w8g(args.output_dir,smoke=args.smoke,device=args.device,epochs=args.epochs,check=args.check)
if __name__=='__main__':main()
