"""Real frozen DEV benchmark; TEST remains disabled."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.week8.final_dev import run_w8j,OUT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--smoke',action='store_true');mode.add_argument('--full',action='store_true')
    p.add_argument('--device',default='cuda');p.add_argument('--output-dir',type=Path,default=OUT)
    a=p.parse_args();run_w8j(a.output_dir,smoke=a.smoke,include_w8g=True,device=a.device,check=a.check)
if __name__=='__main__':main()
