"""Real frozen-policy DAFiR-1 with matched G0/G1 and optional fixed W8-G revisions."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.week8.full_system import run_w8h,OUT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--smoke',action='store_true');mode.add_argument('--full',action='store_true')
    p.add_argument('--policy',choices=['dafir1'],default='dafir1',help='DAFiR-2 has no frozen state-specific controller and is unsupported')
    p.add_argument('--include-w8g',action='store_true',help='Report all three completed W8-G revision adapters separately; no validation winner selection')
    p.add_argument('--device',default='cuda');p.add_argument('--output-dir',type=Path,default=OUT)
    a=p.parse_args();run_w8h(a.output_dir,smoke=a.smoke,include_w8g=a.include_w8g,device=a.device,check=a.check)
if __name__=='__main__':main()
