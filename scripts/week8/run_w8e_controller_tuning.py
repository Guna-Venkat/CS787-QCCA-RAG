"""Real FIT-OOF controller construction; fail closed on missing measured inputs."""
from pathlib import Path
import sys,argparse,json
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.week8.selective_controller import run_controller,gate_inputs,OUT
from src.week8.closeout import cached_split


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--full',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=OUT)
    a=parser.parse_args()
    if a.check:
        close=ROOT/'results/week8/w8d_closeout/closeout_summary.json'
        if not close.is_file():raise FileNotFoundError('Finish the cached W8-D closeout first')
        for split in ['fit','internal_val']:
            df=cached_split(split);q=gate_inputs(df,split)
            print(f'{split}: {len(q)} measured gate inputs, {len(df)} canonical candidates')
        print('Ready: FIT-only tuning, no oracle, no mock fallback; DAFiR-1 only')
    else:
        if (a.output_dir/'controller_summary.json').exists():
            # Completed runs are inspected, not silently retuned/reselected.
            print('A completed frozen run exists. Inspect controller_summary.json and saved decisions; use a new directory only for an explicitly new experiment.')
            summary=run_controller(a.output_dir)
            print('Frozen selected policy:',summary['selected_policy'])
            return
        run_controller(a.output_dir)

if __name__=='__main__':main()
