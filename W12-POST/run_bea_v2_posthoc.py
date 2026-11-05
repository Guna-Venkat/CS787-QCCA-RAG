"""Separate manual BEA_V2 post-hoc entry point. No calibration or fitting stages."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['status','prepare','generate','analyze','all'])
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--allow-missing-control-restore',action='store_true')
    args=parser.parse_args()
    if args.stage!='status' and not args.execute:
        parser.error('Explicit --execute is required; default mode never generates answers')
    from src.post_w12.bea_v2_posthoc.runner import preflight,prepare,generate
    if args.stage=='status':
        print(json.dumps(preflight(),indent=2));return
    if args.stage in ('prepare','all'):
        prepare(allow_missing_control_restore=args.allow_missing_control_restore)
    if args.stage in ('generate','all'):
        generate(explicit=True,allow_missing_control_restore=args.allow_missing_control_restore)
    if args.stage in ('analyze','all'):
        from src.post_w12.bea_v2_posthoc.analysis import analyze
        analyze()

if __name__=='__main__':main()
