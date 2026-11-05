"""Explicit manual stages; no full run without --execute."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'W12-POST'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['status','smoke','prepare','generate','analyze','full'])
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--allow-model-download',action='store_true')
    parser.add_argument('--reader-smoke',action='store_true')
    args=parser.parse_args()
    if args.stage!='status' and not args.execute:parser.error('Use explicit --execute')
    if args.reader_smoke and args.stage!='smoke':parser.error('--reader-smoke only applies to the bounded 3-query smoke')
    from reranker_baseline.runner import preflight,smoke,prepare_full,generate_full
    if args.stage=='status':print(json.dumps(preflight(),indent=2));return
    if args.stage=='smoke':
        r=smoke(allow_download=args.allow_model_download,reader=args.reader_smoke)
        print(json.dumps({k:v for k,v in r.items() if k!='examples'},indent=2));return
    if args.stage in ['prepare','full']:prepare_full(explicit=True,allow_download=args.allow_model_download)
    if args.stage in ['generate','full']:generate_full(explicit=True)
    if args.stage in ['analyze','full']:
        from reranker_baseline.analysis import analyze
        r=analyze();print(r['final_table'].to_markdown(index=False));print(r['report'])

if __name__=='__main__':main()
