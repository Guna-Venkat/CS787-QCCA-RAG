from pathlib import Path
import sys,argparse,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1');os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
from src.week12.protocol import load,authorize,require_full
from src.week12.runner import execute,prepare,generate

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=list('0ABCDEFGHI'),required=True);p.add_argument('--full',action='store_true');p.add_argument('--explicit',action='store_true');p.add_argument('--real-smoke',action='store_true');p.add_argument('--limit',type=int,default=3);p.add_argument('--internal-stage',choices=['prepare','generate']);p.add_argument('--canonical',choices=['SARA_CANONICAL_BM25_K5_N10','CANONICAL_RAG_BM25']);p.add_argument('--train',action='store_true');p.add_argument('--prepare-corpus',action='store_true');p.add_argument('--include-secondary',action='store_true',help='Explicit opt-in to secondary C/D surfaces; CORE_FINAL defaults True');args=p.parse_args();cfg=load();authorize(args.full,not args.full,args.explicit,args.limit)
 if args.full:require_full(cfg,args.phase)
 if args.canonical:
  from src.week12.runner import examples
  from src.week12.canonical import prepare as cp,generate as cg
  import copy
  xs=examples(cfg,limit=3 if not args.full else None);cfg=copy.deepcopy(cfg)
  if not args.full:cfg['output_root']+='/smoke/canonical/'+__import__('src.week12.protocol',fromlist=['code_hash']).code_hash()[:12]
  if args.internal_stage=='prepare':cp(cfg,xs,args.canonical)
  elif args.internal_stage=='generate':print(cg(cfg,xs,args.canonical))
  else:
   import subprocess,json
   from src.week12.protocol import root,atomic,identity
   canonical_receipt=ROOT/load()['output_root']/'qualification'/(args.canonical+'.json')
   if args.full:
    if not canonical_receipt.exists() or json.loads(canonical_receipt.read_text()).get('status')!='PASS' or json.loads(canonical_receipt.read_text()).get('identity')!=identity(load()):raise RuntimeError('Qualify the canonical specialist smoke in notebook B first')
   cmd=[str(ROOT/'sara_env/bin/python'),str(Path(__file__)),'--phase',args.phase,'--canonical',args.canonical]+(['--full','--explicit'] if args.full else [])
   for stage in ['prepare','generate']:subprocess.run(cmd+['--internal-stage',stage],check=True,cwd=ROOT)
   if not args.full:atomic(canonical_receipt,dict(status='PASS',system=args.canonical,real_queries=1,split='development',identity=identity(load()),canonical_k=5 if args.canonical.startswith('SARA') else 10,n=10,max_new_tokens=64))
  return
 if args.prepare_corpus:
  if not args.full:raise ValueError('Real training-corpus preparation is manual full opt-in')
  from src.week12.lora import prepare_corpus
  prepare_corpus(cfg);return
 if args.train:
  from src.week12.lora import train
  print(train(cfg,explicit=args.full and args.explicit));return
 if args.internal_stage:
  if not args.full and not args.real_smoke:raise ValueError('Internal real execution needs bounded real-smoke or manual full flags')
  if args.internal_stage=='prepare':prepare(cfg,args.phase,False,args.real_smoke,args.limit,not args.include_secondary)
  else:generate(cfg,args.phase,False,args.real_smoke,not args.include_secondary)
 else:
  result=execute(args.phase,cfg,full=args.full,smoke=not args.full,explicit=args.explicit,limit=args.limit,real_smoke=args.real_smoke,core_final=not args.include_secondary);print('Returned',len(result),'rows/items; completed shards reused.')
if __name__=='__main__':main()
