"""Manual Week13 only. Defaults inspect readiness; never starts evaluation."""
from pathlib import Path
import sys,argparse,subprocess,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
from src.week12.protocol import load
from src.week12.confirmation import final_freeze,run

def main():
 p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');p.add_argument('--explicit',action='store_true');p.add_argument('--stage',choices=['prepare','generate']);args=p.parse_args();cfg=load()
 if not args.full:
  print('READINESS ONLY: no confirmation execution requested.')
  try:f,split,h=final_freeze(cfg);print('Frozen confirmation counts:',{d:len(v['CONFIRM']) for d,v in split['datasets'].items()})
  except (RuntimeError,ValueError) as e:print(str(e))
  return
 if not args.explicit:raise ValueError('Week13 confirmation requires manual --full --explicit after final freeze')
 final_freeze(cfg)
 if args.stage:run(cfg,args.stage)
 else:
  for stage in ['prepare','generate']:subprocess.run([sys.executable,str(Path(__file__)),'--full','--explicit','--stage',stage],cwd=ROOT,check=True)
if __name__=='__main__':main()
