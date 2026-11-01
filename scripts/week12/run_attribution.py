"""J cached setup; K manual full only (default synthetic); L saved output analysis."""
import argparse,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1');os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
from src.week12.protocol import load,guard,lock
from src.week12_attribution.protocol import directory,freeze

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['setup','generate','analyze'],required=True);p.add_argument('--full',action='store_true');p.add_argument('--explicit',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args();cfg=load();guard()
 if a.full and (not a.explicit or a.smoke):raise ValueError('Manual full needs --full --explicit and no --smoke')
 if a.stage=='setup':
  if a.smoke:
   from src.week12_attribution.synthetic import prepare
   print('SYNTHETIC SETUP',len(prepare(cfg)[0]))
  else:
   from src.week12_attribution.experiment import prepare
   rows,cost=prepare(cfg);print('FROZEN REAL VAL SETUP; NO GENERATION',len(rows),cost)
 elif a.stage=='generate':
  from src.week12_attribution.experiment import execute
  if not a.full and not a.smoke:raise ValueError('Real perturbations require --full --explicit; otherwise use --smoke')
  with lock(directory(cfg,a.smoke)/'run.lock'):execute(cfg,full=a.full,smoke=a.smoke)
 else:
  from src.week12_attribution.analysis import analyze
  result=analyze(cfg,a.smoke);print(result['status'],result.get('folder'))
if __name__=='__main__':main()
