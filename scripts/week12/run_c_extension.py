"""Manual C-EXT only. Default: deterministic synthetic smoke, no pretrained models."""
import argparse,sys,os,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1');os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
from src.week12.protocol import load,authorize,root,lock
from src.week12_extension import baselines as b

def main():
 p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');p.add_argument('--explicit',action='store_true');p.add_argument('--stage',choices=['prepare','generate']);p.add_argument('--specialist',choices=['SARA_CANONICAL_BM25_K5_N10','SARA_SFR_K5_N10']);p.add_argument('--qualify',action='store_true');a=p.parse_args();cfg=load();authorize(a.full,not a.full,a.explicit,3)
 if a.full:
  status=root(cfg)/'validation/C/status.json'
  if not status.exists() or not json.loads(status.read_text()).get('complete'):raise ValueError('Let frozen Part C finish before C-EXT full execution')
 if a.specialist:
  from src.week12_extension import specialists as s
  if not a.full and not a.qualify:raise ValueError('Specialist requires explicit --qualify one-query real smoke, or --full --explicit')
  if a.full:s.require_qualified(cfg,a.specialist)
  if a.stage:return getattr(s,a.stage)(cfg,a.specialist,smoke=not a.full)
 else:
  if a.stage:return getattr(b,a.stage)(cfg,synthetic=not a.full)
 cmd=[sys.executable,str(Path(__file__).resolve())]+(['--full','--explicit'] if a.full else [])+(['--specialist',a.specialist] if a.specialist else [])+(['--qualify'] if a.qualify else [])
 with lock(b.folder(cfg,not a.full and not a.specialist)/'run.lock'):
  for stage in ['prepare','generate']:subprocess.run(cmd+['--stage',stage],cwd=ROOT,check=True)
  if a.qualify:print(s.qualify(cfg,a.specialist))
  if not a.specialist:print(b.merged_analysis(cfg,synthetic=not a.full)['summary'])
if __name__=='__main__':main()
