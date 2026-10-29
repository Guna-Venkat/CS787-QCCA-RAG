"""Execute copies with repository defaults only: synthetic smoke, no full runs."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import nbformat
from nbclient import NotebookClient
from src.week12.protocol import load,root,atomic,code_hash

def main():
 cfg=load();out=root(cfg)/'smoke/notebooks'/code_hash()[:12];out.mkdir(parents=True,exist_ok=True);results=[]
 for path in sorted((ROOT/'notebooks/week12').glob('*.ipynb')):
  n=nbformat.read(path,as_version=4);flags=n.cells[1].source
  assert 'RUN_FULL = False' in flags and 'RUN_FULL_LORA_TRAIN = False' in flags
  target=out/path.name
  try:NotebookClient(n,timeout=900,kernel_name='sara_env',resources={'metadata':{'path':str(ROOT)}}).execute();nbformat.write(n,target);results.append(dict(notebook=path.name,status='PASS'));print(path.name,'PASS',flush=True)
  except Exception:
   nbformat.write(n,target);results.append(dict(notebook=path.name,status='FAIL'));atomic(out/'validation.json',results);raise
 atomic(out/'validation.json',results);print('All defaults smoke-only; no full runs.')
if __name__=='__main__':main()
