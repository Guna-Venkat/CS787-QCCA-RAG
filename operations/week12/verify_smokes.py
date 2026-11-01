"""Bounded qualification only. Never supplies a full-run flag."""
from pathlib import Path
import sys,subprocess,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.week12.protocol import load,root,atomic,identity

def main():
 cfg=load();out=root(cfg)/'qualification';out.mkdir(parents=True,exist_ok=True);steps=[]
 jobs=[('real_qbgf',['scripts/week12/run.py','--phase','B','--real-smoke','--limit','3']),('canonical_sara',['scripts/week12/run.py','--phase','B','--canonical','SARA_CANONICAL_BM25_K5_N10']),('canonical_rag',['scripts/week12/run.py','--phase','B','--canonical','CANONICAL_RAG_BM25']),('real_qlora_synthetic',['scripts/week12/qualify_lora.py'])]
 for name,args in jobs:
  assert '--full' not in args and '--train' not in args
  begin=time.time()
  with (out/(name+'.log')).open('w') as f:result=subprocess.run([str(ROOT/'sara_env/bin/python')]+args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  steps.append(dict(name=name,status='PASS' if result.returncode==0 else 'FAIL',seconds=time.time()-begin));atomic(out/'smoke_suite.json',dict(identity=identity(cfg),steps=steps,full_runs_started=False,unique_real_development_queries=3,confirmation_used=False))
  print(name,steps[-1]['status'],flush=True)
  if result.returncode:raise SystemExit(result.returncode)
 # Resume proof: second real invocation must leave all generation shard hashes/counts unchanged.
 from src.week12.runner import directory,execute
 from src.week12.protocol import sha
 p=directory(cfg,'B',False,True)/'rows';before={f.name:sha(f) for f in p.glob('*.json')};execute('B',cfg,smoke=True,real_smoke=True,limit=3);after={f.name:sha(f) for f in p.glob('*.json')};assert before==after
 atomic(out/'resume_smoke.json',dict(passed=True,result_shards_unchanged=True,full_run_started=False,identity=identity(cfg)))
 print('Resume passed; no full runs.',flush=True)
if __name__=='__main__':main()
