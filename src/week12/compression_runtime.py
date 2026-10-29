"""Optional official baseline lock, separate from unchanged split/science config."""
import json,subprocess
from .protocol import ROOT,root,digest,sha

def lock_path(cfg):return root(cfg)/'protocol/llmlingua_qualification_lock.json'
def status(cfg):
 p=lock_path(cfg)
 if not p.exists():return dict(status='NOT_QUALIFIED',reason='No successful locked runtime qualification')
 v=json.loads(p.read_text());payload={k:x for k,x in v.items() if k!='checksum'}
 if v['checksum']!=digest(payload):raise ValueError('LLMLingua runtime lock changed')
 return v

def compress(cfg,text,budget,tok):
 import time
 v=status(cfg)
 if v['status']!='QUALIFIED':raise RuntimeError(v['reason'])
 if sha(v['snapshot']+'/model.safetensors')!=v['weight_sha256']:raise ValueError('LLMLingua checkpoint changed')
 start=time.perf_counter();n=len(tok.encode(text,add_special_tokens=False))
 for factor in [1.,.8,.6]:
  req=dict(snapshot=v['snapshot'],text=text,rate=min(1.,budget*factor/max(1,n)))
  p=subprocess.run([v['python'],str(ROOT/'scripts/week12/llmlingua_worker.py')],input=json.dumps(req),text=True,capture_output=True,cwd=ROOT,timeout=300)
  if p.returncode:raise RuntimeError('Isolated LLMLingua worker failed: '+p.stderr[-1500:])
  output=json.loads(p.stdout)['compressed_prompt'];ids=tok.encode(output,add_special_tokens=False)
  if len(ids)<=budget:return dict(text=output,evidence_ids=ids,evidence_tokens=len(ids),unused_budget=budget-len(ids),selected_unit_ids=[],units=[],choices=[],selection_rank_ids=[],method='LLMLINGUA2',budget=budget,source_support=None,evidence_hash=digest([output,ids]),serialization='external-token-deletion',compression_seconds=time.perf_counter()-start,compressor_id=v['id'],compressor_revision=v['revision'],compressor_lock_hash=sha(lock_path(cfg)),notes='Official LLMLingua-2 token deletion, actual Qwen budget validated; no exact-span or causal-support claim')
 raise ValueError('Official LLMLingua output exceeds Qwen budget after predefined retries; no truncation')
