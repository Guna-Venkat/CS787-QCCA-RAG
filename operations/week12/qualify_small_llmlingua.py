"""One official small compressor qualification, synthetic text only. No answer LM."""
from pathlib import Path
import sys,json,subprocess,shutil,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.week12.protocol import load,root,atomic,immutable,digest,sha

def main():
 c=load();out=root(c)/'qualification/scheduler_patch';out.mkdir(parents=True,exist_ok=True)
 spec=json.loads(Path('/tmp/w12_llmlingua_selected.json').read_text());foot=json.loads(Path('/tmp/w12_llmlingua_footprints.json').read_text());atomic(out/'official_checkpoint_footprints.json',foot)
 runtime=ROOT/'results/week12/tools/llmlingua_env/bin/python';text=('The measurement detected a red signal. The unrelated appendix describes blue paint and old equipment. ')*120
 result=dict(status='NOT_QUALIFIED',reason=None,model=spec,disk_free_before_download=spec['free_bytes_before'],disk_free_now=shutil.disk_usage(ROOT).free,environment=str(runtime),original_environment_modified=False,no_caches_deleted=True,synthetic_only=True,answer_generation_calls=0)
 if result['disk_free_now']<10*2**30:result['reason']='Insufficient retained 10 GiB safety reserve'
 else:
  try:
   from transformers import AutoTokenizer
   tok=AutoTokenizer.from_pretrained(c['models']['Qwen']['id'],revision=c['models']['Qwen']['revision'],local_files_only=True);raw=len(tok.encode(text,add_special_tokens=False));checks=[]
   for budget in [256,1024]:
    fitted=False
    for factor in [1.,.8,.6]:
     start=time.perf_counter();p=subprocess.run([str(runtime),str(ROOT/'scripts/week12/llmlingua_worker.py')],input=json.dumps(dict(snapshot=spec['snapshot'],text=text,rate=min(1.,budget*factor/raw))),capture_output=True,text=True,cwd=ROOT,timeout=300)
     (out/f'worker_{budget}_{factor}.log').write_text(p.stderr)
     if p.returncode:raise RuntimeError(p.stderr[-2000:])
     compressed=json.loads(p.stdout)['compressed_prompt'];n=len(tok.encode(compressed,add_special_tokens=False))
     if n<=budget:checks.append(dict(budget=budget,raw_qwen_tokens=raw,compressed_qwen_tokens=n,retry_factor=factor,seconds=time.perf_counter()-start));fitted=True;break
    if not fitted:raise ValueError('Official output failed actual Qwen budget after registered retries')
   versions=subprocess.check_output([str(runtime),'-c',"import importlib.metadata,json;print(json.dumps({p:importlib.metadata.version(p) for p in ['llmlingua','transformers','torch','nltk','tiktoken','numpy']}))"],text=True)
   lock=dict(status='QUALIFIED',id=spec['id'],revision=spec['revision'],snapshot=spec['snapshot'],python=str(runtime),package_versions=json.loads(versions),weight_sha256=sha(Path(spec['snapshot'])/'model.safetensors'),worker_sha256=sha(ROOT/'scripts/week12/llmlingua_worker.py'),synthetic_budget_checks=checks,reason=None,source='https://github.com/microsoft/LLMLingua',frozen_science_config_unchanged=True)
   lock['checksum']=digest(lock);immutable(root(c)/'protocol/llmlingua_qualification_lock.json',lock);result.update(status='QUALIFIED',checks=checks,package_versions=json.loads(versions))
  except Exception as e:result['reason']=type(e).__name__+': '+str(e)
 atomic(out/'llmlingua_attempt.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
