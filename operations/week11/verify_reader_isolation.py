"""Tiny real-model operational replay; no existing scientific results are written."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from operations.week11.isolated_readers import IsolatedTextReader
from src.week11.protocol import load,gate,output
from src.week11.cache import Store,digest,atomic,file_hash


def gpu():
    return subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader'],text=True).strip()


def main():
    cfg=load();gate(cfg,'D');root=output(cfg,'D');identity=json.loads((root/'manifest.json').read_text())['identity']
    rows=Store(root/'rows',identity).rows()
    samples=[]
    for dataset,granularity,budget in [('HotpotQA','PASSAGE',512),('SciFact','SENTENCE',256)]:
        samples.append(next(r for r in rows if r['reader']=='Mistral' and r['dataset']==dataset and r['evidence_granularity']==granularity and r['budget']==budget and r['selector']=='semantic'))
    targets={r['selection_hash'] for r in samples};selections={}
    for path in (ROOT/cfg['cache_root']/'selections').rglob('*.json'):
        if len(path.stem)!=64:continue
        obj=json.loads(path.read_text());s=obj['row'];key=digest(s)
        if key in targets:
            assert key==obj['checksum'];selections[key]=s
        if len(selections)==len(targets):break
    assert set(selections)==targets
    before=gpu();checks=[]
    reader=IsolatedTextReader(cfg,'Mistral');pid=reader.process.pid
    try:
        for r in samples:
            path=ROOT/cfg['cache_root']/'datasets/queries'/(digest([r['dataset'],r['query_id']])+'.json')
            obj=json.loads(path.read_text());assert digest(obj['row'])==obj['checksum'];q=obj['row']['inference']['question']
            result=reader.generate(q,selections[r['selection_hash']]['text'],r['task_type'])
            for key in ['prediction','prompt_tokens','generated_tokens']:assert result[key]==r[key],(r['dataset'],r['query_id'],key)
            checks.append(dict(dataset=r['dataset'],query_id=r['query_id'],action_id=r['action_id'],prediction_exact_match=True,prompt_and_generation_tokens_exact_match=True))
    finally:reader.close()
    assert not reader.process.is_alive();after=gpu()
    # Qualify the previously blocked reader in a fresh, owned CUDA process.
    llama=IsolatedTextReader(cfg,'Llama')
    try:
        result=llama.generate('Is water composed of hydrogen and oxygen?','','freeform_qa')
        assert isinstance(result['prediction'],str) and result['generated_tokens']>0
        llama_check=dict(reader='Llama',generation_succeeded=True,prediction=result['prediction'],prompt_tokens=result['prompt_tokens'],generated_tokens=result['generated_tokens'])
    finally:llama.close()
    assert not llama.process.is_alive()
    receipt=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),passed=True,mode='operational replay and one qualification call; not scientific outcome rows',
        extra_operational_lm_calls=3,canonical_Mistral_replays=checks,Llama_qualification=llama_check,Mistral_worker_pid=pid,workers_exited=True,
        gpu_before=before,gpu_after_Mistral_close=after,gpu_after_Llama_close=gpu(),
        frozen_scientific_identity=identity,driver_sha256=file_hash(ROOT/'operations/week11/isolated_readers.py'),verification_driver_sha256=file_hash(Path(__file__)))
    atomic(ROOT/'results/week11/monitoring/reader_isolation_verification.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
