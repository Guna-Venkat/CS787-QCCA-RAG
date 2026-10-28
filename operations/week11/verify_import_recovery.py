"""Tiny public-reader replay plus SARA text/latent execution qualification."""
from pathlib import Path
from datetime import datetime,timezone
import json
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.week11.protocol import load,gate,output,code_identity
from src.week11.cache import Store,digest,atomic,file_hash
from operations.week11.isolated_readers import IsolatedTextReader,IsolatedSaraReader


def question(cfg,dataset,qid):
    p=ROOT/cfg['cache_root']/'datasets/queries'/(digest([dataset,qid])+'.json')
    obj=json.loads(p.read_text());assert digest(obj['row'])==obj['checksum']
    return obj['row']['inference']['question']


def main():
    cfg=load();gate(cfg,'E');root=output(cfg,'D')
    identity=json.loads((root/'manifest.json').read_text())['identity']
    row=next(r for r in Store(root/'rows',identity).rows() if r['reader']=='Mistral' and r['dataset']=='HotpotQA' and r['budget']==512 and r['evidence_granularity']=='PASSAGE' and r['selector']=='semantic')
    selection=None;latent=None;latent_key=None
    for p in (ROOT/cfg['cache_root']/'selections').rglob('*.json'):
        if len(p.stem)!=64:continue
        obj=json.loads(p.read_text());s=obj['row']
        if obj['checksum']==row['selection_hash']:
            assert digest(s)==obj['checksum'];selection=s
        if obj['key'].get('compressor')=='C3' and s.get('latent_vectors'):
            assert digest(s)==obj['checksum'];latent=s;latent_key=obj['key']
        if selection is not None and latent is not None:break
    assert selection is not None and latent is not None
    public=IsolatedTextReader(cfg,'Mistral')
    try:
        result=public.generate(question(cfg,row['dataset'],row['query_id']),selection['text'],row['task_type'])
        assert all(result[k]==row[k] for k in ['prediction','prompt_tokens','generated_tokens'])
        public_bootstrap=public.bootstrap
    finally:public.close()
    specialist=IsolatedSaraReader(dict(checkpoint=cfg['sara_checkpoint']))
    try:
        q=question(cfg,'QASPER',latent_key['query_id'])
        baseline=specialist.generate_text(q,'')
        tokens=specialist.tokenizer.encode(latent['text'],add_special_tokens=False)
        pool=[dict(text=u['source_text']) for u in latent['latent_pool']]
        adapted=specialist.generate(q,pool,cfg['sara_k'],np.asarray(latent['latent_vectors']))
        assert baseline['generated_tokens']>0 and adapted['generated_tokens']>0 and len(tokens)>0
        specialist_bootstrap=specialist.bootstrap
    finally:specialist.close()
    assert not public.process.is_alive() and not specialist.process.is_alive()
    receipt=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),passed=True,extra_operational_lm_calls=3,
        mode='operational import/SARA qualification only; no scientific outcome rows written',
        public_prediction_and_token_counts_exact=True,public_query_id=row['query_id'],public_bootstrap=public_bootstrap,
        SARA_text_and_latent_generation_succeeded=True,SARA_query_id=latent_key['query_id'],SARA_evidence_tokens=len(tokens),SARA_bootstrap=specialist_bootstrap,
        workers_exited=True,GPU_after=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader'],text=True).strip(),
        core_code_hash=digest(code_identity()),driver_sha256=file_hash(ROOT/'operations/week11/isolated_readers.py'),verification_driver_sha256=file_hash(Path(__file__)))
    atomic(ROOT/'results/week11/monitoring/import_recovery_verification.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
