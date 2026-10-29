"""Faithful specialist references on DEVELOPMENT QASPER only; no TEST paths."""
import json,gc
from .protocol import ROOT,root,guard,digest,sha,Store,identity,atomic

def assert_base_pin(cfg,checkpoint):
    """Require the historical loader's local main ref to match the frozen base."""
    from pathlib import Path
    spec=cfg['models']['Mistral']
    adapter=json.loads((checkpoint/'adapter_config.json').read_text())
    if adapter.get('base_model_name_or_path')!=spec['id']:
        raise ValueError('Canonical adapter base differs from the frozen Mistral model')
    ref=Path.home()/'.cache/huggingface/hub'/('models--'+spec['id'].replace('/','--'))/'refs/main'
    if not ref.exists() or ref.read_text().strip()!=spec['revision']:
        raise ValueError('Historical loader cached base revision differs from the frozen pin')

def prepare(cfg,examples,system):
    if system not in ['SARA_CANONICAL_BM25_K5_N10','CANONICAL_RAG_BM25']:raise ValueError('Unqualified historical SFR variant cannot impersonate original SARA')
    if cfg['sara']['canonical_k']!=5 or cfg['sara']['canonical_n']!=10:raise ValueError('Canonical SARA is fixed BM25 k5/n10')
    q={e['query_id'] for e in examples if e['dataset']=='QASPER'}
    if cfg.get('split_protocol'):
        rows=[dict(id=e['query_id'],example_id=e['paper_id'],question=e['question'],answer=e['evaluation']['gold_answer'],context=e['metadata']['canonical_context']) for e in examples if e['dataset']=='QASPER']
    else:
        path=ROOT/cfg['datasets']['QASPER']['path'];guard('QASPER','dev',path)
        rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()];rows=[r for r in rows if str(r['id']) in q]
    assert len(rows)==len(q)
    out=root(cfg)/'canonical_inputs';out.mkdir(parents=True,exist_ok=True)
    receipt=out/(system+'_preparation.json');input_hash=digest([rows,cfg['models']['SFR'],system])
    if receipt.exists():
        old=json.loads(receipt.read_text());artifact=out/old['artifact']
        if old['input_hash']!=input_hash or sha(artifact)!=old['artifact_hash']:raise ValueError('Canonical prepared inputs changed')
        if system.startswith('SARA'):
            import torch
            return torch.load(artifact,map_location='cpu',weights_only=False)['records']
        return json.loads(artifact.read_text())
    subset=out/'development_subset.jsonl';subset.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    from src.week2.precompute_dev_embeddings import retrieve_dev_passages
    records=retrieve_dev_passages(subset,chunk_size=256,num_passages=10)
    if system.startswith('SARA'):
        import torch
        from sentence_transformers import SentenceTransformer
        spec=cfg['models']['SFR'];encoder=SentenceTransformer(spec['id'],revision=spec['revision'],device='cuda',local_files_only=True,model_kwargs={'torch_dtype':torch.bfloat16})
        vectors=encoder.encode([p['text'] for r in records for p in r['passages']],batch_size=16,convert_to_tensor=True,show_progress_bar=False).detach().cpu().to(torch.bfloat16)
        assert tuple(vectors.shape)==(len(records)*10,4096) and torch.isfinite(vectors).all()
        del encoder;gc.collect();torch.cuda.empty_cache()
        torch.save(dict(embeddings={r['question_id']:vectors[i*10:(i+1)*10] for i,r in enumerate(records)},records=records,identity=digest([cfg['models']['SFR'],records]),system=system),out/(system+'.pt'))
    else:
        atomic(out/(system+'.json'),records)
    artifact=out/(system+('.pt' if system.startswith('SARA') else '.json'))
    atomic(receipt,dict(input_hash=input_hash,artifact=artifact.name,artifact_hash=sha(artifact)))
    return records

def generate(cfg,examples,system):
    if system not in ['SARA_CANONICAL_BM25_K5_N10','CANONICAL_RAG_BM25']:raise ValueError('Unqualified specialist label')
    import torch
    from src.week9.final_h05.historical_backend import HistoricalBackend
    from src.week9.final_h05.literal import bm25_prompt
    if system.startswith('SARA'):
        data=torch.load(root(cfg)/'canonical_inputs'/(system+'.pt'),weights_only=False,map_location='cpu');records=data['records'];checkpoint=cfg['sara']['checkpoint'];k=5
    elif system=='CANONICAL_RAG_BM25':
        records=json.loads((root(cfg)/'canonical_inputs'/(system+'.json')).read_text());data={};checkpoint='Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42';k=10
    else:raise ValueError('Canonical system identity unqualified')
    ck=ROOT/checkpoint;assert_base_pin(cfg,ck);hashes={p.name:sha(p) for p in ck.glob('*') if p.is_file() and p.suffix in ('.json','.safetensors','.model')}
    store=Store(root(cfg)/'canonical_references'/system,identity(cfg,dict(system=system,checkpoint=hashes,input_hash=digest(records))))
    pending=[r for r in records if store.get(r['question_id']) is None]
    if not pending:return store.rows()
    backend=HistoricalBackend(dict(checkpoint=checkpoint,device='cuda'))
    try:
        byid={e['query_id']:e for e in examples}
        from src.week11.evaluation import evaluate
        for r in pending:
            prompt,_=bm25_prompt()(r['question'],r['passages'],k,backend.tokenizer)
            embeds=data['embeddings'][r['question_id']][5:].to('cuda',dtype=backend.model.dtype) if k==5 else None
            result=backend.generate(prompt,64,embeds);e=byid[r['question_id']]
            row=dict(dataset='QASPER',query_id=e['query_id'],paper_id=e['paper_id'],cluster_id=e['cluster_id'],method=system,reader='CANONICAL_ADAPTED_MISTRAL',k=k,n=10,split='development',raw_prediction=result['content'],prediction=result['content'],checkpoint_hash=digest(hashes),prompt_tokens=result['prompt_tokens'],generated_tokens=result['generated_tokens'],num_lm_calls=1,**evaluate(result['content'],e))
            store.put(r['question_id'],row)
    finally:backend.model=None;gc.collect();torch.cuda.empty_cache()
    return store.rows()
