"""QASPER-only Mistral/SARA references; never represent these as budget matched."""
import copy,json,gc
from src.week12.protocol import ROOT,root,digest,sha,atomic,Store
from .baselines import folder,extension_identity
SYSTEMS=('SARA_CANONICAL_BM25_K5_N10','SARA_SFR_K5_N10')

def config(cfg,system):
    if system not in SYSTEMS:raise ValueError('Unqualified specialist')
    out=copy.deepcopy(cfg);out['output_root']=str((folder(cfg)/'specialists'/system).relative_to(ROOT));return out

def prepare(cfg,system,smoke=False):
    from src.week12.runner import examples
    from src.week12.canonical import prepare as canonical_prepare,assert_base_pin
    from src.week12.protocol import cohort
    xs=[e for e in examples(cfg) if e['dataset']=='QASPER']
    if smoke:xs=xs[:1]
    sc=config(cfg,system)
    if smoke:sc['output_root']+='/qualification_smoke'
    assert_base_pin(cfg,ROOT/cfg['sara']['checkpoint'])
    if system==SYSTEMS[0]:return canonical_prepare(sc,xs,system)
    # Same canonical 256-token splitter and SARA checkpoint; only retrieval order changes.
    import torch,numpy as np
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter
    from sentence_transformers import SentenceTransformer
    from src.week10.root_cause.encoding import SFR_INSTRUCTION
    out=root(sc)/'canonical_inputs';out.mkdir(parents=True,exist_ok=True)
    st=Store(out/'sfr_records',dict(extension=extension_identity(cfg),split_ids=[e['query_id'] for e in xs],splitter='llama_index SentenceSplitter(chunk_size=256), canonical defaults',query_prefix=SFR_INSTRUCTION))
    encoder=None
    try:
        for e in xs:
            if st.get(e['query_id']) is not None:continue
            if encoder is None:
                spec=cfg['models']['SFR'];encoder=SentenceTransformer(spec['id'],revision=spec['revision'],device='cuda',local_files_only=True,model_kwargs={'torch_dtype':torch.bfloat16})
            context=e['metadata']['canonical_context'];context=context if isinstance(context,list) else [context]
            nodes=SentenceSplitter(chunk_size=256).get_nodes_from_documents([Document(text=str(t)) for t in context if str(t).strip()])
            texts=[n.text for n in nodes]
            if not texts:raise ValueError('Empty canonical source')
            # Store only selected raw BF16 latent vectors, rank using normalized copies.
            vec=encoder.encode(texts,batch_size=8,convert_to_tensor=True,show_progress_bar=False).detach().cpu().float()
            q=encoder.encode([SFR_INSTRUCTION+e['question']],convert_to_tensor=True,show_progress_bar=False).detach().cpu().float()[0]
            score=torch.nn.functional.normalize(vec,dim=1)@torch.nn.functional.normalize(q,dim=0)
            order=sorted(range(len(texts)),key=lambda i:(-float(score[i]),i))[:10]
            if len(order)<10:raise ValueError('Cannot qualify SFR SARA n10 without ten genuine source chunks; no silent padding')
            passages=[dict(passage_id=e['paper_id']+':'+digest([i,texts[i]]),rank=j+1,score=float(score[i]),text=texts[i]) for j,i in enumerate(order)]
            record=dict(question_id=e['query_id'],paper_id=e['paper_id'],question=e['question'],passages=passages)
            path=out/(digest(e['query_id'])+'.pt');torch.save(vec[order].to(torch.bfloat16),path)
            st.put(e['query_id'],dict(record=record,vector_file=path.name,vector_hash=sha(path),source_hash=digest(context)))
    finally:
        del encoder;gc.collect();torch.cuda.empty_cache()
    return st.rows()

def generate(cfg,system,smoke=False):
    from src.week12.runner import examples
    from src.week12.canonical import generate as canonical_generate,assert_base_pin
    sc=config(cfg,system)
    if smoke:sc['output_root']+='/qualification_smoke'
    xs=[e for e in examples(cfg) if e['dataset']=='QASPER'];xs=xs[:1] if smoke else xs
    if system==SYSTEMS[0]:return canonical_generate(sc,xs,system)
    import torch
    from src.week9.final_h05.historical_backend import HistoricalBackend
    from src.week9.final_h05.literal import bm25_prompt
    from src.week11.evaluation import evaluate
    assert_base_pin(cfg,ROOT/cfg['sara']['checkpoint'])
    p=root(sc)/'canonical_inputs/sfr_records/identity.json';source=Store(p.parent,json.loads(p.read_text()))
    ck=ROOT/cfg['sara']['checkpoint'];hashes={p.name:sha(p) for p in ck.glob('*') if p.is_file() and p.suffix in ('.json','.safetensors','.model')}
    st=Store(root(sc)/'canonical_references'/system,dict(extension=extension_identity(cfg),checkpoint_hashes=hashes,input_identity=source.identity));backend=None
    try:
        for e in xs:
            if st.get(e['query_id']) is not None:continue
            r=source.get(e['query_id']);path=p.parent.parent/r['vector_file']
            if sha(path)!=r['vector_hash']:raise ValueError('Latent vectors changed')
            vec=torch.load(path,map_location='cpu',weights_only=True)
            if tuple(vec.shape)!=(10,4096) or not torch.isfinite(vec).all():raise ValueError('SARA latent dimensions/nonfinite')
            if backend is None:backend=HistoricalBackend(dict(checkpoint=cfg['sara']['checkpoint'],device='cuda'))
            prompt,_=bm25_prompt()(e['question'],r['record']['passages'],5,backend.tokenizer)
            result=backend.generate(prompt,64,vec[5:].to('cuda',dtype=backend.model.dtype))
            st.put(e['query_id'],dict(dataset='QASPER',query_id=e['query_id'],paper_id=e['paper_id'],cluster_id=e['cluster_id'],method=system,reader='CANONICAL_ADAPTED_MISTRAL',budget='SPECIALIST_NATURAL',k=5,n=10,split='VALIDATION',prediction=result['content'],raw_prediction=result['content'],prompt_tokens=result['prompt_tokens'],generated_tokens=result['generated_tokens'],checkpoint_hash=digest(hashes),comparison_scope='specialist cross-reader reference; NOT 256/1024 matched',**evaluate(result['content'],e)))
    finally:
        if backend:backend.model=None
        gc.collect();torch.cuda.empty_cache()
    return st.rows()

def qualify(cfg,system):
    """Receipt written only after separate-process real preparation and real generation pass."""
    sc=config(cfg,system);sc['output_root']+='/qualification_smoke'
    p=root(sc)/'canonical_references'/system/'identity.json'
    if not p.exists():raise ValueError('Run real specialist smoke first')
    rows=Store(p.parent,json.loads(p.read_text())).rows()
    if len(rows)!=1 or not rows[0]['prediction'].strip():raise ValueError('One nonempty real QASPER smoke answer required')
    receipt=dict(status='QUALIFIED',system=system,extension=extension_identity(cfg),smoke_identity=json.loads(p.read_text()),checkpoint_hash=rows[0]['checkpoint_hash'],real_queries=1,k=5,n=10,parity='canonical chunk256/Mistral adapted/checkpoint/prompt/BF16 latent4096/greedy64; SFR changes retrieval only; not matched to Qwen budgets')
    atomic(folder(cfg)/'qualification'/(system+'.json'),receipt);return receipt

def require_qualified(cfg,system):
    p=folder(cfg)/'qualification'/(system+'.json')
    if not p.exists():raise ValueError('Qualify specialist real one-query smoke before full execution')
    r=json.loads(p.read_text())
    if r['extension']!=extension_identity(cfg) or r['status']!='QUALIFIED':raise ValueError('Stale specialist qualification')
    ck=ROOT/cfg['sara']['checkpoint'];hashes={p.name:sha(p) for p in ck.glob('*') if p.is_file() and p.suffix in ('.json','.safetensors','.model')}
    if digest(hashes)!=r['checkpoint_hash']:raise ValueError('Specialist checkpoint changed')
