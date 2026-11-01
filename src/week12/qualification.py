import importlib.util,importlib.metadata,json,shutil,subprocess
from pathlib import Path
from .protocol import ROOT,root,atomic,sha,digest,identity

def resources(cfg):
    import psutil
    hub=Path.home()/'.cache/huggingface/hub';models={}
    from huggingface_hub import snapshot_download
    for name,spec in cfg['models'].items():
        try:p=Path(snapshot_download(spec['id'],revision=spec['revision'],local_files_only=True));models[name]=dict(status='CACHED',revision=spec['revision'],snapshot=str(p),weight_files=len(list(p.glob('*.safetensors'))))
        except Exception as e:models[name]=dict(status='MISSING',reason=type(e).__name__)
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    cache_bytes=sum(p.stat().st_size for p in hub.rglob('*') if p.is_file() and not p.is_symlink())
    return dict(disk_free_GiB=shutil.disk_usage(ROOT).free/2**30,HF_cache_GiB=cache_bytes/2**30,CPU_RAM_available_GiB=psutil.virtual_memory().available/2**30,GPU_memory=gpu,models=models,packages={p:importlib.metadata.version(p) if importlib.util.find_spec(p) else None for p in ['torch','transformers','peft','bitsandbytes','llmlingua']},expected_LLMLingua_weight_GiB=cfg['llmlingua']['expected_weight_GiB'])
def inventory(cfg):
    checkpoint=ROOT/cfg['sara']['checkpoint'];rag=ROOT/'Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42';has=lambda p:(p/'adapter_model.safetensors').exists()
    rows=[dict(id='NONE',category='canonical_control',reader='Qwen',status='IMPLEMENTED'),dict(id='CANONICAL_RAG_BM25',category='external',reader='adapted Mistral RAG',status='SOURCE_ADAPTER_AVAILABLE_EXECUTION_SMOKE_PENDING' if has(rag) else 'MISSING_CHECKPOINT',checkpoint=str(rag),budget='natural original top10; separate specialist reference'),dict(id='SARA_CANONICAL_BM25_K5_N10',category='external',reader='adapted Mistral SARA',k=5,n=10,status='SOURCE_ADAPTER_AVAILABLE_EXECUTION_SMOKE_PENDING' if has(checkpoint) else 'MISSING_CHECKPOINT',checkpoint=str(checkpoint),max_new_tokens=64),dict(id='STRONG_SFR_RAG',category='external_control',reader='Qwen',status='IMPLEMENTED_NATURAL_TOP10_REFERENCE'),dict(id='LLMLINGUA2',category='external',reader='Qwen',status='QUALIFICATION_REQUIRED'),dict(id='SARA_SFR_K5',category='historical_specialist',status='NOT_QUALIFIED_HISTORICAL_POOL_PARITY_REQUIRED'),dict(id='SARA_SFR_K6_HISTORICAL',category='historical_specialist',status='NOT_QUALIFIED_HISTORICAL_POOL_PARITY_REQUIRED')]+[dict(id=m,category='project_baseline',status='IMPLEMENTED_NEW_PROTOCOL_PORT') for m in ['FIXED_SENTENCE','C2','H3','RAW_STRICT','RAW_SKIP']]+[dict(id=m,category='QBGF_ablation',status='IMPLEMENTED') for m in ['QBGF_NO_QUERY','QBGF_SENTENCE_ONLY','QBGF_PASSAGE_ONLY','QBGF_NO_MASS','QBGF_GREEDY','QBGF_BM25']]

    for row in rows:
        p=root(cfg)/'qualification'/(row['id']+'.json')
        if p.exists():
            rec=json.loads(p.read_text())
            if rec.get('status')=='PASS' and rec.get('identity')==identity(cfg):row['status']='REAL_SMOKE_PASSED';row['qualification_receipt']=str(p)
    from .compression_runtime import status
    compressed=status(cfg)
    for row in rows:
        if row['id']=='LLMLINGUA2':
            row['status']=compressed['status'];row['effective_model']=compressed.get('id');row['effective_revision']=compressed.get('revision')
    return rows

def qualify_llm(cfg,allow_download=False,smoke=True):
    from .compression_runtime import status
    locked=status(cfg)
    if locked['status']=='QUALIFIED':return locked
    before=resources(cfg);spec=cfg['llmlingua'];result=dict(status='NOT_QUALIFIED',reason=None,model=spec,resource_before=before,full_run_started=False)
    if not importlib.util.find_spec('llmlingua'):result['reason']='MISSING_DEPENDENCY; no automatic installation'
    elif not spec.get('revision'):result['reason']='PINNED_REVISION_REQUIRED; choose before outcomes'
    elif before['disk_free_GiB']<max(10,2*spec['expected_weight_GiB']):result['reason']='INSUFFICIENT_DISK_HEADROOM'
    else:
        try:
            from huggingface_hub import snapshot_download
            path=snapshot_download(spec['id'],revision=spec['revision'],local_files_only=not allow_download)
            if smoke:
                from transformers import AutoTokenizer
                tok=AutoTokenizer.from_pretrained(cfg['models']['Qwen']['id'],revision=cfg['models']['Qwen']['revision'],local_files_only=True)
                text='The experiment measures a red signal. The unrelated appendix describes blue paint. '*20
                out=compress_llm(cfg,text,64,tok,path);assert out['evidence_tokens']<=64
            result.update(status='QUALIFIED',reason=None,snapshot=path,smoke=smoke)
        except Exception as e:result['reason']=f'{type(e).__name__}: {e}'
    atomic(root(cfg)/'qualification'/'execution'/identity(cfg)['code_hash'][:12]/'llmlingua.json',result);return result

def compress_llm(cfg,text,budget,tok,path=None):
    from .compression_runtime import status,compress
    if status(cfg)['status']=='QUALIFIED':return compress(cfg,text,budget,tok)
    from llmlingua import PromptCompressor
    from huggingface_hub import snapshot_download
    import time
    if not cfg['llmlingua'].get('revision'):raise RuntimeError('No pinned LLMLingua revision')
    path=path or snapshot_download(cfg['llmlingua']['id'],revision=cfg['llmlingua']['revision'],local_files_only=True)
    compressor=PromptCompressor(model_name=str(path),use_llmlingua2=True,device_map='cpu');start=time.perf_counter()
    for factor in [1.,.8,.6]:
        result=compressor.compress_prompt_llmlingua2(text,rate=min(1.,budget*factor/max(1,len(tok.encode(text,add_special_tokens=False)))),force_tokens=['\n'],chunk_end_tokens=['.','\n'],return_word_label=True)
        s=result['compressed_prompt'];ids=tok.encode(s,add_special_tokens=False)
        if len(ids)<=budget:return dict(text=s,evidence_ids=list(ids),evidence_tokens=len(ids),unused_budget=budget-len(ids),selected_unit_ids=[],units=[],choices=[],selection_rank_ids=[],method='LLMLINGUA2',budget=budget,source_support=None,evidence_hash=digest([s,ids]),serialization='external-token-deletion',compression_seconds=time.perf_counter()-start,notes='Task-agnostic token compression; no exact contiguous span or query conditioning claim')
    raise ValueError('LLMLingua cannot meet actual Qwen budget; no silent truncation')
