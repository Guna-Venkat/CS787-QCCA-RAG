"""Resume by validated query/action shards. Full work is manual opt-in only."""
import copy,json,subprocess,sys,time
from pathlib import Path
from .protocol import ROOT,root,load,Store,identity,digest,atomic,immutable,lock,authorize,require_full,cohort,sha
from .algorithm import METHODS,choose,serialize,validate_spans
from .reader import parse_label

MAIN=['NONE','RAW_STRICT','RAW_SKIP','FIXED_SENTENCE','C2','H3','QBGF_FULL','STRONG_SFR_RAG','BM25_QWEN_REFERENCE']
ABLATIONS=['QBGF_FULL','QBGF_NO_QUERY','QBGF_SENTENCE_ONLY','QBGF_PASSAGE_ONLY','QBGF_NO_MASS','QBGF_GREEDY','QBGF_BM25']
def fixture_examples():
    from src.week11.fixtures import examples
    from src.week11.datasets import normalize,document
    xs=examples();qa=normalize('QASPER','dev','fixture-qasper','What signal was observed?',[document('QASPER','fixture-paper','An experiment observed a red signal. The appendix discusses blue paint.')],['red signal'],gold=[])
    return [qa,next(e for e in xs if e['dataset']=='HotpotQA'),next(e for e in xs if e['dataset']=='SciFact')]
def examples(cfg,synthetic=False,limit=None):
    if synthetic:return fixture_examples()[:limit or 3]
    if limit and cfg.get('split_protocol'):
        from .splits import load_examples
        return load_examples(cfg,'VAL',per_dataset_limit=1)[:limit]
    from .data import development_examples
    xs=development_examples(cfg)
    if limit:
        # Same deterministic <=3 unique real queries for all qualification checks.
        return [next(e for e in xs if e['dataset']==d) for d in cfg['datasets']][:limit]
    return xs

def directory(cfg,phase,synthetic=False,real_smoke=False,core_final=True):
    if not core_final and phase in ('C','D'):phase=phase+'_secondary_enabled'
    ident=identity(cfg)
    return root(cfg)/('smoke/synthetic/'+ident['code_hash'][:12] if synthetic else 'smoke/real/'+ident['code_hash'][:12] if real_smoke else ('validation' if cfg.get('split_protocol') else 'development'))/phase

def phase_methods(phase,core_final=True):
    from .execution import methods
    return methods(phase,core_final)

def prepare(cfg,phase,synthetic=False,real_smoke=False,limit=3,core_final=True):
    from .laboratory import Lab
    from .protocol import code_hash
    from .statistics import registry
    from src.week11.readers import tokenizers
    from src.week11.protocol import load as oldcfg
    if synthetic:
        from src.week11.budgets import FixtureTokenizer
        tok=FixtureTokenizer();legacy={'Qwen':tok}
    else:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(cfg['models']['Qwen']['id'],revision=cfg['models']['Qwen']['revision'],local_files_only=True)
        legacy=tokenizers(oldcfg())
    folder=directory(cfg,phase,synthetic,real_smoke,core_final);meta=identity(cfg,dict(phase=phase,synthetic=synthetic,real_smoke=real_smoke));store=Store(folder/'prepared',meta);xs=examples(cfg,synthetic,limit if synthetic or real_smoke else None);lab=Lab(cfg,tok,synthetic)
    methods=['QBGF_FULL'] if real_smoke else phase_methods(phase,core_final)
    qualification_path=root(cfg)/'qualification/llmlingua.json'
    if phase=='E' and not synthetic:
        from .compression_runtime import status as compressor_status
        qual=compressor_status(cfg)
        if qual.get('status')!='QUALIFIED':atomic(folder/'status.json',dict(status='NOT_QUALIFIED',reason=qual.get('reason','LLMLingua not qualified'),generation_started=False));return []
        immutable(folder/'compressor_identity.json',dict(qualification_hash=digest(qual)))
    try:
        for index,e in enumerate(xs):
            for retriever in sorted({'BM25' if m in ('QBGF_BM25','BM25_QWEN_REFERENCE') else 'SFR' for m in methods}):
                needed=[m for m in methods if ('BM25' if m in ('QBGF_BM25','BM25_QWEN_REFERENCE') else 'SFR')==retriever]
                from .execution import budgets as allowed_budgets
                def budgets_for(method):return allowed_budgets(cfg,phase,method,core_final,real_smoke)
                # Completed preparation skips ALL retrieval/embedding work for this query.
                if all(store.get([e['dataset'],e['query_id'],m,b]) is not None for m in needed for b in budgets_for(m)):continue
                pool,sent,props,ps,ss,lineage=lab.prepare(e,retriever)
                for m in needed:
                    for b in budgets_for(m):
                        key=[e['dataset'],e['query_id'],m,b]
                        if store.get(key) is not None:continue
                        if m=='LLMLINGUA2':
                            if synthetic:sel=choose(pool,sent,props,ps,ss,tok,b,'C2');sel.update(method=m,synthetic_compressor_stub=True)
                            else:
                                from .qualification import compress_llm
                                sel=compress_llm(cfg,serialize(pool,tok)['text'],b,tok)
                        elif m in ('STRONG_SFR_RAG','BM25_QWEN_REFERENCE'):
                            sel=serialize(pool,tok);sel.update(method=m,budget='NATURAL',unused_budget=None,selected_unit_ids=[p['unit_id'] for p in sel['units']],selection_rank_ids=[p['unit_id'] for p in pool],choices=[],evidence_hash=digest(sel['evidence_ids']),source_support=1.)
                        else:sel=choose(pool,sent,props,ps,ss,tok,b,'QBGF_FULL' if m=='QBGF_BM25' else m,legacy);sel['method']=m
                        validate_spans(sel['units'],e['inference']['documents']);sel.update(lineage=lineage,raw_reference_tokens=serialize(pool,tok)['evidence_tokens'])
                        store.put(key,dict(selection=sel,example=e if synthetic else {k:v for k,v in e.items() if k not in ('inference','document_text')},task_type=e['task_type'],question=e['question']))
            print(f'W12-{phase} prepared {index+1}/{len(xs)}; completed selections are reused',flush=True)
    finally:lab.close()
    return store.rows()

def evaluate_result(cfg,e,sel,result,baseline):
    from src.week11.evaluation import evaluate
    from .statistics import behavioral,compression
    metrics=evaluate(result['prediction'],e)
    if e['task_type']=='verification':metrics['correct']=int(result['prediction']==e['evaluation']['gold_label'])
    else:metrics['correct']=None
    from src.week11.evaluation import evidence_metrics
    evmetrics=evidence_metrics(sel['units'],e) if sel['source_support'] is not None else dict(any_gold_recall=None,all_gold_recall=None,evidence_coverage=None,gold_annotation_count=None,unmapped_gold=None)
    raw=result.get('raw_prediction',result['prediction']);base=baseline or dict(score=metrics['task_score'],prediction=result['prediction'])
    return dict(dataset=e['dataset'],query_id=e['query_id'],paper_id=e['paper_id'],cluster_id=e['cluster_id'],split='VALIDATION' if cfg.get('split_protocol') else 'development',answer_type=e['evaluation'].get('answer_type','unknown'),method=sel['method'],budget=sel['budget'],prediction=result['prediction'],raw_prediction=raw,gold=e['evaluation']['gold_label'] if e['task_type']=='verification' else e['evaluation']['gold_answer'],**metrics,**behavioral(metrics['task_score'],base['score'],e['task_type']=='verification',cfg['help_epsilon']),answer_changed=result['prediction']!=base['prediction'],format_compliant=result['prediction'] in cfg['labels'] if e['task_type']=='verification' else None,evidence_tokens=sel['evidence_tokens'],unused_budget=sel['unused_budget'],budget_utilization=sel['evidence_tokens']/sel['budget'] if isinstance(sel['budget'],int) and sel['budget'] else None,compression_ratio=compression(sel['evidence_tokens'],sel['raw_reference_tokens']),compression_reference='complete frozen retrieved top10 source pool under additive block serialization',selected_unit_ids=sel['selected_unit_ids'],choices=sel['choices'],evidence_hash=sel['evidence_hash'],source_hash=sel['lineage']['source_hash'],candidate_ids=sel['lineage']['candidate_ids'],candidate_content_hashes=sel['lineage']['candidate_content_hashes'],source_support=sel['source_support'],preparation_seconds=sel['lineage'].get('preparation_seconds'),selection_seconds=sel.get('selection_seconds'),compression_seconds=sel.get('compression_seconds'),embedding_original_compute_seconds=sel['lineage'].get('embedding_original_compute_seconds'),embedding_lookup_compute_seconds=sel['lineage'].get('embedding_lookup_compute_seconds'),**evmetrics,end_to_end_latency=None,num_lm_calls=1,cumulative_prompt_tokens=result['prompt_tokens'],**{k:v for k,v in result.items() if k not in ('prediction','raw_prediction')})

def generate(cfg,phase,synthetic=False,real_smoke=False,core_final=True):
    from .reader import Reader
    from src.week11.readers import FixtureReader
    folder=directory(cfg,phase,synthetic,real_smoke,core_final);prepared=Store(folder/'prepared',identity(cfg,dict(phase=phase,synthetic=synthetic,real_smoke=real_smoke))).rows()
    variants=['BASE_NF4','LORA_NF4'] if phase=='G' else ['BASE_FP16']
    adapter_hash=None
    if phase=='G' and not synthetic:
        receipt=json.loads((root(cfg)/'lora/complete.json').read_text());adapter_hash=receipt['adapter_hash']
        if adapter_hash!=digest({p.name:sha(p) for p in (root(cfg)/'lora/adapter').glob('*') if p.is_file()}):raise ValueError('LoRA adapter changed after training receipt')
    store=Store(folder/'rows',identity(cfg,dict(phase=phase,synthetic=synthetic,real_smoke=real_smoke,adapter_hash=adapter_hash)))
    gen=Store(root(cfg)/(('smoke/generation/'+('synthetic/' if synthetic else 'real/')+identity(cfg)['code_hash'][:12]) if synthetic or real_smoke else 'generation'),identity(cfg,dict(synthetic=synthetic,real_smoke=real_smoke,adapter_hash=adapter_hash)))
    xs={ (e['dataset'],e['query_id']):e for e in examples(cfg,synthetic,3 if synthetic or real_smoke else None)}
    for variant in variants:
        reader=None
        try:
            for n,item in enumerate(prepared):
                sel=item['selection'];e=xs[item['example']['dataset'],item['example']['query_id']]
                if phase=='G' and not synthetic:
                    cpath=directory(cfg,'C')/'prepared';cstore=Store(cpath,json.loads((cpath/'identity.json').read_text()));control=cstore.get([e['dataset'],e['query_id'],sel['method'],sel['budget']])
                    if control is None or control['selection']['evidence_hash']!=sel['evidence_hash']:raise ValueError('LoRA evidence differs from frozen base-Qwen source selection')
                key=[e['dataset'],e['query_id'],sel['method'],sel['budget'],variant]
                if store.get(key) is not None:continue
                if reader is None:reader=FixtureReader(dict(readers=['Qwen']),'Qwen') if synthetic else Reader(cfg,variant,root(cfg)/'lora/adapter' if variant=='LORA_NF4' else None)
                # Explicit no-evidence control under exactly the same reader/parser/prompt.
                base_sel=dict(sel,text='',evidence_ids=[],method='NONE',evidence_tokens=0);basekey=[e['dataset'],e['query_id'],digest([]),variant]
                base=gen.get(basekey)
                def call(s):
                    if synthetic:
                        r=reader.generate(e['question'],s['text'],e['task_type']);r.update(reader=variant,raw_prediction=r['prediction'],model_hash='SYNTHETIC',prompt_hash=digest([e['question'],s['evidence_ids']]))
                        if e['task_type']=='verification':r['prediction']=parse_label(r['prediction'],cfg['labels'])
                        return r
                    return reader.generate(e,s)
                if base is None:base=call(base_sel);gen.put(basekey,base)
                gkey=[e['dataset'],e['query_id'],digest(sel['evidence_ids']),variant];result=gen.get(gkey);physical=result is None
                if result is None:result=call(sel);gen.put(gkey,result)
                from src.week11.evaluation import evaluate
                b=dict(score=evaluate(base['prediction'],e)['task_score'],prediction=base['prediction']);row=evaluate_result(cfg,e,sel,result,b);row.update(synthetic=synthetic,real_smoke=real_smoke,physical_lm_calls=int(physical),identity_hash=digest(store.identity),config_hash=digest(cfg),code_hash=store.identity['code_hash']);store.put(key,row)
                if (n+1)%10==0:print(f'W12-{phase} {variant} saved {n+1}/{len(prepared)} rows',flush=True)
        finally:
            if reader:reader.close()
    atomic(folder/'status.json',dict(complete=len(store.rows())==len(prepared)*len(variants),rows=len(store.rows()),expected=len(prepared)*len(variants),synthetic=synthetic,real_smoke=real_smoke,full_generation=not synthetic and not real_smoke))
    return store.rows()

def execute(phase,cfg=None,full=False,smoke=True,explicit=False,limit=3,real_smoke=False,core_final=True):
    cfg=cfg or load();authorize(full,smoke,explicit,limit)
    if phase=='0':
        from .data import audit
        return audit(cfg)
    if phase in ('H','I'):
        from .analysis import analyze
        return analyze(cfg,synthetic=smoke)
    if phase=='F':
        if full:raise ValueError('Use the dedicated manual training entrypoint after corpus inspection')
        from .lora import tiny_smoke
        return tiny_smoke(cfg)
    if phase=='E' and not full:
        from .qualification import qualify_llm
        return qualify_llm(cfg,allow_download=False,smoke=True)
    if phase=='B' and not real_smoke and not full:
        from .qualification import resources,inventory
        result=dict(resources=resources(cfg),baseline_inventory=inventory(cfg));atomic(root(cfg)/'qualification/execution'/identity(cfg)['code_hash'][:12]/'baselines.json',result)
    if full:require_full(cfg,phase)
    if not full and not smoke: # Read only. Do not generate from hidden defaults.
        p=directory(cfg,phase,core_final=core_final)/'rows';return Store(p,json.loads((p/'identity.json').read_text())).rows() if (p/'identity.json').exists() else []
    if real_smoke and (not smoke or full):raise ValueError('Real smoke requires smoke-only mode')
    synthetic=not full and not real_smoke
    if synthetic:prepare(cfg,phase,True,False,limit,core_final);return generate(cfg,phase,True,False,core_final)
    # Encoder and generator use separate OS processes, releasing GPU context at exit.
    command=[str(ROOT/'sara_env/bin/python'),str(ROOT/'scripts/week12/run.py'),'--phase',phase]
    if not core_final:command.append('--include-secondary')
    mode=['--real-smoke','--limit',str(limit)] if real_smoke else ['--full','--explicit']
    with lock(directory(cfg,phase,False,real_smoke,core_final)/'run.lock'):
        for stage in ['prepare','generate']:subprocess.run(command+mode+['--internal-stage',stage],cwd=ROOT,check=True)
    path=directory(cfg,phase,False,real_smoke,core_final)/'rows';return Store(path,json.loads((path/'identity.json').read_text())).rows()
