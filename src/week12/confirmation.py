"""Week13-only, final-freeze-gated confirmation. No QASPER TEST route exists."""
import json
from .protocol import ROOT,root,identity,digest,sha,Store,atomic,guard
from .splits import read_freeze,source_paths

def final_freeze(cfg):
    guard();path=root(cfg)/'W13_FINAL_METHOD_FREEZE.json'
    if not path.exists():raise RuntimeError('Week13 blocked: finish VAL and freeze final system in notebook I')
    f=json.loads(path.read_text());split=read_freeze(cfg)
    if f['identity']!=identity(cfg) or f['split_hash']!=split['split_hash']:raise ValueError('Scientific identity changed after final freeze')
    expected={d:[r['query_id'] for r in p['CONFIRM']] for d,p in split['datasets'].items()}
    if f['confirmation_ids']!=expected:raise ValueError('Final confirmation IDs differ from immutable split')
    if f['retain_lora']:
        h=digest({p.name:sha(p) for p in (root(cfg)/'lora/adapter').glob('*') if p.is_file()})
        if h!=f['adapter_hash']:raise ValueError('Final adapter changed')
    return f,split,sha(path)

def examples(cfg):
    f,split,_=final_freeze(cfg);paths=source_paths(cfg);xs=[]
    from src.week11.datasets import hotpot,scifact,read_jsonl
    from .data import native_qasper
    for d,p in split['datasets'].items():
        wanted={r['query_id']:r for r in p['CONFIRM']}
        if not wanted:continue
        if d=='QASPER':
            if not split['notes'][d]['fresh_confirmation_available']:raise ValueError('QASPER has no fresh confirmation')
            group=native_qasper(paths['QASPER_train'],allowed_ids=set(wanted))
        elif d=='HotpotQA':group=hotpot([r for r in json.loads(paths['HotpotQA_val'].read_text()) if str(r['_id']) in wanted],'dev')
        else:group=scifact([r for r in read_jsonl(paths['SciFact_val']) if str(r['id']) in wanted],read_jsonl(paths['SciFact_corpus']),'dev')
        if {e['query_id'] for e in group}!=set(wanted):raise ValueError('Missing frozen confirmation IDs')
        for e in group:e['cluster_id']=wanted[e['query_id']]['group'];e['partition']='CONFIRM'
        xs+=group
    return xs

def methods(cfg,f):
    pairs=f.get('confirmation_methods')
    if not pairs:raise ValueError('Final freeze must enumerate applicable confirmation methods')
    return [tuple(p) for p in pairs]

def run(cfg,stage):
    f,split,fh=final_freeze(cfg);folder=ROOT/'results/week13'/cfg['schema'];ident=identity(cfg,dict(final_freeze=fh,split_hash=split['split_hash'],data_role='CONFIRM'));store=Store(folder/'prepared',ident);xs=examples(cfg)
    if stage=='prepare':
        from .laboratory import Lab
        from .algorithm import choose,serialize,validate_spans
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(cfg['models']['Qwen']['id'],revision=cfg['models']['Qwen']['revision'],local_files_only=True);lab=Lab(cfg,tok)
        from src.week11.readers import tokenizers
        from src.week11.protocol import load as oldcfg
        legacy=tokenizers(oldcfg())
        try:
            for e in xs:
                for backbone in ['SFR','BM25']:
                    pairs=[(m,r) for m,r in methods(cfg,f) if ('BM25' if m=='BM25_QWEN_REFERENCE' else 'SFR')==backbone]
                    def budgets(m):return ['NATURAL'] if m in ('STRONG_SFR_RAG','BM25_QWEN_REFERENCE') else cfg['budgets']
                    if all(store.get([e['dataset'],e['query_id'],m,b,r]) is not None for m,r in pairs for b in budgets(m)):continue
                    pool,sent,props,ps,ss,lineage=lab.prepare(e,backbone);raw=serialize(pool,tok)
                    for m,r in pairs:
                        for b in budgets(m):
                            key=[e['dataset'],e['query_id'],m,b,r]
                            if store.get(key) is not None:continue
                            if b=='NATURAL':sel=dict(raw,method=m,budget=b,unused_budget=None,choices=[],selected_unit_ids=[u['unit_id'] for u in pool],source_support=1.,evidence_hash=digest([raw['evidence_ids'],raw['text']]))
                            elif m=='LLMLINGUA2':
                                from .qualification import compress_llm
                                sel=compress_llm(cfg,raw['text'],b,tok)
                            else:sel=choose(pool,sent,props,ps,ss,tok,b,m,legacy)
                            if sel['source_support'] is not None:validate_spans(sel['units'],e['inference']['documents'])
                            sel.update(lineage=lineage,raw_reference_tokens=raw['evidence_tokens']);store.put(key,dict(dataset=e['dataset'],query_id=e['query_id'],reader=r,selection=sel))
                lab.units_cache.clear();lab.ranks.clear()
        finally:lab.close()
        atomic(folder/'plan.json',dict(n_queries=len(xs),prepared_rows=len(store.rows()),confirmation_ids=f['confirmation_ids'],final_freeze=fh,QASPER_TEST_used=False,canonical_sara='NOT_APPLICABLE_NO_FRESH_QASPER_CONFIRMATION' if not f['confirmation_ids']['QASPER'] else 'SEPARATE_SPECIALIST_REQUIRED'))
        return
    from .reader import Reader
    from .runner import evaluate_result
    from src.week11.evaluation import evaluate
    output=Store(folder/'rows',ident);cache=Store(folder/'generation',ident);lookup={(e['dataset'],e['query_id']):e for e in xs};reader=None;active=None
    try:
        for row in sorted(store.rows(),key=lambda r:r['reader']):
            sel=row['selection'];e=lookup[row['dataset'],row['query_id']];variant=row['reader'];key=[e['dataset'],e['query_id'],sel['method'],sel['budget'],variant]
            if output.get(key) is not None:continue
            if active!=variant:
                if reader:reader.close()
                reader=Reader(cfg,variant,root(cfg)/'lora/adapter' if variant=='LORA_NF4' else None);active=variant
            def answer(selection):
                k=[e['dataset'],e['query_id'],variant,digest(selection['evidence_ids'])];old=cache.get(k)
                if old is not None:return old
                result=reader.generate(e,selection);cache.put(k,result);return result
            baseline=answer(dict(sel,evidence_ids=[],text=''));result=answer(sel);base=dict(score=evaluate(baseline['prediction'],e)['task_score'],prediction=baseline['prediction']);r=evaluate_result(cfg,e,sel,result,base);r.update(split='CONFIRM',final_freeze_hash=fh);output.put(key,r)
    finally:
        if reader:reader.close()
    if len(output.rows())!=len(store.rows()):raise ValueError('Incomplete confirmation; resume without regenerating completed rows')
    atomic(folder/'complete.json',dict(complete=True,n_rows=len(output.rows()),final_freeze_hash=fh,QASPER_TEST_used=False))
