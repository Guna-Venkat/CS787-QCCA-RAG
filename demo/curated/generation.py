"""One frozen Qwen reader; exact-prompt reuse and durable LOO outcomes."""
import time
from .protocol import METHODS,BUDGETS,grid,store,digest

def run(c,m,attribution=False):
    from .data import examples
    from src.week12.reader import Reader
    from src.week12.algorithm import serialize
    from src.week11.evaluation import evaluate,help_hurt,evidence_metrics
    xs=examples(c,m);prepared=store(m,'prepared');results=store(m,'results');deletions=store(m,'attribution');calls=store(m,'generation_cache');reader=None
    def generate(e,selection):
        nonlocal reader
        key=[e['dataset'],e['query_id'],digest([e['question'],e['task_type'],selection['evidence_ids']])]
        value=calls.get(key)
        if value is not None:return value,True
        if reader is None:reader=Reader(c,'BASE_FP16')
        value=reader.generate(e,selection);calls.put(key,value);return value,False
    started=time.time();keys=grid(m)
    try:
        for n,tup in enumerate(keys,1):
            key=list(tup);d,q,rt,b,a=key;e=xs[d,q];item=prepared.get(key)
            if item is None:raise ValueError('Missing preparation for '+str(key))
            sel=item['selection'];base_sel=dict(sel,text='',evidence_ids=[],units=[],method='NONE',evidence_tokens=0)
            base,_=generate(e,base_sel)
            full=results.get(key)
            if full is None:
                result,reused=generate(e,sel);metric=evaluate(result['prediction'],e);bm=evaluate(base['prediction'],e)
                full=dict(dataset=d,query_id=q,retriever=rt,budget=b,method=a,**result,**metric,baseline_answer=base['prediction'],baseline_score=bm['task_score'],
                          **help_hurt(metric['task_score'],bm['task_score'],e['task_type']=='verification'),evidence_tokens=sel['evidence_tokens'],budget_utilization=sel['evidence_tokens']/b,
                          source_hash=item['lineage']['source_hash'],selected_unit_ids=sel['selected_unit_ids'],evidence_hash=digest([sel['evidence_ids'],sel['text']]),
                          selected_evidence_metrics=evidence_metrics(sel['units'],e),conceptual_lm_calls=1,exact_prompt_cache_reused=reused,presentation_scope='NEW CURATED DEMO RUN; not historical study scores')
                results.put(key,full)
            if attribution:
                # Serializing remaining units is tokenizer work; only load a tokenizer if needed.
                pending=[(i,u) for i,u in enumerate(sel['units']) if deletions.get(key+[u['unit_id']]) is None]
                tok=None
                if pending:
                    from transformers import AutoTokenizer
                    tok=reader.tok if reader else AutoTokenizer.from_pretrained(c['models']['Qwen']['id'],revision=c['models']['Qwen']['revision'],local_files_only=True)
                for i,u in pending:
                    remaining=serialize([v for j,v in enumerate(sel['units']) if j!=i],tok)
                    ablated=dict(sel,**remaining);result,reused=generate(e,ablated);metric=evaluate(result['prediction'],e)
                    importance=float(full['task_score']-metric['task_score']);eps=0. if e['task_type']=='verification' else .01
                    category='HELPFUL' if importance>eps else 'HARMFUL' if importance<-eps else 'NEUTRAL'
                    deletions.put(key+[u['unit_id']],dict(unit_id=u['unit_id'],original_answer=full['prediction'],ablated_answer=result['prediction'],original_score=full['task_score'],ablated_score=metric['task_score'],importance=importance,category=category,exact_flip=result['prediction']!=full['prediction'],prompt_hash=result['prompt_hash'],generated_tokens=result['generated_tokens'],conceptual_lm_calls=1,exact_prompt_cache_reused=reused))
            if n%10==0 or n==len(keys):print(f'Curated demo {"LOO" if attribution else "answers"} {n}/{len(keys)} conditions; elapsed={time.time()-started:.0f}s',flush=True)
    finally:
        if reader:reader.close()
