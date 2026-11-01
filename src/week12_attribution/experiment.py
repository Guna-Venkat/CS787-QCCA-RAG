"""Frozen source units and immutable perturbations; generator is manual opt-in only."""
import json,collections,itertools,time
import numpy as np
from src.week12.protocol import ROOT,root,digest,Store,immutable,atomic
from src.week12.algorithm import serialize,order,validate_spans,cosine
from .protocol import directory,study_identity,freeze,original,cached_vectors

def gold_overlap(unit,example):
    spans=example['evaluation'].get('gold_spans',[])
    if not spans:return None
    valid=[s for s in spans if not s.get('unmapped') and (s.get('alternatives') or s.get('end_char',0)>s.get('start_char',0))]
    hit=any(unit['document_id']==s['document_id'] and any(unit['start_char']<b and a<unit['end_char'] for a,b in s.get('alternatives',[[s['start_char'],s['end_char']]])) for s in valid)
    return True if hit else False if len(valid)==len(spans) else None

def perturb(units,keep,tok):
    kept=[u for i,u in enumerate(units) if i in set(keep)]
    return serialize(kept,tok)

def causal_order(deltas,units):return sorted(range(len(units)),key=lambda i:(-deltas[i],order(units[i])))
def relevance_order(units):return sorted(range(len(units)),key=lambda i:(-units[i]['query_relevance'],order(units[i])))
def retention(full,none,subset):return (subset-none)/(full-none) if full>none else None

def prepare(cfg):
    obj=freeze(cfg);ident=study_identity(cfg);out=directory(cfg);st=Store(out/'prepared',ident)
    from src.week12.splits import load_examples
    from src.week12.laboratory import Lab
    from transformers import AutoTokenizer
    wanted={(d,r['query_id']) for d,rs in obj['selected'].items() for r in rs}
    tok=AutoTokenizer.from_pretrained(cfg['models']['Qwen']['id'],revision=cfg['models']['Qwen']['revision'],local_files_only=True)
    lab=Lab(cfg,tok);evidence_sets=[]
    try:
        for e in load_examples(cfg,'VAL'):
            if (e['dataset'],e['query_id']) not in wanted:continue
            d,q=e['dataset'],e['query_id'];all_units=lab.units(e);passages=[u for u in all_units if u['unit_type']=='PASSAGE']
            for b in (256,1024):
                for method in ('QBGF_FULL',obj['comparators'][d+':'+str(b)]):
                    key=[d,q,method,b]
                    if st.get(key) is not None:continue
                    item=original(cfg,d,q,method,b,'prepared');sel=item['selection'];units=sel['units'];baseline=original(cfg,d,q,method,b);none=original(cfg,d,q,'NONE',b)
                    validate_spans(units,e['inference']['documents'])
                    full=serialize(units,tok)
                    if full['evidence_ids']!=sel['evidence_ids'] or digest(e['inference'])!=baseline['source_hash']:raise ValueError('Original evidence/source identity does not reproduce')
                    poolids=sel['lineage']['candidate_ids'];poolmap={u['unit_id']:u for u in passages};pool=[poolmap[i] for i in poolids]
                    if [digest(u['source_text']) for u in pool]!=sel['lineage']['candidate_content_hashes']:raise ValueError('Frozen candidate contents changed')
                    if method=='BM25_PLAIN_RAG':
                        from src.week9.retrieval import bm25
                        ps=dict(zip([p['unit_id'] for p in passages],map(float,bm25(e['question'],[p['source_text'] for p in passages]))))
                        fine=[u for u in all_units if u['unit_type']=='SENTENCE' and u['passage_id'] in poolids]+[u for u in all_units if u['unit_type']=='PROPOSITION' and u['passage_id'] in poolids]
                        ss=dict(zip([u['unit_id'] for u in fine],map(float,bm25(e['question'],[u['source_text'] for u in fine]))))
                    else:
                        qv=cached_vectors(cfg,[e['question']],'query')[0]
                        ps=dict(zip(poolids,cosine(qv,cached_vectors(cfg,[u['source_text'] for u in pool]))))
                        ss=dict(zip([u['unit_id'] for u in units],cosine(qv,cached_vectors(cfg,[u['source_text'] for u in units])))) if units else {}
                    choice={r['passage_id']:r for r in sel['choices']};enriched=[]
                    for i,u in enumerate(units):
                        parent=u['unit_id'] if u['unit_type']=='PASSAGE' else u['passage_id'];ch=choice.get(parent,{})
                        enriched.append(dict(u,position=i,evidence_token_cost=len(tok.encode(u['source_text'],add_special_tokens=False)),standalone_block_tokens=serialize([u],tok)['evidence_tokens'],parent_retrieval_score=ps[parent],parent_retrieval_rank=poolids.index(parent)+1,query_relevance=ps[parent] if u['unit_type']=='PASSAGE' else ss[u['unit_id']],qbgf_option=ch.get('name'),qbgf_utility=ch.get('utility'),qbgf_passage_weight=ch.get('passage_weight'),gold_overlap=gold_overlap(u,e)))
                    slim={k:e[k] for k in ['dataset','query_id','paper_id','cluster_id','question','task_type','split']};slim['evaluation']={k:v for k,v in e['evaluation'].items() if k!='gold_spans'}
                    st.put(key,dict(example=slim,method=method,budget=b,units=enriched,full=full,original=baseline,none=none,source_hash=baseline['source_hash'],original_evidence_hash=baseline['evidence_hash']))
    finally:lab.close()
    rows=st.rows();assert len(rows)==600
    known=0;upper=0;unit_total=0;unique=set();gen=original_cache(cfg)
    for r in rows:
        n=len(r['units']);unit_total+=n;ms=[m for m in (1,2,4) if m<=n];upper+=n+min(6,n)+2*len(ms)
        sets=[set(range(n))-{i} for i in range(n)]+[{i} for i in relevance_order(r['units'])[:3]]
        for indices in sets:
            p=perturb(r['units'],indices,tok);key=generation_key(r['example'],p);signature=digest(key)
            if signature in unique:continue
            unique.add(signature);known+=gen.get(key) is not None
    plan=dict(queries=150,cells=len(rows),units=unit_total,original_reference_answers=len(rows),original_reference_cache_reuse_percent=100,no_evidence_reference_cache_reuse_percent=100,known_LOO_and_relevance_ONLY_unique=len(unique),known_actions_already_cached=known,known_action_cache_reuse_percent=100*known/len(unique) if unique else 100,perturbation_call_upper_bound_before_global_dedup=upper,remaining_call_upper_bound=upper,causal_ONLY_top3_unknown_until_LOO=True,minimum_free_disk_GiB=2,estimated_additional_disk_GiB_upper=max(.1,(upper*12000+len(rows)*150000)/1024**3),optional_third_method=False,optional_replacement=False,note='Upper bound includes LOO, union of relevance/causal top3 ONLY, and available top1/top2/top4 ONLY/REMOVE; duplicates, original/NONE and compatible C/D cache hits reduce calls. Cached inputs are not zero conceptual prompt cost.')
    immutable(out/'cost_plan.json',plan);return rows,plan

def original_cache(cfg):
    from .protocol import store
    p=root(cfg)/'generation';s=store(p)
    from src.week12.protocol import identity
    if s.identity!=identity(cfg,dict(synthetic=False,real_smoke=False,adapter_hash=None)):raise ValueError('Original generation identity mismatch')
    return s

def generation_key(e,p):return [e['dataset'],e['query_id'],digest(p['evidence_ids']),'BASE_FP16']
def changes(e,original,answer):
    from src.week11.evaluation import hotpot_normalize
    from src.week6.evaluation import exact_match_score
    normalize_answer=exact_match_score.__globals__.get('normalize_answer',hotpot_normalize)
    normalize=(lambda x:x.strip()) if e['dataset']=='SciFact' else normalize_answer if e['dataset']=='QASPER' else hotpot_normalize
    def correct(r):return int(r['task_score']==1) if e['dataset']=='SciFact' else int(r.get('exact_match',0)==1)
    return dict(exact_answer_changed=original['prediction']!=answer['prediction'],normalized_answer_changed=normalize(original['prediction'])!=normalize(answer['prediction']),score_changed=original['task_score']!=answer['task_score'],correctness_changed=correct(original)!=correct(answer),correctness_definition='strict label correctness' if e['dataset']=='SciFact' else 'exact-match correctness',output_label_changed=original['prediction']!=answer['prediction'] if e['dataset']=='SciFact' else None)

def execute(cfg,full=False,limit=3,smoke=False):
    if not full and not smoke:raise ValueError('Default path is synthetic tests only. Full manual opt-in required for real ablation.')
    from src.week12.reader import Reader
    from src.week11.evaluation import evaluate
    from transformers import AutoTokenizer
    import shutil
    if shutil.disk_usage(ROOT).free<2*1024**3:raise RuntimeError('At least 2 GiB free required before attribution generation')
    obj=None if smoke else freeze(cfg);ident=study_identity(cfg,smoke);f=directory(cfg,smoke)
    if smoke:
        from .synthetic import prepare as synthetic_prepare
        prepared,tok,old=synthetic_prepare(cfg)
    else:prepared=Store(f/'prepared',ident).rows()
    if not smoke and len(prepared)!=600:raise ValueError('Run notebook J to freeze and prepare exactly 150 queries')
    if not smoke:tok=AutoTokenizer.from_pretrained(cfg['models']['Qwen']['id'],revision=cfg['models']['Qwen']['revision'],local_files_only=True)
    results=Store(f/'perturbations',ident);gen=Store(f/'generation',ident)
    if not smoke:old=original_cache(cfg)
    reader=None
    def call(r,action,indices):
        nonlocal reader
        key=[r['example']['dataset'],r['example']['query_id'],r['method'],r['budget'],action]
        saved=results.get(key)
        if saved is not None:return saved
        evidence=perturb(r['units'],indices,tok);gkey=generation_key(r['example'],evidence);answer=old.get(gkey) or gen.get(gkey);physical=answer is None
        if answer is None:
            if smoke:
                from .synthetic import answer as fixture_answer
                answer=fixture_answer(r['example'],evidence)
            else:
                if reader is None:reader=Reader(cfg)
                answer=reader.generate(r['example'],evidence)
            gen.put(gkey,answer)
        evaluated=dict(answer,**evaluate(answer['prediction'],r['example']))
        record=dict(dataset=r['example']['dataset'],query_id=r['example']['query_id'],paper_id=r['example']['paper_id'],cluster_id=r['example']['cluster_id'],method=r['method'],budget=r['budget'],action=action,unit_indices=sorted(indices),evidence_tokens=evidence['evidence_tokens'],original_evidence_hash=r['original_evidence_hash'],source_hash=r['source_hash'],physical_lm_calls=int(physical and not smoke),synthetic=smoke,conceptual_lm_calls=1,**evaluated,**changes(r['example'],r['original'],evaluated))
        results.put(key,record);return record
    try:
        for idx,r in enumerate(prepared):
            units=r['units'];n=len(units);all_indices=set(range(n));fullscore=r['original']['task_score'];deltas=[]
            for i in range(n):deltas.append(fullscore-call(r,'LOO:'+str(i),all_indices-{i})['task_score'])
            ranked=causal_order(deltas,units)
            only=set(relevance_order(units)[:3])|set(ranked[:3])
            for i in sorted(only):call(r,'ONLY:'+str(i),{i})
            for m in (1,2,4):
                if m>n:continue
                top=set(ranked[:m]);call(r,'TOP_ONLY:'+str(m),top);call(r,'TOP_REMOVE:'+str(m),all_indices-top)
            immutable(f/'completed_cells'/(digest([r['example']['dataset'],r['example']['query_id'],r['method'],r['budget']])+'.json'),dict(n_units=n,causal_order=ranked,unit_importance=deltas,identity=digest(ident)))
            print(f'Attribution saved {idx+1}/{len(prepared)} cells; completed actions reused',flush=True)
    finally:
        if reader:reader.close()
    atomic(f/'status.json',dict(complete=True,cells=len(prepared),queries=3 if smoke else 150,validation_only=True,full_generation=not smoke,synthetic=smoke))
