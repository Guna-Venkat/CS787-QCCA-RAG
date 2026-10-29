"""Query-conditioned exact MCKP over additive Qwen token-ID source blocks.

No labels/answers enter these functions. DP is exact for the defined serialized
block costs and relevance surrogate, not for downstream answer utility.
"""
import math,time
import numpy as np
from .protocol import digest
KS=(1,2,4,8)
METHODS=['NONE','RAW_STRICT','RAW_SKIP','FIXED_SENTENCE','C2','H3','QBGF_FULL','QBGF_NO_QUERY','QBGF_SENTENCE_ONLY','QBGF_PASSAGE_ONLY','QBGF_NO_MASS','QBGF_GREEDY','QBGF_BM25','LLMLINGUA2']
def softmax(values,tau=5):
    a=np.asarray(values,dtype=np.float64)
    if not np.isfinite(a).all():raise ValueError('Nonfinite relevance')
    if not len(a):return []
    a=tau*(a-a.max());p=np.exp(a);return (p/p.sum()).tolist()
def cosine(q,x):
    q=np.asarray(q,dtype=float);x=np.asarray(x,dtype=float)
    if not np.isfinite(q).all() or not np.isfinite(x).all():raise ValueError('Nonfinite embedding')
    return (x@q/np.maximum(np.linalg.norm(x,axis=-1)*np.linalg.norm(q),1e-12)).tolist()
def order(u):return (tuple(u['document_order']),u['start_char'],u['unit_id'])
def validate_pool(pool):
    seen=set()
    for p in pool:
        if p['unit_id'] in seen:raise ValueError('Repeated candidate passage')
        seen.add(p['unit_id'])
        if any(p['document_id']==q['document_id'] and p['start_char']<q['end_char'] and q['start_char']<p['end_char'] for q in pool if q is not p):raise ValueError('Overlapping passages: canonicalize before optimization')
def validate_spans(units,documents):
    docs={d['document_id']:d['text'] for d in documents}
    for u in units:
        if docs[u['document_id']][u['start_char']:u['end_char']]!=u['source_text']:raise ValueError('Source substring/offset mismatch')
    for i,u in enumerate(units):
        if any(u['document_id']==v['document_id'] and u['start_char']<v['end_char'] and v['start_char']<u['end_char'] for v in units[:i]):raise ValueError('Repeated/overlapping selected source')
def block(units,tok):
    units=sorted(units,key=order);text='\n\n'.join(u['source_text'] for u in units)
    ids=tok.encode(text+'\n\n',add_special_tokens=False) if units else []
    return dict(units=units,text=text,ids=list(ids),cost=len(ids))
def serialize(units,tok):
    grouped={}
    for u in units:grouped.setdefault(u['passage_id'],[]).append(u)
    blocks=[block(g,tok) for g in grouped.values()];blocks.sort(key=lambda b:order(b['units'][0]))
    all_units=[u for b in blocks for u in b['units']]
    return dict(text='\n\n'.join(u['source_text'] for u in all_units),evidence_ids=[i for b in blocks for i in b['ids']],units=all_units,evidence_tokens=sum(b['cost'] for b in blocks),serialization='additive-token-id-blocks-v1')
def options(pool,sentences,passage_scores,sentence_scores,tok,method='QBGF_FULL',tau_p=5,tau_s=5):
    validate_pool(pool)
    if len(passage_scores)!=len(pool):raise ValueError('Passage score alignment')
    noquery=method=='QBGF_NO_QUERY';R=[1/len(pool)]*len(pool) if noquery and pool else softmax(passage_scores,tau_p)
    groups=[]
    for i,p in enumerate(pool):
        ss=sorted([s for s in sentences if s['passage_id']==p['unit_id']],key=order)
        if len({s['unit_id'] for s in ss})!=len(ss):raise ValueError('Duplicate sentence identity')
        scores=[sentence_scores[s['unit_id']] for s in ss] if not noquery else [0.]*len(ss)
        mass=softmax(scores,tau_s);ix=sorted(range(len(ss)),key=lambda j:(-scores[j],order(ss[j])))
        opts=[dict(name='DROP',cost=0,utility=0.,units=[],actual_k=0,mass=0.,ids=[])]
        if method!='QBGF_PASSAGE_ONLY':
            seen=set()
            for k in KS:
                selected=[ss[j] for j in ix[:k]];signature=tuple(sorted(s['unit_id'] for s in selected))
                if not selected or signature in seen:continue
                seen.add(signature);m=min(k,len(ss))/len(ss) if method=='QBGF_NO_MASS' else sum(mass[j] for j in ix[:k])
                b=block(selected,tok);opts.append(dict(name=f'SENT_{k}',cost=b['cost'],utility=R[i]*m,units=b['units'],actual_k=len(selected),mass=m,ids=b['ids'],selection_rank_ids=[u['unit_id'] for u in selected]))
        if method!='QBGF_SENTENCE_ONLY':
            b=block([p],tok);opts.append(dict(name='FULL',cost=b['cost'],utility=R[i],units=[p],actual_k=None,mass=1.,ids=b['ids']))
        # Keep equivalent/dominated options visible for inspection; deterministic DP handles ties.
        for x in opts:x.update(passage_id=p['unit_id'],passage_weight=R[i])
        groups.append(opts)
    return groups

def solve(groups,budget,greedy=False):
    if not isinstance(budget,int) or budget<0:raise ValueError('Integer nonnegative budget required')
    if greedy:
        chosen=[0]*len(groups);spent=0;value=0.
        while True:
            upgrades=[]
            for i,g in enumerate(groups):
                old=g[chosen[i]]
                for j,x in enumerate(g):
                    dc=x['cost']-old['cost'];dv=x['utility']-old['utility']
                    if dv>1e-12 and dc>=0 and spent+dc<=budget:upgrades.append((-dv/max(dc,1e-12),dc,i,j,dv))
            if not upgrades:break
            _,dc,i,j,dv=min(upgrades);chosen[i]=j;spent+=dc;value+=dv
        return [g[j] for g,j in zip(groups,chosen)]
    # Exact cost states; utility ties choose lexicographically smaller option indices.
    states={0:(0.,())}
    for g in groups:
        nxt={}
        for cost,(value,path) in states.items():
            for j,x in enumerate(g):
                c=cost+x['cost'];v=value+x['utility'];p=path+(j,)
                if c>budget:continue
                old=nxt.get(c)
                if old is None or v>old[0]+1e-12 or (abs(v-old[0])<=1e-12 and p<old[1]):nxt[c]=(v,p)
        states=nxt
    if not states:raise ValueError('No feasible options; DROP required')
    cost,(value,path)=min(states.items(),key=lambda x:(-round(x[1][0],12),x[0],x[1][1]))
    return [g[j] for g,j in zip(groups,path)]

def choose(pool,sentences,props,ps,ss,tok,budget,method,legacy_tokenizers=None):
    started=time.perf_counter();validate_pool(pool);choices=[];ranking=[]
    if method.startswith('QBGF'):
        groups=options(pool,sentences,ps,ss,tok,method);choices=solve(groups,budget,method=='QBGF_GREEDY');units=[u for x in choices for u in x['units']]
        ranking=[uid for x in choices for uid in x.get('selection_rank_ids',[u['unit_id'] for u in x['units']])];result=serialize(units,tok)
        assert result['evidence_tokens']==sum(x['cost'] for x in choices)
    elif method=='RAW_STRICT':
        from src.week11.budgets import allocate
        x=allocate(pool,budget,legacy_tokenizers or {'Qwen':tok});units=x['units'];text=x['text'];ids=tok.encode(text,add_special_tokens=False)
        result=dict(text=text,evidence_ids=list(ids),units=units,evidence_tokens=len(ids),serialization='historical-whole-text');ranking=x['selection_rank_ids']
    else:
        if method=='NONE':rank=[]
        elif method=='RAW_SKIP':rank=pool
        elif method in ('FIXED_SENTENCE','C2'):rank=sorted(sentences,key=lambda u:(-ss[u['unit_id']],order(u)))[:8 if method=='C2' else len(sentences)]
        elif method=='H3':
            head={p['unit_id'] for p in pool[:3]};rank=sorted([u for u in sentences if u['passage_id'] in head],key=lambda u:(-ss[u['unit_id']],order(u)))+sorted([u for u in props if u['passage_id'] not in head],key=lambda u:(-ss[u['unit_id']],order(u)))
        else:raise ValueError('Unknown/externally prepared method '+method)
        units=[]
        for u in rank:
            trial=serialize(units+[u],tok)
            if trial['evidence_tokens']>budget:
                if method in ('C2','H3'):break # explicit ported rank-prefix control
                continue
            units.append(u);ranking.append(u['unit_id'])
        result=serialize(units,tok)
    if result['evidence_tokens']>budget:raise AssertionError('Evidence budget violated')
    result.update(method=method,budget=budget,unused_budget=budget-result['evidence_tokens'],selection_rank_ids=ranking,selected_unit_ids=[u['unit_id'] for u in result['units']],choices=[{k:v for k,v in x.items() if k not in ('units','ids')} for x in choices],selection_seconds=time.perf_counter()-started,source_support=1.,evidence_hash=digest([result['evidence_ids'],result['text']]))
    return result
