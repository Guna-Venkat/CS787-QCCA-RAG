"""Pure outcome-free bottom-up marginal selection. All source units remain exact."""
import re,time
import numpy as np
from src.week12.algorithm import serialize,order,validate_pool
from src.week12.protocol import digest

def aspects(query):
    if not isinstance(query,str) or not query.strip():raise ValueError('A nonempty query is required')
    parts=re.split(r'[?;\n]+|,\s*(?:and\s+)?|\b(?:and|but|versus|whereas)\b',query,flags=re.I)
    result=[]
    for part in parts:
        part=' '.join(part.strip(' ,:.?!').split())
        if len(part.split())>=2 and part.lower() not in [r.lower() for r in result]:result.append(part)
    return result[:6] or [' '.join(query.split())]

def normalize(v):
    a=np.asarray(v,dtype=float)
    if a.ndim!=2 or not np.isfinite(a).all() or np.any(np.linalg.norm(a,axis=1)<1e-12):raise ValueError('Invalid/zero embeddings')
    return a/np.linalg.norm(a,axis=1,keepdims=True)

def signals(sentence_vectors,query_vector,aspect_vectors):
    s=normalize(sentence_vectors);q=normalize(np.asarray(query_vector).reshape(1,-1));a=normalize(aspect_vectors)
    return dict(relevance=np.clip(s@q[0],-1,1).tolist(),
                aspect_scores=np.clip(a@s.T,-1,1).tolist(),
                pairwise=np.clip(s@s.T,-1,1).tolist())

def marginal(index,selected,rel,asp,pair,parameters):
    covered=np.maximum(0,asp[:,sorted(selected)].max(axis=1)) if selected else np.zeros(len(asp))
    gain=float(np.maximum(0,asp[:,index]-covered).sum())
    red=float(pair[index,sorted(selected)].max()) if selected else 0.
    value=float(rel[index]+parameters['eta_cov']*gain-parameters['lambda_red']*red)
    return dict(gain=value,relevance=float(rel[index]),redundancy=red,novelty=1-red,coverage_gain=gain)

def sequential_full(missing,selected,rel,asp,pair,parameters):
    """Source-order hypothetical additions; each accepted addition updates E'."""
    hypothetical=set(selected);steps=[]
    for i in missing:
        m=marginal(i,hypothetical,rel,asp,pair,parameters)
        accepted=m['gain']>parameters['tau_context']
        steps.append(dict(index=i,accepted=accepted,**m))
        if accepted:hypothetical.add(i)
    return sum(x['gain'] for x in steps if x['accepted']),steps

def allocate(pool,sentences,scores,tok,budget,parameters):
    start=time.perf_counter();validate_pool(pool)
    ss=list(sentences);n=len(ss);rel=np.asarray(scores['relevance']);asp=np.asarray(scores['aspect_scores']);pair=np.asarray(scores['pairwise']).reshape(n,n)
    if rel.shape!=(n,) or pair.shape!=(n,n) or asp.ndim!=2 or asp.shape[1]!=n or not all(np.isfinite(x).all() for x in [rel,asp,pair]):
        raise ValueError('Global score alignment/finite check failed')
    if len({u['unit_id'] for u in ss})!=n:raise ValueError('Duplicate sentences')
    pmap={p['unit_id']:p for p in pool};indices={pid:[i for i,s in enumerate(ss) if s['passage_id']==pid] for pid in pmap}
    if any(s['passage_id'] not in pmap for s in ss):raise ValueError('Unknown parent')
    selected=set();full=set();history=[];accumulated={pid:[] for pid in pmap};trace=[]
    def units(chosen,whole):
        return [pmap[p] for p in sorted(whole)]+[ss[i] for i in sorted(chosen) if ss[i]['passage_id'] not in whole]
    current=serialize([],tok);stop='NO_CANDIDATES';upgrades=0;full_attempts=[];step_scores=[]
    while True:
        candidates=[];positive_over_budget=0
        attempt_start=len(full_attempts)
        for i,s in enumerate(ss):
            if i in selected:continue
            m=marginal(i,selected,rel,asp,pair,parameters)
            trial=serialize(units(selected|{i},full),tok);dc=trial['evidence_tokens']-current['evidence_tokens']
            if m['gain']>0 and trial['evidence_tokens']>budget:positive_over_budget+=1
            if trial['evidence_tokens']>budget or m['gain']<=0:continue
            ratio=m['gain']/dc if dc>0 else 1e99
            candidates.append(dict(kind='SENTENCE',index=i,parent=s['passage_id'],trial=trial,dc=dc,ratio=ratio,threshold=parameters['tau_stop'],**m))
        # FULL may enter only after evidence accumulation, never passage-first.
        for pid,ix in indices.items():
            if pid in full:continue
            missing=[i for i in ix if i not in selected]
            if not missing:continue
            if len(accumulated[pid])<parameters['min_parent_sentences']:
                if len(accumulated[pid])==1:
                    full_attempts.append(dict(step=len(trace),parent=pid,status='BLOCKED_ONE_SENTENCE',eligible=False))
                continue
            gain,hypothetical=sequential_full(missing,selected,rel,asp,pair,parameters)
            valuable=[x['index'] for x in hypothetical if x['accepted']]
            attempt=dict(step=len(trace),parent=pid,eligible=True,gain=gain,hypothetical=hypothetical)
            full_attempts.append(attempt)
            if not valuable or gain<=0:
                attempt['status']='REJECTED_LOW_GAIN';continue
            trial=serialize(units(selected|set(ix),full|{pid}),tok);dc=trial['evidence_tokens']-current['evidence_tokens']
            if trial['evidence_tokens']>budget:
                attempt['status']='REJECTED_COST';positive_over_budget+=1;continue
            ratio=gain/dc if dc>0 else 1e99
            threshold=max(parameters['tau_stop'],parameters['tau_expand'])
            attempt.update(status='REJECTED_THRESHOLD' if ratio<threshold else 'LOST_COMPETITION',dc=dc,
                           ratio=ratio if dc>0 else None)
            candidates.append(dict(kind='FULL',parent=pid,index=min(ix),trial=trial,dc=dc,ratio=ratio,gain=gain,
                                   valuable_sentence_ids=[ss[i]['unit_id'] for i in valuable],threshold=threshold))
        finite=[x['ratio'] for x in candidates if x['dc']>0]
        step_scores.append(dict(step=len(trace),candidate_count=len(candidates),positive_over_budget=positive_over_budget,
                                marginal_scores=finite,maximum=max(finite) if finite else None,
                                median=float(np.median(finite)) if finite else None))
        admissible=[c for c in candidates if c['ratio']>=c['threshold']]
        if not admissible:
            stop=('LOW_MARGINAL_VALUE' if candidates else 'BUDGET_EXHAUSTED' if positive_over_budget else
                  'ALL_EVIDENCE_EXPOSED' if len(selected)==n else 'NO_POSITIVE_MARGINAL');break
        best=min(admissible,key=lambda c:(-c['ratio'],-c['gain'],c['dc'],order(ss[c['index']]),c['kind']))
        if best['kind']=='FULL':
            selected.update(indices[best['parent']]);full.add(best['parent']);upgrades+=1
            for attempt in full_attempts[attempt_start:]:
                if attempt['parent']==best['parent']:attempt['status']='CHOSEN'
        else:
            selected.add(best['index']);accumulated[best['parent']].append(dict(unit_id=ss[best['index']]['unit_id'],marginal_gain=best['gain']))
            history.append(ss[best['index']]['unit_id'])
        current=best['trial'];record={k:v for k,v in best.items() if k not in ('trial','index','threshold')}
        if record['ratio']==1e99:record['ratio']=None;record['nonpositive_incremental_cost']=True
        record.update(step=len(trace)+1,selected_tokens=current['evidence_tokens'],unused_tokens=budget-current['evidence_tokens']);trace.append(record)
    if current['evidence_tokens']>budget:raise AssertionError('Budget exceeded')
    choices=[]
    for pid,ix in indices.items():
        count=len(selected&set(ix));state='FULL' if pid in full else 'MULTI_SENTENCE' if count>1 else 'SENTENCE' if count else 'DROP'
        choices.append(dict(passage_id=pid,state=state,selected_sentences=count,accumulated=accumulated[pid]))
    current.update(method='BEA_QBGF',budget=budget,unused_budget=budget-current['evidence_tokens'],
                   selected_unit_ids=[u['unit_id'] for u in current['units']],selection_rank_ids=history,
                   choices=choices,trace=trace,stop_reason=stop,fidelity_upgrades=upgrades,
                   full_attempts=full_attempts,step_scores=step_scores,
                   final_marginal_score=step_scores[-1]['maximum'],
                   selection_seconds=time.perf_counter()-start,source_support=1.,
                   evidence_hash=digest([current['evidence_ids'],current['text']]))
    return current

def behavior(selection,sentences,scores):
    # FULL covers every constituent source sentence for coverage/redundancy diagnostics.
    ids={u['unit_id'] for u in selection['units']};whole={u['unit_id'] for u in selection['units'] if u['unit_type']=='PASSAGE'}
    ix=[i for i,s in enumerate(sentences) if s['unit_id'] in ids or s['passage_id'] in whole]
    asp=np.asarray(scores['aspect_scores']);pair=np.asarray(scores['pairwise']);cov=np.maximum(0,asp[:,ix].max(1)) if ix else np.zeros(len(asp))
    represented=int(sum(v>=0.5 for v in cov));units=selection['units'];tokens=selection['evidence_tokens']
    pairs=[float(pair[i,j]) for k,i in enumerate(ix) for j in ix[k+1:]]
    return dict(selected_units=len(units),represented_parents=len({u['passage_id'] for u in units}),
                sentence_units=sum(u['unit_type']=='SENTENCE' for u in units),full_units=sum(u['unit_type']=='PASSAGE' for u in units),
                mean_pairwise_redundancy=float(np.mean(pairs)) if pairs else None,
                redundancy_definition='Pairwise cosine over exposed underlying sentence units, including sentences covered by FULL; undefined for <2',
                aspect_coverage=float(cov.mean()) if len(cov) else 0.,aspect_count=len(cov),represented_aspects=represented,
                tokens_per_represented_aspect=tokens/represented if represented else None,
                fidelity_upgrades=selection.get('fidelity_upgrades',0),empty_evidence=not bool(units),
                full_eligible_query=any(a['eligible'] for a in selection.get('full_attempts',[])),
                **{'full_'+status.lower():sum(a['status']==status for a in selection.get('full_attempts',[]))
                   for status in ['CHOSEN','REJECTED_COST','REJECTED_LOW_GAIN','REJECTED_THRESHOLD','LOST_COMPETITION','BLOCKED_ONE_SENTENCE']},
                full_considered=sum(a['eligible'] for a in selection.get('full_attempts',[])),
                stopped_with_unused_budget=selection.get('stop_reason')=='LOW_MARGINAL_VALUE' and tokens<selection['budget'])
