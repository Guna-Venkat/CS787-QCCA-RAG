"""Iterative context-dependent source-budget transitions, not exact static MCKP."""
import time
import numpy as np
from src.week12.algorithm import serialize,order,validate_pool
from src.post_w12.bea_qbgf.allocator import marginal
from .protocol import digest

def utility(i,selected,rel,asp,pair,ph,pm,params):
    m=marginal(i,selected,rel,asp,pair,params);raw=float(ph[i]);weight=raw
    return dict(ig=m['gain'],coverage_gain=m['coverage_gain'],redundancy=m['redundancy'],relevance=m['relevance'],
                p_help=float(ph[i]),p_harm=float(pm[i]),help_minus_harm_diagnostic=float(ph[i]-pm[i]),downstream_weight=raw,positive_weight=weight,value=weight*max(0.,m['gain']))

def sequence(indices,selected,count,rel,asp,pair,ph,pm,params,ss,full=False):
    """Re-rank after EVERY hypothetical addition, including zero-value FULL context."""
    chosen=set(selected);remaining=set(indices)-chosen;steps=[]
    while remaining and (full or len(steps)<count):
        values=[(i,utility(i,chosen,rel,asp,pair,ph,pm,params)) for i in remaining]
        i,v=min(values,key=lambda x:(-x[1]['value'],order(ss[x[0]]),ss[x[0]]['unit_id']))
        if not full and v['value']<=0:break
        steps.append(dict(index=i,unit_id=ss[i]['unit_id'],**v));chosen.add(i);remaining.remove(i)
    return steps

def allocate(pool,sentences,scores,probabilities,tok,budget,params,threshold,states=(1,2,4,8)):
    begin=time.perf_counter();validate_pool(pool);ss=list(sentences);n=len(ss)
    rel=np.asarray(scores['relevance']);asp=np.asarray(scores['aspect_scores']);pair=np.asarray(scores['pairwise']).reshape(n,n)
    ph=np.asarray(probabilities['p_help']);pm=np.asarray(probabilities['p_harm'])
    if rel.shape!=(n,) or ph.shape!=(n,) or pm.shape!=(n,) or asp.shape[1]!=n or not all(np.isfinite(a).all() for a in [rel,asp,pair,ph,pm]):raise ValueError('Misaligned or nonfinite candidate data')
    if np.any((ph<0)|(ph>1)|(pm<0)|(pm>1)):raise ValueError('Probability bounds violated')
    pmap={p['unit_id']:p for p in pool};ix={pid:[i for i,s in enumerate(ss) if s['passage_id']==pid] for pid in pmap}
    if len({s['unit_id'] for s in ss})!=n or any(s['passage_id'] not in pmap for s in ss):raise ValueError('Duplicate sentence/unknown parent')
    selected=set();full=set();contributions={p:0 for p in pmap};trace=[];attempts=[];marginals=[];seen_states={p:set() for p in pmap}
    def units(chosen,whole):return [pmap[p] for p in sorted(whole)]+[ss[i] for i in sorted(chosen) if ss[i]['passage_id'] not in whole]
    current=serialize([],tok)
    while True:
        candidates=[];positive_over_budget=0;step_attempts=[]
        for pid,indices in ix.items():
            if pid in full:continue
            count=len(selected&set(indices));remaining=len(indices)-count
            if not remaining:continue
            # The next discrete cumulative state only: 0->1->2->4->8->all.
            targets=sorted({min(t,len(indices)) for t in states if min(t,len(indices))>count})
            if count>=2:targets.append(len(indices))
            next_target=min(targets) if targets else None
            proposals=[]
            if next_target is not None:
                steps=sequence(indices,selected,next_target-count,rel,asp,pair,ph,pm,params,ss)
                if len(steps)==next_target-count:proposals.append(('SENTENCES',steps,selected|{s['index'] for s in steps},set(full)))
            if contributions[pid]>=params['min_parent_sentences']:
                steps=sequence(indices,selected,remaining,rel,asp,pair,ph,pm,params,ss,full=True)
                proposals.append(('FULL',steps,selected|set(indices),full|{pid}))
            elif contributions[pid]==1:
                step_attempts.append(dict(parent=pid,kind='FULL',status='BLOCKED_ONE_POSITIVE',step=len(trace)))
            unique={}
            for kind,steps,chosen,whole in proposals:
                trial=serialize(units(chosen,whole),tok);signature=digest(trial['evidence_ids'])
                v=sum(x['value'] for x in steps);dc=trial['evidence_tokens']-current['evidence_tokens'];ratio=v/dc if dc>0 else float('inf')
                event=dict(parent=pid,kind=kind,step=len(trace),value=v,cost=dc,evidence_tokens=trial['evidence_tokens'])
                if v<=0:event['status']='REJECTED_VALUE'
                elif trial['evidence_tokens']>budget:event['status']='REJECTED_COST';positive_over_budget+=1
                elif ratio<threshold:event['status']='REJECTED_THRESHOLD'
                else:event['status']='LOST_COMPETITION'
                step_attempts.append(event)
                if signature in unique:continue # exact-equivalent short-source states are deduplicated
                unique[signature]=kind
                if v>0 and trial['evidence_tokens']<=budget:
                    candidates.append(dict(kind=kind,parent=pid,steps=steps,chosen=chosen,whole=whole,trial=trial,value=v,cost=dc,ratio=ratio,event=event))
            seen_states[pid].add(count)
        finite=[c['ratio'] for c in candidates if np.isfinite(c['ratio'])]
        marginals.append(dict(step=len(trace),values=finite,maximum=max(finite) if finite else None))
        valid=[c for c in candidates if c['ratio']>=threshold]
        if not valid:
            stop='LOW_MARGINAL_VALUE' if candidates else 'BUDGET_EXHAUSTED' if positive_over_budget else 'NO_POSITIVE_VALUE'
            attempts.extend(step_attempts);break
        best=min(valid,key=lambda a:(-a['ratio'],-a['value'],a['cost'],order(pmap[a['parent']]),a['kind']))
        best['event']['status']='CHOSEN';attempts.extend(step_attempts)
        selected=best['chosen'];full=best['whole'];contributions[best['parent']]+=sum(s['value']>0 for s in best['steps'])
        current=best['trial']
        trace.append(dict(step=len(trace)+1,parent=best['parent'],kind=best['kind'],value=best['value'],incremental_cost=best['cost'],
                          marginal_score=best['ratio'] if np.isfinite(best['ratio']) else None,
                          nonpositive_incremental_cost=best['cost']<=0,sentences=best['steps'],evidence_tokens=current['evidence_tokens']))
    if current['evidence_tokens']>budget:raise AssertionError('Budget violation')
    return dict(current,method='BEA_V2',budget=budget,unused_budget=budget-current['evidence_tokens'],
                selected_unit_ids=[u['unit_id'] for u in current['units']],evidence_hash=digest([current['evidence_ids'],current['text']]),
                trace=trace,transition_attempts=attempts,
                full_attempts=[dict(a,eligible=a['status']!='BLOCKED_ONE_POSITIVE',status='BLOCKED_ONE_SENTENCE' if a['status']=='BLOCKED_ONE_POSITIVE' else 'REJECTED_LOW_GAIN' if a['status']=='REJECTED_VALUE' else a['status']) for a in attempts if a['kind']=='FULL'],
                marginal_distribution=marginals,stop_reason=stop,
                fidelity_upgrades=sum(t['kind']=='FULL' for t in trace),full_sources=len(full),
                source_positive_contributions=contributions,source_states={p:sorted(v) for p,v in seen_states.items()},
                selection_seconds=time.perf_counter()-begin,source_support=1.)
