"""Paired query metrics, paper bootstrap, oracle concentration and stability."""
from collections import Counter
import numpy as np

def interval_coverage(interval,chunks):
    a,b=interval; pieces=sorted((max(a,c['char_start']),min(b,c['char_end'])) for c in chunks
        if c['char_start']<b and c['char_end']>a)
    covered=0;cursor=a
    for x,y in pieces:
        if y>cursor:covered+=y-max(cursor,x);cursor=y
    return covered/max(1,b-a)


def retrieval_metrics(ranked,evaluation,token_count):
    occurrences=evaluation.get('gold_occurrences'); groups=evaluation.get('gold_chunk_sets',[])
    n=len(occurrences) if occurrences is not None else len(groups)
    def coverage(selected):
        if occurrences is not None:
            return [max((interval_coverage(span,selected) for span in alternatives),default=0.) for alternatives in occurrences]
        ids={c['chunk_id'] for c in selected}
        return [float(bool(ids & set(g))) for g in groups]
    first=next((i+1 for i in range(len(ranked)) if any(v>=1-1e-9 for v in coverage(ranked[:i+1]))),None)
    rows=[]
    for k in (1,2,4,6,8,10):
        selected=ranked[:k]; cov=coverage(selected);hits=[v>=1-1e-9 for v in cov]
        words=[set(c['text'].lower().split()) for c in selected]
        pairs=[len(a&b)/max(1,len(a|b)) for i,a in enumerate(words) for b in words[i+1:]]
        rows.append(dict(k=k,actual_k=len(selected),gold_span_count=n,
            any_recall=float(any(hits)) if n else None,all_recall=float(all(hits)) if n else None,
            coverage=float(np.mean(cov)) if n else None,first_gold_rank=first,mrr=(1/first if first else 0.) if n else None,
            unmapped_gold_spans=sum(not g for g in (occurrences if occurrences is not None else groups)),
            retrieved_tokens=sum(token_count(c['text']) for c in selected),
            duplicate_rate=1-len({c['text'] for c in selected})/max(1,len(selected)),
            mean_lexical_jaccard=float(np.mean(pairs)) if pairs else None,
            candidate_diversity=1-float(np.mean(pairs)) if pairs else None,
            evidence_mapping='Exact canonical-source spans after whitespace/BIBREF normalization; repeated occurrences treated as alternatives. Full-span union coverage defines recall; unmapped gold remains missed. Diversity is lexical Jaccard proxy.'))
    return rows


def oracle(rows,epsilon=.01):
    qids=sorted({r['question_id'] for r in rows}); actions=sorted({r['action'] for r in rows})
    by={(r['question_id'],r['action']):r for r in rows}
    if len(by)!=len(rows) or len(rows)!=len(qids)*len(actions): raise ValueError('Oracle requires complete paired surface')
    means={a:np.mean([by[q,a]['f1'] for q in qids]) for a in actions}; best=max(means,key=means.get)
    exact=[]; eps=[]; gains=[]
    for q in qids:
        group=[by[q,a] for a in actions]; peak=max(r['f1'] for r in group)
        order=lambda r:(r.get('k',0),r.get('cumulative_prompt_tokens',0),r['action'])
        a=min((r for r in group if r['f1']==peak),key=order)
        b=min((r for r in group if r['f1']>=peak-epsilon),key=order)
        exact.append(a);eps.append(b);gains.append(peak-by[q,best]['f1'])
    total=sum(gains); desc=sorted(gains,reverse=True)
    paper_gains=Counter();type_gains=Counter()
    for q,gain in zip(qids,gains):
        row=by[q,best];paper_gains[row.get('paper_id','unknown')]+=gain
        type_gains[str(row.get('answer_type','unknown'))]+=gain
    return dict(best_fixed=best,best_fixed_f1=float(means[best]),exact_oracle_f1=float(np.mean([r['f1'] for r in exact])),
        epsilon_oracle_f1=float(np.mean([r['f1'] for r in eps])),oracle_headroom=float(np.mean(gains)),
        epsilon=epsilon,exact_action_distribution=dict(Counter(r['action'] for r in exact)),
        epsilon_action_distribution=dict(Counter(r['action'] for r in eps)),
        epsilon_mean_k=float(np.mean([r.get('k',0) for r in eps])),
        epsilon_median_k=float(np.median([r.get('k',0) for r in eps])),
        epsilon_mean_prompt_tokens=float(np.mean([r.get('cumulative_prompt_tokens',0) for r in eps])),
        positive_gain_at_most_001_fraction=sum(0<g<=.01 for g in gains)/max(1,sum(g>0 for g in gains)),
        concentration={str(p):sum(desc[:max(1,int(np.ceil(len(desc)*p/100)))])/total if total>0 else None for p in (1,5,10,20)},
        top10_percent_gain_contribution=sum(desc[:max(1,int(np.ceil(len(desc)*.1)))])/total if total>0 else None,
        paper_gain_contribution={p:g/total if total>0 else None for p,g in paper_gains.most_common()},
        answer_type_gain_contribution={p:g/total if total>0 else None for p,g in type_gains.most_common()},
        deployable=False)

def capture(policy,best,ceiling): return (policy-best)/(ceiling-best) if ceiling-best>1e-6 else None

def paired(a,b,resamples=10000,seed=42):
    from src.week9.evaluation import paired_paper_bootstrap
    return paired_paper_bootstrap(a,b,n=resamples,seed=seed)

def stability(rows):
    groups={}
    for r in rows: groups.setdefault((r['question_id'],r['action']),[]).append(r['delta'])
    values=[]
    for key,deltas in groups.items():
        if len(deltas)<2: raise ValueError('Stability requires repeated, paired seeds')
        labels=[1 if d>.01 else -1 if d<-.01 else 0 for d in deltas]
        agreement=max(Counter(labels).values())/len(labels)
        values.append(dict(question_id=key[0],action=key[1],agreement=agreement,delta_variance=float(np.var(deltas)),
            sign_stable=len(set(labels))==1))
    return values
