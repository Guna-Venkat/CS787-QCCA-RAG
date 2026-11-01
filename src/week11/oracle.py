from collections import Counter
import numpy as np
import pandas as pd


def capture(policy,best,ceiling,tolerance=1e-6):
    return (policy-best)/(ceiling-best) if ceiling-best>tolerance else None


def surface(rows,epsilon=.01):
    frame=pd.DataFrame(rows)
    if frame.dataset.nunique()!=1 or frame.task_type.nunique()!=1:raise ValueError('Oracle is dataset/task-specific')
    if frame.duplicated(['query_id','action_id']).any():raise ValueError('Duplicate oracle actions')
    counts=frame.groupby('action_id').query_id.agg(set)
    if any(q!=counts.iloc[0] for q in counts):raise ValueError('Incomplete action surface; no silent complete-case filtering')
    means=frame.groupby('action_id').task_score.mean();best=means.idxmax();base=frame[frame.action_id==best].set_index('query_id')
    result=[]
    epsilon=0. if frame.task_type.iloc[0]=='verification' else epsilon
    for q,g in frame.groupby('query_id'):
        peak=g.task_score.max();order=['actual_evidence_tokens','prompt_tokens','action_id']
        exact=g[g.task_score==peak].sort_values(order).iloc[0]
        approx=g[g.task_score>=peak-epsilon].sort_values(order).iloc[0]
        result.append(dict(dataset=frame.dataset.iloc[0],query_id=q,best_fixed_id=best,best_fixed_score=float(base.loc[q].task_score),
            oracle_score=float(peak),epsilon_score=float(approx.task_score),exact_action=exact.action_id,epsilon_action=approx.action_id,
            granularity=approx.evidence_granularity,reader=approx.get('reader_system',approx.reader),budget=approx.budget,compressor=approx.compressor,
            gain=float(peak-base.loc[q].task_score),epsilon_tokens=int(approx.actual_evidence_tokens)))
    gains=sorted([r['gain'] for r in result],reverse=True);total=sum(gains);concentration=[]
    for p in (1,5,10,20):
        concentration.append(dict(top_percent=p,gain_fraction=sum(gains[:max(1,int(np.ceil(len(gains)*p/100)))])/total if total else None))
    curve=[dict(fraction=(i+1)/len(gains),cumulative_gain_fraction=sum(gains[:i+1])/total if total else 0.) for i in range(len(gains))]
    return dict(best_fixed=best,best_fixed_score=float(means.max()),exact_oracle=float(np.mean([r['oracle_score'] for r in result])),
        epsilon_oracle=float(np.mean([r['epsilon_score'] for r in result])),oracle_headroom=float(np.mean(gains)),
        epsilon=epsilon,per_query=result,concentration=concentration,curve=curve,
        exact_distribution=dict(Counter(r['exact_action'] for r in result)),
        epsilon_distribution=dict(Counter(r['epsilon_action'] for r in result)),deployable=False)
