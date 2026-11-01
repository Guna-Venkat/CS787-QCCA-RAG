import numpy as np
import pandas as pd


def paired(left,right,n=10000,seed=42):
    a,b=pd.DataFrame(left),pd.DataFrame(right)
    keys=['dataset','query_id']
    if a.duplicated(keys).any() or b.duplicated(keys).any():raise ValueError('Duplicate outcomes')
    x=a.merge(b,on=keys,suffixes=('_a','_b'),validate='one_to_one')
    if len(x)!=len(a) or len(x)!=len(b) or not len(x):raise ValueError('Unpaired/incomplete actions')
    if x.dataset.nunique()!=1:raise ValueError('Bootstrap cannot pool incompatible dataset scores')
    if (x.cluster_id_a!=x.cluster_id_b).any():raise ValueError('Cluster identity mismatch')
    x['delta']=x.task_score_a-x.task_score_b
    groups=x.groupby('cluster_id_a').delta.agg(['sum','count']);rng=np.random.default_rng(seed);values=[]
    for start in range(0,n,500):
        ix=rng.integers(0,len(groups),size=(min(500,n-start),len(groups)))
        values.extend(groups['sum'].to_numpy()[ix].sum(1)/groups['count'].to_numpy()[ix].sum(1))
    lo,hi=np.quantile(values,[.025,.975])
    return dict(point=float(x.delta.mean()),ci_low=float(lo),ci_high=float(hi),resamples=n,seed=seed,
                cluster='paper_id' if x.dataset.iloc[0]=='QASPER' else 'query_or_claim_id')
