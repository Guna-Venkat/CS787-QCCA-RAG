import itertools
import pandas as pd
from .bootstrap import paired

AXES=['reader','retriever','evidence_granularity','budget','compressor']
INTERACTIONS=[('reader','evidence_granularity'),('retriever','evidence_granularity'),
              ('evidence_granularity','budget'),('reader','compressor'),('retriever','reader')]


def describe(frame):
    if frame.empty:return pd.DataFrame(),pd.DataFrame()
    frame=frame.copy()
    frame.loc[frame.compressor=='C3','reader']=frame.loc[frame.compressor=='C3','reader']+'/SARA-specialist'
    means=[];interactions=[]
    # Never average raw task scores across datasets. Dataset effects are average ranks.
    for dataset,g in frame.groupby('dataset'):
        for axis in AXES:
            for value,h in g.groupby(axis,dropna=False):
                means.append(dict(dataset=dataset,axis=axis,level=str(value),task_score=h.task_score.mean(),
                    n_rows=len(h),interpretation='descriptive marginal mean; unequal block exposure, not a causal main effect'))
        for a,b in INTERACTIONS:
            for (x,y),h in g.groupby([a,b],dropna=False):
                interactions.append(dict(dataset=dataset,axis_a=a,level_a=str(x),axis_b=b,level_b=str(y),
                    task_score=h.task_score.mean(),n_rows=len(h),interpretation='descriptive cell mean'))
        normalized=[]
        for reader,h in g.groupby('reader'):
            baseline=h[h.evidence_granularity=='NONE'].task_score.mean()
            if pd.isna(baseline) or 1-baseline<=1e-6:continue
            actions=h[h.evidence_granularity!='NONE'].groupby('action_id').task_score.mean()
            normalized.extend(((actions-baseline)/(1-baseline)).tolist())
        means.append(dict(dataset=dataset,axis='dataset',level=dataset,task_score=None,n_rows=len(g),
            normalized_gain=sum(normalized)/len(normalized) if normalized else None,
            interpretation='Dataset effect uses fraction of remaining baseline headroom recovered; raw task scores never pooled'))
    return pd.DataFrame(means),pd.DataFrame(interactions)


def difference_in_differences(cells,n=10000,seed=42):
    if len(cells)!=4:raise ValueError('Four explicitly matched cells required')
    deltas=[]
    for left,right in [(cells[0],cells[1]),(cells[2],cells[3])]:
        a,b=pd.DataFrame(left),pd.DataFrame(right)
        x=a.merge(b,on=['dataset','query_id','cluster_id'],validate='one_to_one',suffixes=('_a','_b'))
        if len(x)!=len(a) or len(x)!=len(b):raise ValueError('Interaction is not paired')
        x['task_score']=x.task_score_a-x.task_score_b
        deltas.append(x[['dataset','query_id','cluster_id','task_score']].to_dict('records'))
    return paired(*deltas,n=n,seed=seed)
