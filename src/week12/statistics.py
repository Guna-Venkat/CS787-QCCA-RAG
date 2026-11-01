"""Dataset-specific metrics and paired clustered inference; no pooled fake score."""
import numpy as np
from .protocol import digest

def metric(rows,name):
    if not rows:return None
    if name in ('f1','accuracy'):return float(np.mean([r['f1' if name=='f1' else 'correct'] for r in rows]))
    labels=['SUPPORT','REFUTE','NOT_ENOUGH_INFO'];values=[]
    for label in labels:
        tp=sum(r['gold']==label and r['prediction']==label for r in rows);fp=sum(r['gold']!=label and r['prediction']==label for r in rows);fn=sum(r['gold']==label and r['prediction']!=label for r in rows)
        values.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return float(np.mean(values))
def components(rows,name):
    if name!='macro_f1':return np.array([[r['f1' if name=='f1' else 'correct'],1.] for r in rows])
    labels=['SUPPORT','REFUTE','NOT_ENOUGH_INFO'];return np.array([[float(r['gold']==l and r['prediction']==l) for l in labels]+[float(r['gold']!=l and r['prediction']==l) for l in labels]+[float(r['gold']==l and r['prediction']!=l) for l in labels] for r in rows])
def reduce(x,name):
    if name!='macro_f1':return x[...,0]/x[...,1]
    tp=x[...,:3];den=2*tp+x[...,3:6]+x[...,6:9];return np.divide(2*tp,den,out=np.zeros_like(tp),where=den!=0).mean(-1)
def paired(left,right,name,n=10000,seed=42):
    if len({r['dataset'] for r in left+right})!=1:raise ValueError('Never pool raw incompatible task metrics')
    a={r['query_id']:r for r in left};b={r['query_id']:r for r in right}
    if len(a)!=len(left) or len(b)!=len(right) or a.keys()!=b.keys() or not a:raise ValueError('Incomplete/duplicate paired surface')
    q=sorted(a);assert all(a[k]['cluster_id']==b[k]['cluster_id'] for k in q)
    clusters=sorted({a[k]['cluster_id'] for k in q});ix={c:i for i,c in enumerate(clusters)}
    av=components([a[k] for k in q],name);bv=components([b[k] for k in q],name);aa=np.zeros((len(clusters),av.shape[1]));bb=aa.copy()
    for j,k in enumerate(q):aa[ix[a[k]['cluster_id']]]+=av[j];bb[ix[a[k]['cluster_id']]]+=bv[j]
    point=float(reduce(aa.sum(0),name)-reduce(bb.sum(0),name));rng=np.random.default_rng(seed);boot=[];null=[]
    for start in range(0,n,256):
        count=min(256,n-start);draw=rng.integers(0,len(clusters),(count,len(clusters)))
        boot.extend(reduce(aa[draw].sum(1),name)-reduce(bb[draw].sum(1),name))
        swap=rng.integers(0,2,(count,len(clusters),1));x=(aa[None]*(1-swap)+bb[None]*swap).sum(1);y=(bb[None]*(1-swap)+aa[None]*swap).sum(1)
        null.extend(reduce(x,name)-reduce(y,name))
    lo,hi=np.quantile(boot,[.025,.975]);p=(1+sum(abs(float(v))>=abs(point)-1e-12 for v in null))/(n+1)
    return dict(point=point,ci_low=float(lo),ci_high=float(hi),raw_p=float(p),resamples=n,seed=seed,n_queries=len(q),n_clusters=len(clusters),metric=name,test='two-sided paired cluster label-swap randomization; exchangeability under null')
def holm(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);out=[0.]*len(values);prev=0.
    for rank,i in enumerate(order):prev=max(prev,min(1.,values[i]*(len(values)-rank)));out[i]=prev
    return out

def registry(cfg,llm=False,lora=False):
    specs=[]
    for dataset in cfg['datasets']:
        for budget in cfg['budgets']:
            for other in ['RAW_SKIP','FIXED_SENTENCE','H3','QBGF_NO_QUERY']+(['C2'] if budget==256 else []):specs.append(dict(id=f'{dataset}:{budget}:QBGF_FULL-{other}',dataset=dataset,budget=budget,left='QBGF_FULL',right=other,reader='BASE_FP16',metric=cfg['statistics']['primary_metric'][dataset]))
            if llm:specs.append(dict(id=f'{dataset}:{budget}:QBGF_FULL-LLMLINGUA2',dataset=dataset,budget=budget,left='QBGF_FULL',right='LLMLINGUA2',reader='BASE_FP16',metric=cfg['statistics']['primary_metric'][dataset]))
            if lora:specs.append(dict(id=f'{dataset}:{budget}:LORA-BASE',dataset=dataset,budget=budget,left='QBGF_FULL',right='QBGF_FULL',reader='LORA_NF4',right_reader='BASE_NF4',metric=cfg['statistics']['primary_metric'][dataset]))
    return specs

def pareto(rows):
    out=[]
    for r in rows:
        same=[v for v in rows if v['dataset']==r['dataset'] and v['reader']==r['reader']]
        dominated=any(v['prompt_tokens']<=r['prompt_tokens'] and v['quality']>=r['quality'] and (v['prompt_tokens']<r['prompt_tokens'] or v['quality']>r['quality']) for v in same)
        out.append(dict(r,pareto=not dominated))
    return out

def behavioral(score,baseline,classification=False,epsilon=.01):
    d=score-baseline;e=0 if classification else epsilon
    return dict(delta_task_score=d,HELP=d>e,HURT=d<-e,NEUTRAL=abs(d)<=e)
def compression(tokens,reference):return None if reference<=0 else 1-tokens/reference


def fusion_registry(cfg):
    """Prospective fusion diagnostics, preserving the frozen original 39 contrasts."""
    return [dict(id=f'{d}:{b}:QBGF_FULL-{other}',dataset=d,budget=b,left='QBGF_FULL',right=other,reader='BASE_FP16',metric=cfg['statistics']['primary_metric'][d]) for d in cfg['datasets'] for b in cfg['budgets'] for other in ['QBGF_SENTENCE_ONLY','QBGF_PASSAGE_ONLY']]
