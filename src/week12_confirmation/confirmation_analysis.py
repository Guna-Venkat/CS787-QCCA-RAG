"""Predefined paired confirmation families; real scoring needs the explicit permit."""
import json
from statistics import mean
from pathlib import Path
import pandas as pd
from src.week12.protocol import Store,digest,sha,atomic,root
from src.week12.statistics import metric,paired,holm
from .freeze_guard import require,verify,protocol,folder,registry,METHODS,BUDGETS,COUNTS

def completed_rows(out,ident):
    p=out/'complete.json'
    if not p.exists():raise ValueError('Report/scoring unavailable before exact-grid completion')
    done=json.loads(p.read_text())
    if not done.get('complete') or done['identity']!=digest(ident):raise ValueError('Incompatible completion receipt')
    sums=json.loads((out/'result_checksums.json').read_text());actual={p.name:sha(p) for p in (out/'rows').glob('*.json') if p.name!='identity.json'}
    if actual!=sums:raise ValueError('Result checksum mismatch')
    rows=Store(out/'rows',ident).rows()
    if len(rows)!=done['expected'] or len(rows)!=done['logical_rows']:raise ValueError('Incomplete result grid')
    keys={(r['dataset'],r['query_id'],r['method'],r['budget']) for r in rows}
    ids={(r['dataset'],r['query_id']) for r in rows};expected={(d,q,m,b) for d,q in ids for m in METHODS for b in BUDGETS}
    if keys!=expected or len(keys)!=len(rows):raise ValueError('Wrong method/budget grid')
    return rows

def tables(rows,n=10000):
    summary=[];classes=[];contrasts=[]
    for d in ['HotpotQA','SciFact']:
        for b in BUDGETS:
            for m in METHODS:
                r=[v for v in rows if (v['dataset'],v['budget'],v['method'])==(d,b,m)]
                if not r:raise ValueError('Missing system')
                invalid=sum(v.get('format_compliant') is False for v in r)
                summary.append(dict(dataset=d,budget=b,method=m,n_queries=len(r),n_groups=len({v['cluster_id'] for v in r}),primary_metric='answer_F1' if d=='HotpotQA' else 'macro_F1',quality=metric(r,'f1' if d=='HotpotQA' else 'macro_f1'),secondary_metric='EM' if d=='HotpotQA' else 'accuracy',secondary=mean(v['exact_match'] for v in r) if d=='HotpotQA' else metric(r,'accuracy'),evidence_tokens=mean(v['evidence_tokens'] for v in r),prompt_tokens=mean(v['prompt_tokens'] for v in r),generated_tokens=mean(v['generated_tokens'] for v in r),budget_utilization=mean(v['budget_utilization'] for v in r),empty_evidence_count=sum(not v['evidence_token_ids'] for v in r),output_compliance=(len(r)-invalid)/len(r) if d=='SciFact' else None,invalid_format_count=invalid if d=='SciFact' else None,invalid_format_rate=invalid/len(r) if d=='SciFact' else None,answer_generation_seconds=mean(v['generation_seconds'] for v in r),conceptual_lm_calls=mean(v['conceptual_lm_calls'] for v in r)))
                if d=='SciFact':
                    for label in ['SUPPORT','REFUTE','NOT_ENOUGH_INFO']:
                        tp=sum(v['gold']==label and v['prediction']==label for v in r);fp=sum(v['gold']!=label and v['prediction']==label for v in r);fn=sum(v['gold']==label and v['prediction']!=label for v in r)
                        classes.append(dict(dataset=d,budget=b,method=m,label=label,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,support=sum(v['gold']==label for v in r)))
    for spec in registry():
        select=lambda m:[r for r in rows if (r['dataset'],r['budget'],r['method'])==(spec['dataset'],spec['budget'],m)]
        contrasts.append(dict(spec,**paired(select(spec['left']),select(spec['right']),spec['metric'],n,42)))
    for family in ['primary','secondary']:
        subset=[r for r in contrasts if r['family']==family]
        for r,p in zip(subset,holm([r['raw_p'] for r in subset])):r['holm_p']=p;r['family_size']=len(subset)
    return dict(summary=pd.DataFrame(summary),scifact_classes=pd.DataFrame(classes),contrasts=pd.DataFrame(contrasts))

def validation_comparison(cfg,summary,contrasts):
    from src.week12.runner import directory
    # Existing C rows only; no new evaluation, no cross-sample paired test.
    path=root(cfg)/'analysis/summary.csv'
    if not path.exists():return pd.DataFrame([dict(status='VAL summary unavailable; no fabricated comparison')])
    val=pd.read_csv(path);result=[]
    for r in summary.to_dict('records'):
        v=val[(val.dataset==r['dataset']) & (val.method==r['method']) & (val.budget.astype(str)==str(r['budget']))]
        if 'reader' in v:v=v[v.reader=='BASE_FP16']
        if len(v)!=1:continue
        score=float(v.iloc[0]['quality']);result.append(dict(dataset=r['dataset'],method=r['method'],budget=r['budget'],VAL=score,confirmation=r['quality'],change=r['quality']-score,inference='Unpaired descriptive sample shift; not a paired test'))
    df=pd.DataFrame(result)
    if not df.empty:
        for c in contrasts.to_dict('records'):
            pick=lambda m:df[(df.dataset==c['dataset'])&(df.budget==c['budget'])&(df.method==m)]
            a,b=pick(c['left']),pick(c['right'])
            if len(a)==len(b)==1:
                mask=(df.dataset==c['dataset'])&(df.budget==c['budget'])&(df.method==c['left'])
                df.loc[mask,'VAL_'+c['right']+'_contrast']=float(a.VAL.iloc[0]-b.VAL.iloc[0]);df.loc[mask,'CONF_'+c['right']+'_contrast']=c['point']
    return df

def _write(out,rows,cfg,n,synthetic=False):
    frames=tables(rows,n)
    frames['validation_comparison']=pd.DataFrame([dict(status='SYNTHETIC; no real VAL/CONF comparison')]) if synthetic else validation_comparison(cfg,frames['summary'],frames['contrasts'])
    for name,df in frames.items():df.to_csv(out/(name+'.csv'),index=False)
    from .confirmation_report import write
    write(out,frames,synthetic)
    return frames

def analyze(cfg,permit):
    require(permit);f,s=verify(cfg);ident=protocol(cfg);out=folder();rows=completed_rows(out,ident)
    expected={(d,q,m,b) for d,ids in f['confirmation_ids'].items() for q in ids for m in METHODS for b in BUDGETS}
    if {(r['dataset'],r['query_id'],r['method'],r['budget']) for r in rows}!=expected:raise ValueError('Changed confirmation cohort')
    groups={(d,r['query_id']):r['group'] for d,p in s['datasets'].items() for r in p['CONFIRM']}
    if any(r['cluster_id']!=groups[r['dataset'],r['query_id']] for r in rows):raise ValueError('Changed confirmation group')
    return _write(out,rows,cfg,10000)

def analyze_synthetic(out,ident):
    if not ident.get('synthetic') or out.resolve()==folder().resolve():raise PermissionError('Synthetic analysis cannot read real confirmation')
    from src.week12.protocol import load
    rows=completed_rows(out,ident)
    if not all(r.get('synthetic') for r in rows):raise PermissionError('Not synthetic fixtures')
    return _write(out,rows,load(),100,True)
