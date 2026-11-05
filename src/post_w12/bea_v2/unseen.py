"""Metadata-only unseen-VAL planning. No automatic expanded inference."""
from .protocol import *

def plan(c=None):
    c=c or load();parent,v1,s,cohort,f=freeze(c)
    decision=read(root(c)/'pilot/PILOT_DECISION.json')
    if decision['decision']!='PROMISING':raise ValueError('Unseen-VAL planning requires a promising complete V2 pilot')
    folds=read(root(c)/'CROSSFIT_FOLDS.json');out={}
    v1_expanded=old(c)/'expanded/EXPANDED_COHORT.json'
    expanded=read(v1_expanded)['selected'] if v1_expanded.exists() else {}
    for d in DATASETS:
        excluded_ids=set(folds['assignments'][d]);excluded_groups={r['group'] for r in cohort['selected'][d]}
        for r in expanded.get(d,[]):excluded_ids.add(r['query_id']);excluded_groups.add(r['group'])
        eligible=[r for r in s['datasets'][d]['VAL'] if r['query_id'] not in excluded_ids and r['group'] not in excluded_groups]
        eligible.sort(key=lambda r:digest([42,d,r['query_id']]))
        out[d]=[dict(query_id=r['query_id'],group=r['group'],paper_id=r['paper_id']) for r in eligible[:100]]
    obj=dict(label='POST_W12_EXPLORATORY_VAL',selected=out,selection='Hash-order metadata only, exclude all trained/development groups and any V1 expanded groups',
             status='PLANNED_ONLY',requires='Separately freeze all-development predictors and fresh exact cached/prepared SFR problems before unseen inference; crossfit predictors are never substituted')
    write(root(c)/'expanded_unseen_val/COHORT_PLAN.json',obj);return obj
