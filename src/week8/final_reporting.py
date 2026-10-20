"""Strict paired, paper-clustered reporting for measured W8-I/J answers."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.week8.qcla_experiment import atomic_json


def paired_interval(treatment,control,n_resamples=2000,seed=42):
    columns=['question_id','paper_id','token_f1']
    t=treatment[columns].copy();c=control[columns].copy()
    for frame in (t,c):
        if frame.question_id.duplicated().any() or frame[columns].isna().any().any():raise ValueError('Duplicate or missing paired identities/outcomes')
    t=t.set_index('question_id').sort_index();c=c.set_index('question_id').sort_index()
    if not t.index.equals(c.index) or not t.paper_id.equals(c.paper_id):raise ValueError('Paired comparison membership/paper mismatch')
    if not len(t) or n_resamples<1:raise ValueError('Empty comparison or invalid resamples')
    delta=t.token_f1.to_numpy(float)-c.token_f1.to_numpy(float)
    if not np.isfinite(delta).all():raise ValueError('Non-finite paired outcomes')
    codes,papers=pd.factorize(t.paper_id);n=len(papers)
    sums=np.bincount(codes,weights=delta);counts=np.bincount(codes)
    rng=np.random.default_rng(seed);draw=rng.integers(n,size=(n_resamples,n))
    samples=sums[draw].sum(axis=1)/counts[draw].sum(axis=1)
    return {'delta_f1':float(delta.mean()),'ci95_low':float(np.quantile(samples,.025)),
        'ci95_high':float(np.quantile(samples,.975)),'queries':len(t),'papers':n,'resamples':n_resamples,
        'interval_method':'paired paper-cluster percentile bootstrap; query-weighted mean; exploratory, no multiplicity correction'}


def report_systems(systems,folder,note):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    base=systems['semantic_k4'];metrics=[];intervals=[]
    for name,f in systems.items():
        interval=paired_interval(f,base)
        intervals.append({'treatment':name,'control':'semantic_k4',**interval})
        metrics.append({'system':name,'queries':len(f),'mean_token_f1':float(f.token_f1.mean()),
            'mean_final_prompt_tokens':float(f.final_prompt_tokens.mean()),
            'mean_cumulative_prompt_tokens':float(f.cumulative_prompt_tokens.mean()),
            'promotion_rate':float(f.selected_action.eq('PROMOTE').mean()) if 'selected_action' in f else None})
    # Fixed matched comparisons; never choose a validation/DEV winner.
    for name in systems:
        if name.endswith('/G1') and name[:-2]+'G0' in systems:
            control=name[:-2]+'G0';intervals.append({'treatment':name,'control':control,**paired_interval(systems[name],systems[control])})
        if name.endswith('/G0') and not name.startswith('original_SARA/') and 'original_SARA/G0' in systems:
            intervals.append({'treatment':name,'control':'original_SARA/G0',**paired_interval(systems[name],systems['original_SARA/G0'])})
    table=pd.DataFrame(metrics);table.to_csv(folder/'benchmark.csv',index=False)
    pd.DataFrame(intervals).to_csv(folder/'paired_paper_intervals.csv',index=False)
    atomic_json(folder/'report_notes.json',{'status':note,'TEST_used':False,'no_winner_selection':True,
        'cost_scope':'Actual serialized prompt tokens. Cumulative prompt tokens are a compute proxy, not FLOPs or deployment latency.',
        'unmeasured_systems':['DAFiR-2: no frozen second-state controller','QCLA + live gate: uncalibrated draft distribution'],
        'inference_scope':'Exploratory intervals; multiple comparisons are not confirmatory significance tests.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(15,9))
    handles=[]
    for ax,column,title in zip(axes,['mean_final_prompt_tokens','mean_cumulative_prompt_tokens'],['Final prompt','Cumulative prompts']):
        positions={}
        for i,row in enumerate(metrics,1):
            point=ax.scatter(row[column],row['mean_token_f1'],color=plt.get_cmap('tab20')((i-1)%20),label=f"{i}. {row['system']}")
            key=(round(row[column],5),round(row['mean_token_f1'],7))
            offset=positions.get(key,0);positions[key]=offset+1
            ax.annotate(str(i),(row[column],row['mean_token_f1']),xytext=(5,4+10*offset),textcoords='offset points',fontsize=8)
            if len(handles)<len(metrics):handles.append(point)
        ax.set_xlabel(title+' (tokens)');ax.set_ylabel('Mean token F1');ax.grid(alpha=.2);ax.margins(.12)
    fig.suptitle(note,fontsize=10)
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=8,frameon=False)
    fig.tight_layout(rect=(0,.28,1,.96));fig.savefig(folder/'quality_cost.png',dpi=160);plt.close(fig)
    return table
