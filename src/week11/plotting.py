"""Every figure consumes a table, saves that table, and is task/dataset-specific."""
from pathlib import Path
import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(frame,spec,directory,synthetic=False):
    name,_,kind,x,y,hue=spec
    if frame.empty or x not in frame:return []
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);outputs=[]
    combined=kind in ('rank_heatmap','count_heatmap','trajectory')
    groups=[('cross_dataset',frame)] if combined else list(frame.groupby('dataset')) if 'dataset' in frame else [('study',frame)]
    if name in ('recall_curves','fig01_retrieval') and 'evidence_granularity' in frame:
        groups=[(str(d)+' / '+str(g),part) for (d,g),part in frame.groupby(['dataset','evidence_granularity'])]
    for dataset,data in groups:
        if y and y not in data:continue
        data=data.copy();label=y or 'count'
        if x=='budget':
            # CSV round-trips make mixed numeric/FULL budgets strings.
            # Preserve numeric order rather than drawing 1024 before 256.
            levels=sorted(data[x].dropna().astype(str).unique(),key=lambda v:float('inf') if v=='FULL' else float(v))
            data[x]=pd.Categorical(data[x].astype(str),categories=levels,ordered=True)
        if kind in ('bar','heatmap') and 'evidence_granularity' in data and 'evidence_granularity' not in (x,hue):
            data=data[data.evidence_granularity!='NONE']
            if data.empty:continue
        if 'compressor' in data and 'reader' in data:
            data.loc[data.compressor=='C3','reader']=data.loc[data.compressor=='C3','reader']+'/SARA-specialist'
        fig,ax=plt.subplots(figsize=(9,5.5))
        if kind=='trajectory':
            # Deliberately separate panels: incompatible split/metric scales are not pooled.
            plt.close(fig);weeks=['W3','W6','W8','W9','W10','W11'];fig,axes=plt.subplots(2,3,figsize=(13,7.5))
            for axis,week in zip(axes.flat,weeks):
                subset=data[data.week==week];axis.set_title(week)
                if subset.empty:axis.text(.5,.5,'Reviewed record pending',ha='center',va='center',transform=axis.transAxes)
                else:
                    axis.bar(range(len(subset)),subset.value)
                    labels=[textwrap.fill(str(p),22) for p in subset.protocol]
                    axis.set_xticks(range(len(subset)),labels,fontsize=9)
                    axis.set_ylabel(str(subset.metric.iloc[0]).replace('_',' '),fontsize=9)
            fig.suptitle('Separate split/protocol panels — no pooled cross-week gain'+(' [SYNTHETIC SMOKE]' if synthetic else ''),fontsize=12)
        elif kind in ('heatmap','rank_heatmap','difference_heatmap','count_heatmap'):
            if hue not in data:plt.close(fig);continue
            if kind=='rank_heatmap':
                # Rank within each dataset across the compared x/hue cells.
                data=data.groupby(['dataset',x] if x!='dataset' else ['dataset',hue],dropna=False)[y].mean().reset_index()
                data[y]=data.groupby('dataset')[y].rank(ascending=False,method='average');label='Within-dataset rank (lower is better)'
            if kind=='count_heatmap':table=pd.crosstab(data[x].astype(str),data[hue].astype(str));label='Count'
            else:table=data.pivot_table(index=x,columns=hue,values=y,aggfunc='mean',dropna=True)
            if table.empty:plt.close(fig);continue
            if kind=='difference_heatmap':
                reference='Mistral' if 'Mistral' in table.index else 'BM25' if 'BM25' in table.index else None
                if reference is None:plt.close(fig);continue
                table=table.subtract(table.loc[reference],axis=1);label='Delta vs '+reference
            plot=ax.imshow(table.to_numpy(dtype=float),aspect='auto',cmap='viridis');fig.colorbar(plot,ax=ax,label=label)
            ax.set_xticks(range(len(table.columns)),table.columns.astype(str),rotation=35,ha='right')
            ax.set_yticks(range(len(table.index)),table.index.astype(str));ax.set_xlabel(hue);ax.set_ylabel(x)
        elif kind=='hist':
            for key,g in data.groupby(hue) if hue and hue in data else [('all',data)]:
                values=pd.to_numeric(g[x],errors='coerce').dropna()
                if len(values):ax.hist(values,bins=min(20,max(3,len(values))),alpha=.55,label=str(key))
            ax.set_xlabel(x);ax.set_ylabel('Count');ax.legend()
        elif kind=='help_hurt':
            table=data.groupby(x)[['HELP','HURT']].mean();table.plot.bar(ax=ax,rot=30)
            ax.set_xlabel(x);ax.set_ylabel('Fraction vs own no-evidence baseline')
        elif kind=='scatter':
            for key,g in data.groupby(hue) if hue and hue in data else [('all',data)]:
                ax.scatter(pd.to_numeric(g[x],errors='coerce'),pd.to_numeric(g[y],errors='coerce'),label=str(key),alpha=.75)
            ax.set_xlabel(x);ax.set_ylabel(label);ax.legend(fontsize=8)
        elif kind=='bar':
            if hue and hue in data:table=data.pivot_table(index=x,columns=hue,values=y,aggfunc='mean')
            else:table=data.groupby(x)[[y]].mean()
            if table.empty:plt.close(fig);continue
            table.plot.bar(ax=ax,rot=30);ax.set_xlabel(x);ax.set_ylabel(label)
        else:
            for key,g in data.groupby(hue) if hue and hue in data else [('all',data)]:
                means=g.groupby(x,dropna=False)[y].mean();positions=np.arange(len(means))
                if kind=='line':ax.plot(means.index.astype(str),means.values,marker='o',label=str(key))
                else:ax.plot(means.index.astype(str),means.values,'o',label=str(key))
            ax.set_xlabel(x);ax.set_ylabel(label);ax.tick_params(axis='x',labelrotation=30);ax.legend(fontsize=8)
        if kind!='trajectory':
            title=f'{name.replace("_"," ")} — {dataset}'+(' [SYNTHETIC SMOKE]' if synthetic else '')
            if kind in ('bar','line','heatmap'):
                averaged=[c for c in ['reader','retriever','selector','budget','compressor','evidence_granularity']
                          if c in data and c not in [x,hue] and data[c].astype(str).nunique()>1]
                if averaged:title+='\nDescriptive marginal means over '+', '.join(averaged)
            ax.set_title(title,fontsize=11)
        fig.tight_layout(rect=(0,0,1,.94) if kind=='trajectory' else (0,0,1,1))
        stem=re.sub(r'[^A-Za-z0-9_.-]+','_',f'{dataset}_{label}_{name}')
        if synthetic:stem='SMOKE_'+stem
        data.to_csv(directory/(stem+'.csv'),index=False)
        for extension in ['png','pdf']:fig.savefig(directory/(stem+'.'+extension),dpi=160,bbox_inches='tight')
        plt.close(fig);outputs.append(dict(name=name,dataset=dataset,png=str(directory/(stem+'.png')),pdf=str(directory/(stem+'.pdf')),data=str(directory/(stem+'.csv'))))
    return outputs
