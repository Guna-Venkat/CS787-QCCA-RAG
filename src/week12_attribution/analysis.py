"""Clustered descriptive input-ablation analysis; no inference-time controller."""
import json,collections
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr,pearsonr
from src.week12.protocol import Store,atomic,immutable,digest,sha
from .protocol import directory,study_identity
from .experiment import causal_order,relevance_order,retention

def correlation_ci(rows,n=10000,seed=42):
    x=np.array([r['relevance'] for r in rows]);y=np.array([r['importance'] for r in rows]);groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[r['cluster_id']].append(i)
    if len(rows)<2 or np.ptp(x)==0 or np.ptp(y)==0:return dict(rho=None,ci_low=None,ci_high=None,pearson_secondary=None,valid_bootstrap=0,n_units=len(rows),n_clusters=len(groups),reason='constant/insufficient values; never imputed zero')
    rho=float(spearmanr(x,y).statistic);pearson=float(pearsonr(x,y).statistic);keys=sorted(groups);rng=np.random.default_rng(seed);boot=[]
    for _ in range(n):
        indices=np.concatenate([groups[keys[i]] for i in rng.integers(0,len(keys),len(keys))]);a,b=x[indices],y[indices]
        if np.ptp(a)>0 and np.ptp(b)>0:boot.append(float(spearmanr(a,b).statistic))
    lo,hi=np.quantile(boot,[.025,.975]) if boot else (None,None)
    return dict(rho=rho,ci_low=float(lo) if lo is not None else None,ci_high=float(hi) if hi is not None else None,pearson_secondary=pearson,valid_bootstrap=len(boot),n_units=len(rows),n_clusters=len(groups),reason=None)

def mean_ci(rows,value,n=10000,seed=42):
    groups=collections.defaultdict(list)
    for r in rows:
        if r.get(value) is not None:groups[r['cluster_id']].append(r[value])
    if not groups:return dict(mean=None,ci_low=None,ci_high=None,n=0)
    sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()]);rng=np.random.default_rng(seed);samples=[]
    for start in range(0,n,256):
        ix=rng.integers(0,len(groups),(min(256,n-start),len(groups)));samples.extend((sums[ix].sum(1)/counts[ix].sum(1)).tolist())
    lo,hi=np.quantile(samples,[.025,.975]);return dict(mean=float(sums.sum()/counts.sum()),ci_low=float(lo),ci_high=float(hi),n=int(counts.sum()))

def analyze(cfg,smoke=False):
    ident=study_identity(cfg,smoke);f=directory(cfg,smoke);prepared=Store(f/'prepared',ident).rows();status=f/'status.json'
    if not status.exists() or not json.loads(status.read_text()).get('complete'):
        return dict(status='PENDING_GENERATION',message='No attribution outcomes are invented; run K manually',folder=str(f))
    st=Store(f/'perturbations',ident);units=[];queries=[];curves=[];singles=[];pairs=[];records=[]
    for r in prepared:
        e=r['example'];prefix=[e['dataset'],e['query_id'],r['method'],r['budget']];full=r['original']['task_score'];none=r['none']['task_score'];n=len(r['units']);delta=[]
        def get(action):
            row=st.get(prefix+[action])
            if row is None:raise ValueError('Missing completed perturbation: '+str(prefix+[action]))
            return row
        for i,u in enumerate(r['units']):
            ab=get('LOO:'+str(i));d=full-ab['task_score'];delta.append(d);records.append(ab)
            units.append(dict(dataset=e['dataset'],query_id=e['query_id'],cluster_id=e['cluster_id'],method=r['method'],budget=r['budget'],unit_id=u['unit_id'],unit_index=i,granularity=u['unit_type'],tokens=u['evidence_token_cost'],relevance=u['query_relevance'],parent_retrieval_score=u['parent_retrieval_score'],parent_retrieval_rank=u['parent_retrieval_rank'],importance=d,category='HELPFUL' if d>0 else 'HARMFUL' if d<0 else 'NEUTRAL',gold_overlap=u.get('gold_overlap'),full_score=full,ablated_score=ab['task_score'],ablated_answer=ab['prediction'],exact_flip=ab['exact_answer_changed'],normalized_flip=ab['normalized_answer_changed'],correctness_flip=ab['correctness_changed']))
        rank=causal_order(delta,r['units']);rel=relevance_order(r['units']);top=rel[0] if n else None;best=max(delta) if n else None
        # Jaccard over deterministic top3; source-order ties cannot induce relevance agreement.
        a,b=set(rank[:3]),set(rel[:3]);within=float(spearmanr([u['query_relevance'] for u in r['units']],delta).statistic) if n>1 and np.ptp(delta)>0 and np.ptp([u['query_relevance'] for u in r['units']])>0 else None
        words=[set(u['source_text'].lower().split()) for u in r['units']];red=[len(x&y)/max(1,len(x|y)) for i,x in enumerate(words) for y in words[i+1:]]
        queries.append(dict(dataset=e['dataset'],query_id=e['query_id'],cluster_id=e['cluster_id'],method=r['method'],budget=r['budget'],full_score=full,none_score=none,units=n,evidence_tokens=r['full']['evidence_tokens'],harmful_count=sum(d<0 for d in delta),harmful_fraction=sum(d<0 for d in delta)/n if n else None,helpful_fraction=sum(d>0 for d in delta)/n if n else None,neutral_fraction=sum(d==0 for d in delta)/n if n else None,top1_agreement=float(rank[0]==rel[0]) if n else None,top1_tie_aware=float(delta[top]==best) if n else None,top3_overlap=len(a&b)/max(1,min(3,n)) if n else None,top3_jaccard=len(a&b)/len(a|b) if a|b else None,top_relevance_negative=float(delta[top]<0) if n else None,top_relevance_zero=float(delta[top]==0) if n else None,within_query_rho=within,importance_concentration=max(delta)/sum(max(0,d) for d in delta) if any(d>0 for d in delta) else None,relevance_concentration=max(np.exp(np.array([u['query_relevance'] for u in r['units']])-max(u['query_relevance'] for u in r['units']))/sum(np.exp(np.array([u['query_relevance'] for u in r['units']])-max(u['query_relevance'] for u in r['units'])))) if n else None,redundancy=float(np.mean(red)) if red else None,gold_coverage=sum(u.get('gold_overlap') is True for u in r['units'])/sum(u.get('gold_overlap') is not None for u in r['units']) if any(u.get('gold_overlap') is not None for u in r['units']) else None,top_causal_gold_overlap=r['units'][rank[0]].get('gold_overlap') if n else None))
        for i in sorted(set(rank[:3])|set(rel[:3])):
            v=get('ONLY:'+str(i));records.append(v);singles.append(dict(dataset=e['dataset'],query_id=e['query_id'],cluster_id=e['cluster_id'],method=r['method'],budget=r['budget'],unit_id=r['units'][i]['unit_id'],selected_by_relevance_top3=i in rel[:3],selected_by_causal_top3=i in rank[:3],full_score=full,only_score=v['task_score'],sufficiency_loss=full-v['task_score'],answer_changed=v['exact_answer_changed']))
        for m in (1,2,4):
            if m>n:continue
            only,remove=get('TOP_ONLY:'+str(m)),get('TOP_REMOVE:'+str(m));records.extend([only,remove])
            curves.append(dict(dataset=e['dataset'],query_id=e['query_id'],cluster_id=e['cluster_id'],method=r['method'],budget=r['budget'],m=m,full_score=full,only_score=only['task_score'],remove_score=remove['task_score'],sufficiency_loss=full-only['task_score'],comprehensiveness=full-remove['task_score'],retention=retention(full,none,only['task_score']),only_evidence_tokens=only['evidence_tokens'],only_prompt_tokens=only['prompt_tokens'],remove_evidence_tokens=remove['evidence_tokens']))
    U=pd.DataFrame(units);Q=pd.DataFrame(queries);C=pd.DataFrame(curves);S=pd.DataFrame(singles);correlations=[];summary=[];gold=[];systems=[]
    for key,g in U.groupby(['dataset','budget','method']):
        base=dict(zip(['dataset','budget','method'],key));correlations.append(dict(base,**correlation_ci(g.to_dict('records'),n=200 if smoke else 10000)))
        q=Q[(Q.dataset==key[0])&(Q.budget==key[1])&(Q.method==key[2])];summary.append(dict(base,n_queries=len(q),n_units=len(g),mean_importance=g.importance.mean(),median_importance=g.importance.median(),helpful=(g.importance>0).mean(),neutral=(g.importance==0).mean(),harmful=(g.importance<0).mean(),mean_query_harmful=q.harmful_fraction.mean(),top1=q.top1_agreement.mean(),top1_tie_aware=q.top1_tie_aware.mean(),top3=q.top3_overlap.mean(),top_relevance_negative=q.top_relevance_negative.mean(),top_relevance_zero=q.top_relevance_zero.mean(),within_query_rho=q.within_query_rho.mean(),within_query_valid=int(q.within_query_rho.notna().sum()),exact_answer_flip=g.exact_flip.mean(),correctness_flip=g.correctness_flip.mean()))
        for overlap,h in g[g.gold_overlap.notna()].groupby('gold_overlap'):gold.append(dict(base,gold_overlapping=bool(overlap),n_units=len(h),mean_importance=h.importance.mean(),median_importance=h.importance.median(),helpful=(h.importance>0).mean(),harmful=(h.importance<0).mean()))
    for (d,b,qid),g in Q.groupby(['dataset','budget','query_id']):
        full=g[g.method=='QBGF_FULL'].iloc[0];comp=g[g.method!='QBGF_FULL'].iloc[0];difference=full.full_score-comp.full_score;group='WIN' if difference>0 else 'LOSS' if difference<0 else 'TIE'
        Q.loc[g.index,'qbgf_outcome']=group
        pairs.append(dict(dataset=d,budget=b,query_id=qid,cluster_id=full.cluster_id,harmful_fraction_difference=float(full.harmful_fraction-comp.harmful_fraction) if pd.notna(full.harmful_fraction) and pd.notna(comp.harmful_fraction) else None,quality_difference=difference,group=group,comparator=comp.method))
    pair_df=pd.DataFrame(pairs);effects=[]
    for (d,b),g in pair_df.groupby(['dataset','budget']):effects.append(dict(dataset=d,budget=b,contrast='QBGF - comparator harmful unit fraction (query weighted)',**mean_ci(g.to_dict('records'),'harmful_fraction_difference',n=200 if smoke else 10000)))
    curve_summary=[]
    for (d,b,m,k),g in C.groupby(['dataset','budget','method','m']):
        for measure in ['sufficiency_loss','comprehensiveness','retention']:
            curve_summary.append(dict(dataset=d,budget=b,method=m,top_m=k,measure=measure,**mean_ci(g.to_dict('records'),measure,n=200 if smoke else 10000)))
    from src.week12.statistics import metric
    for r in prepared:
        e=r['example'];systems.append(dict(dataset=e['dataset'],query_id=e['query_id'],cluster_id=e['cluster_id'],method=r['method'],budget=r['budget'],**{k:r['original'][k] for k in ('f1','correct','gold','prediction')}))
    system_summary=[]
    for (d,b,m),g in pd.DataFrame(systems).groupby(['dataset','budget','method']):system_summary.append(dict(dataset=d,budget=b,method=m,n_queries=len(g),quality=metric(g.to_dict('records'),cfg['statistics']['primary_metric'][d]),accuracy=metric(g.to_dict('records'),'accuracy') if d=='SciFact' else None))
    out=f/'analysis';out.mkdir(parents=True,exist_ok=True)
    tables={'unit_importance':U,'query_profiles':Q,'sufficiency_comprehensiveness':C,'single_unit_sufficiency':S,'relevance_utility':pd.DataFrame(correlations),'unit_summary':pd.DataFrame(summary),'gold_overlap':pd.DataFrame(gold),'paired_harmful_effects':pd.DataFrame(effects),'curve_summary':pd.DataFrame(curve_summary),'system_quality':pd.DataFrame(system_summary),'win_loss_pairs':pair_df,'perturbation_cost_and_flips':pd.DataFrame(records)}
    for name,frame in tables.items():frame.to_csv(out/(name+'.csv'),index=False)
    immutable(out/'QUANTITATIVE_RESULTS_FREEZE.json',dict(identity=digest(ident),tables={name:sha(out/(name+'.csv')) for name in tables},no_generated_interpretation=True,retrospective=True))
    cases=clean(case_studies(prepared,U,pair_df));atomic(out/'qualitative_candidates.json',cases)
    plots(out,U,Q,C,pd.DataFrame(summary),pd.DataFrame(gold),smoke)
    (out/'REPORT.md').write_text(('# SYNTHETIC SMOKE ONLY — ' if smoke else '# ')+ 'Evidence input-ablation diagnostic\n\nVALIDATION ONLY. Causal input ablation, not internal-model explanation.\n\n'+pd.DataFrame(summary).to_markdown(index=False)+'\n\n## Relevance versus utility\n\n'+pd.DataFrame(correlations).to_markdown(index=False)+'\n\nZero impact does not establish non-use; redundancy, interactions and unchanged metric can mask sensitivity. Importance is context-dependent, not additive causal credit. Causal ranking and subset evaluation use the same outcomes: these are retrospective diagnostic ceilings. Spearman CIs resample paper/query/document clusters, not independent units; constant-score correlations remain undefined. Gold non-overlap is not evidence of incorrectness. Query-weighted paired harmful fractions account for differing unit counts; token lengths are separately recorded. No runtime efficiency claim is made. Qualitative cases require manual review.\n')
    return dict(status='COMPLETE',folder=str(out),tables=tables,qualitative_cases=cases)

def case_studies(prepared,U,pairs):
    lookup={(r['example']['dataset'],r['example']['query_id'],r['method'],r['budget']):r for r in prepared};cases=[]
    def add(kind,row):
        k=(row['dataset'],row['query_id'],row['method'],int(row['budget']));r=lookup[k]
        cases.append(dict(kind=kind,dataset=k[0],query_id=k[1],method=k[2],budget=k[3],question=r['example']['question'],gold=r['example']['evaluation'],original_answer=r['original']['prediction'],selected_evidence=[dict(unit_id=u['unit_id'],text=u['source_text'],relevance=u['query_relevance']) for u in r['units']],unit= row,interpretation='MANUAL_REVIEW_REQUIRED; no internal reasoning inference'))
    for group in ['WIN','LOSS']:
        selected=pairs[pairs.group==group].copy();selected['magnitude']=selected.quality_difference.abs()
        for _,p in selected.sort_values(['magnitude','query_id'],ascending=[False,True]).head(3).iterrows():
            g=U[(U.dataset==p.dataset)&(U.query_id==p.query_id)&(U.budget==p.budget)&(U.method=='QBGF_FULL')]
            if not g.empty:add('QBGF_'+group,g.sort_values(['importance','unit_index'],ascending=[False,True]).iloc[0].to_dict())
    candidates=[]
    for _,g in U.groupby(['dataset','query_id','method','budget']):
        rel=g.relevance.median();candidates+=g[(g.relevance<rel)&(g.importance>0)].to_dict('records')
    for row in sorted(candidates,key=lambda r:(-r['importance'],r['query_id'],r['unit_index']))[:3]:add('LOW_RELEVANCE_HIGH_IMPORTANCE',row)
    candidates=[]
    for _,g in U.groupby(['dataset','query_id','method','budget']):candidates+=g.sort_values(['relevance','unit_index'],ascending=[False,True]).head(3).query('importance <= 0').to_dict('records')
    for row in sorted(candidates,key=lambda r:(r['importance'],-r['relevance'],r['query_id']))[:3]:add('HIGH_RELEVANCE_HARMFUL_OR_ZERO_ABLATION_EFFECT',row)
    return cases

def plots(out,U,Q,C,summary,gold,smoke=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    folder=out/'figures';folder.mkdir(exist_ok=True)
    for dataset in sorted(U.dataset.unique()):
        for view in range(1,8):
            fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=False)
            for ax,b in zip(axs,[256,1024]):
                u=U[(U.dataset==dataset)&(U.budget==b)];q=Q[(Q.dataset==dataset)&(Q.budget==b)];c=C[(C.dataset==dataset)&(C.budget==b)];s=summary[(summary.dataset==dataset)&(summary.budget==b)]
                for method,g in u.groupby('method'):
                    if view==1:ax.scatter(g.relevance,g.importance,s=12,alpha=.4,label=method);ax.set(xlabel='Unit query relevance',ylabel='Leave-one-out importance')
                    elif view==2:ax.hist(g.importance,bins=21,alpha=.5,label=method);ax.set(xlabel='Leave-one-out importance',ylabel='Units')
                    elif view in (3,4):
                        h=c[c.method==method];v='only_score' if view==3 else 'comprehensiveness';curve=h.groupby('m')[v].mean();ax.plot(curve.index,curve.values,'o-',label=method);ax.set(xlabel='Causal top-m units',ylabel='Subset answer score' if view==3 else 'Full minus removed score')
                        if view==3:ax.axhline(q[q.method==method].full_score.mean(),ls='--',alpha=.5,label=method+' full')
                if view==5 and not s.empty:
                    xs=np.arange(len(s));bottom=np.zeros(len(s))
                    for col,color in [('helpful','#2a9d8f'),('neutral','#999999'),('harmful','#e76f51')]:ax.bar(xs,s[col],bottom=bottom,label=col,color=color);bottom+=s[col].to_numpy()
                    ax.set_xticks(xs,s.method,rotation=20);ax.set_ylabel('Unit fraction')
                if view==6:
                    h=q[q.method=='QBGF_FULL'].groupby('qbgf_outcome')[['harmful_fraction','neutral_fraction','redundancy']].mean();h.plot.bar(ax=ax);ax.set_ylabel('Mean profile (query weighted)')
                if view==7:
                    h=gold[(gold.dataset==dataset)&(gold.budget==b)]
                    for method,g in h.groupby('method'):ax.plot(g.gold_overlapping.astype(int),g.mean_importance,'o-',label=method)
                    ax.set_xticks([0,1],['Non-gold overlap','Gold overlap']);ax.set_ylabel('Mean leave-one-out importance')
                ax.set_title(dataset+' / '+str(b));handles,labels=ax.get_legend_handles_labels()
                if handles:ax.legend(fontsize=7)
            names={1:'relevance_utility',2:'importance_distribution',3:'sufficiency',4:'comprehensiveness',5:'unit_categories',6:'win_loss_profile',7:'gold_utilization'};fig.suptitle(('SYNTHETIC SMOKE ONLY — ' if smoke else '')+names[view].replace('_',' ').title());fig.tight_layout();fig.savefig(folder/(dataset+'_'+names[view]+'.png'));plt.close(fig)


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    return None if isinstance(value,float) and not np.isfinite(value) else value
