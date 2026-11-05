"""All-cell crossfit developmental comparisons; no new model selection."""
import collections
import numpy as np
import pandas as pd
from .protocol import *
from .baselines import baseline
from src.week12.statistics import metric,paired,holm

def analyze(c=None):
    c=c or load();parent,v1,s,cohort,f=freeze(c);store=Store(root(c)/'pilot/rows',f['identity']);groups=collections.defaultdict(list);utility=[];orders=[];cache=[]
    for d,members in cohort['selected'].items():
        for member in members:
            q=member['query_id']
            for b in c['budgets']:
                for m in c['methods']:
                    r=store.get([d,q,m,b]) if m=='BEA_V2' else baseline(c,v1,f,d,q,m,b)
                    if r is None:raise ValueError('Incomplete V2 pilot; resume missing rows before comparing')
                    groups[d,b,m].append(dict(r,method=m))
                    if m=='BEA_V2':
                        for other in c['methods'][:-1]:
                            oldrow=baseline(c,v1,f,d,q,other,b);score='correct' if d=='SciFact' else 'f1';delta=r[score]-oldrow[score]
                            utility.append(dict(dataset=d,budget=b,query_id=q,comparator=other,outcome='win' if delta>0 else 'loss' if delta<0 else 'tie',**{k:v for k,v in r['behavior'].items() if k.startswith('mean_')}))
                        item=Store(root(c)/'pilot/selections',f['identity']).get([d,q,'BEA_V2',b])
                        for transition in item['selection']['trace']:
                            for i,term in enumerate(transition['sentences']):orders.append(dict(dataset=d,budget=b,query_id=q,transition_order=transition['step'],within_transition_order=i,**term))
                    cache.append(dict(dataset=d,query_id=q,budget=b,method=m,source='V2 row' if m=='BEA_V2' else 'read-only frozen control or isolated restoration',conceptual_lm_calls=1))
    summary=[];evidence=[];contrasts=[]
    def mean(rs,key):
        values=[r[key] for r in rs if r.get(key) is not None];return float(np.mean(values)) if values else None
    for (d,b,m),rs in sorted(groups.items()):
        summary.append(dict(label=LABEL,study=c['study'],dataset=d,budget=b,method=m,n_queries=len(rs),n_groups=len({r['cluster_id'] for r in rs}),
                            metric='macro_f1' if d=='SciFact' else 'f1',quality=metric(rs,'macro_f1' if d=='SciFact' else 'f1'),
                            EM=mean(rs,'exact_match') if d!='SciFact' else None,accuracy=metric(rs,'accuracy') if d=='SciFact' else None,
                            evidence_tokens=mean(rs,'evidence_tokens'),prompt_tokens=mean(rs,'prompt_tokens'),generated_tokens=mean(rs,'generated_tokens'),conceptual_lm_calls=len(rs)))
        bs=[r['behavior'] for r in rs]
        evidence.append(dict(dataset=d,budget=b,method=m,unused_tokens=mean(rs,'unused_budget'),budget_utilization=mean(rs,'budget_utilization'),
                             **{key:mean(bs,key) for key in ['selected_units','represented_parents','mean_pairwise_redundancy','aspect_coverage','fidelity_upgrades','mean_selected_p_help','mean_selected_p_harm','mean_selected_D','mean_help_minus_harm_diagnostic','mean_transition_IG','mean_transition_U']},
                             full_frequency=float(np.mean([r.get('full_units',0)>0 for r in bs])),
                             source_expansion_frequency=float(np.mean([r.get('source_expansion',False) for r in bs])) if m=='BEA_V2' else None,
                             threshold_stop_frequency=float(np.mean([r.get('stop_reason')=='LOW_MARGINAL_VALUE' for r in rs]))))
    for d in DATASETS:
        for b in c['budgets']:
            for m in c['methods'][:-1]:contrasts.append(dict(dataset=d,budget=b,left='BEA_V2',right=m,**paired(groups[d,b,'BEA_V2'],groups[d,b,m],'macro_f1' if d=='SciFact' else 'f1',c['statistics']['resamples'],c['statistics']['seed'])))
    for row,p in zip(contrasts,holm([r['raw_p'] for r in contrasts])):row['holm_p']=p;row['exploratory_family_size']=18
    S=pd.DataFrame(summary);E=pd.DataFrame(evidence);C=pd.DataFrame(contrasts);U=pd.DataFrame(utility)
    for name,frame in [('summary',S),('evidence_behavior',E),('paired_comparisons',C),('utility_behavior',U),('utility_by_selection_order',pd.DataFrame(orders)),('all_methods_cache_manifest',pd.DataFrame(cache))]:text(root(c)/'pilot'/(name+'.csv'),frame.to_csv(index=False))
    # Preregistered mechanical developmental gate, no retrospective best-cell promotion.
    gate=c['decision'];v2=[r for rows in groups.values() for r in rows if r['method']=='BEA_V2'];conditions={};degraded=[];improved=[]
    for d in DATASETS:
        for b in c['budgets']:
            cp=next(r for r in contrasts if (r['dataset'],r['budget'],r['right'])==(d,b,'BEA_QBGF_V1'))
            margin=gate['macro_f1_margin'] if d=='SciFact' else gate['qa_margin']
            new=next(r for r in summary if (r['dataset'],r['budget'],r['method'])==(d,b,'BEA_V2'));oldrow=next(r for r in summary if (r['dataset'],r['budget'],r['method'])==(d,b,'BEA_QBGF_V1'))
            if cp['ci_high']< -margin:degraded.append([d,b])
            if (cp['point']>=gate['minimum_quality_gain'] and new['evidence_tokens']<=1.05*oldrow['evidence_tokens']) or (cp['point']>=-margin and new['evidence_tokens']<=.85*oldrow['evidence_tokens']):improved.append([d,b])
    predictor=pd.read_csv(root(c)/'utility_model/fold_metrics.csv');auc=predictor[predictor.target=='HELPFUL'].roc_auc.dropna()
    conditions=dict(no_consistent_degradation=len(degraded)<=gate['maximum_clearly_degraded_cells'],tradeoff_improvement_present=bool(improved),
                    helpful_predictor_not_globally_uninformative=bool(len(auc) and auc.mean()>gate['uninformative_mean_auc_ceiling']),
                    no_empty_collapse=float(np.mean([r['evidence_tokens']==0 for r in v2]))<=gate['maximum_empty_fraction'],
                    no_full_budget_collapse=float(np.mean([r['budget_utilization']>=.95 for r in v2]))<=gate['maximum_near_full_fraction'])
    decision=dict(label=LABEL,decision='PROMISING' if all(conditions.values()) else 'NOT PROMISING',conditions=conditions,
                  clearly_degraded_vs_v1=degraded,improved_tradeoff_cells=improved,interpretation='Developmental crossfit gate only; nominal paired uncertainty is not model-training uncertainty; no fresh confirmation or automatic expansion')
    write(root(c)/'pilot/PILOT_DECISION.json',dict(decision,identity_hash=f['identity_hash']))
    text(root(c)/'pilot/PILOT_DECISION.md','# BEA_V2 developmental decision\n\n```json\n'+json.dumps(decision,indent=2)+'\n```\n\nNo automatic unseen expansion. Point-competitive efficiency is not proven statistical noninferiority.\n')
    # All plots index the complete three-field key; no outcome-driven omitted cells.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root(c)/'pilot/figures';folder.mkdir(parents=True,exist_ok=True)
    for field in ['quality','evidence_tokens']:
        fig,axs=plt.subplots(2,3,figsize=(14,7),squeeze=False)
        indexed={(r['dataset'],r['budget'],r['method']):r for r in summary}
        for j,d in enumerate(DATASETS):
            for i,b in enumerate(c['budgets']):
                axs[i,j].bar(range(4),[indexed[d,b,m][field] for m in c['methods']]);axs[i,j].set_xticks(range(4),['FIXED','QBGF','BEA1','BEA2']);axs[i,j].set_title(f'{d} · {b}');axs[i,j].set_ylabel(field)
        fig.tight_layout();fig.savefig(folder/(field+'.png'),dpi=160);plt.close(fig)
    questions=['Can historical attribution predict helpful evidence?','Does utility multiplication improve BEA-v1?','Does it recover Hotpot/QASPER quality?','Does it retain SciFact efficiency/quality?','Does it spend more on under-allocated queries?','Does it still reduce redundancy?','Does predicted harm suppress candidates?','Does source fidelity become active?','Does usefulness transfer across held-out groups?','Is the remaining bottleneck optimizer, representation or utility estimation?']
    content='# BEA_V2 crossfit development pilot\n\n'+S.to_markdown(index=False)+'\n\n## Paired comparisons\n\n'+C.to_markdown(index=False)+'\n\n## Evidence behavior\n\n'+E.to_markdown(index=False)+'\n\n## Selected utility by wins/losses\n\n'+U.groupby(['dataset','budget','comparator','outcome']).mean(numeric_only=True).to_markdown()+'\n\n## Required research questions\n\n'
    for i,q in enumerate(questions,1):content+=f'{i}. **{q}** '+(['See held-out fold AUC/PR-AUC/Brier/calibration in UTILITY_MODEL_REPORT.md.','See all six BEA2−BEA1 paired cells; no universal claim from one favorable cell.','Inspect Hotpot/QASPER comparisons against all controls at both budgets.','Inspect SciFact macro-F1 and accuracy separately; no pooled score.','Compare per-query unused tokens versus V1; observational association, not a causal ablation.','Compare defined mean pairwise redundancy in the evidence table.','P_harm is diagnostic-only and does not suppress allocation; the primary weight is P_help. This experiment does not claim reliable harmful-evidence filtering.','See FULL/source expansion frequency and exact transition traces; no separate causal FULL ablation.','OOF units/groups are held out, but this previously analyzed cohort remains developmental; unseen VAL is needed.','Unresolved causally: the combined method changes weighting/allocation; individual LOO misses interactions.'][i-1])+'\n\n'
    content+='## Limits\n\nHistorical LOO is intervention-specific; neutral does not mean ignored/useless. Training uses previously selected sentences, not unbiased candidates; usefulness is predictive, not causal necessity. The primary downstream weight D equals P_help; P_harm and the signed help-minus-harm difference are diagnostics only. Static models may miss interactions and distribution shift. SciFact labels use claim correctness; its aggregate primary metric is macro-F1. The pilot population has already been analyzed. Frozen reader/serializer/decoding are shared; cache hits are conceptual LM calls, not zero-cost systems. Token counts and recorded generation timing do not establish end-to-end latency. Unseen expansion is separate and disabled.\n'
    text(root(c)/'pilot/PILOT_REPORT.md',content);write(root(c)/'pilot/analysis_complete.json',dict(identity_hash=f['identity_hash'],rows=1200))
    return dict(summary=S,paired_comparisons=C,evidence_behavior=E,decision=decision)
