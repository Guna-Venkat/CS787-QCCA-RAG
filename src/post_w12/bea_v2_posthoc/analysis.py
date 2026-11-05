"""Descriptive engineering comparison, never a new calibration/promotion gate."""
import collections
import numpy as np
import pandas as pd
from .protocol import *
from .runner import stores
from src.post_w12.bea_v2.data import problem
from src.week12.statistics import metric,paired,holm

def mean(rs,key):
    values=[r[key] for r in rs if r.get(key) is not None]
    return float(np.mean(values)) if values else None

def represented(row,p):
    ids=set(row['selected_unit_ids']);units={u['unit_id']:u for u in p['pool']+p['sentences']}
    if not ids<=units.keys():raise ValueError('Unknown selected source unit')
    parents={units[i]['passage_id'] for i in ids}
    full={i for i in ids if units[i]['unit_type']=='PASSAGE'}
    sentences={u['unit_id'] for u in p['sentences'] if u['unit_id'] in ids or u['passage_id'] in full}
    return parents,sentences

def jaccard(a,b):
    return len(a&b)/len(a|b) if a|b else 1.

def analyze(c=None):
    c,old,parent,v1,s,cohort,f=context(c)
    with lock(c):
        complete=read(output(c)/'COMPLETE.json')
        if (complete['identity_hash']!=f['identity_hash'] or complete['logical_rows']!=1200
            or complete['keys_hash']!=digest(keys(cohort))):
            raise ValueError('Analysis requires the complete exact 1,200-row post-hoc grid')
        selections,store,_=stores(c,f)
        rows=[store.get(k) for k in keys(cohort)]
        if any(r is None for r in rows):raise ValueError('Incomplete grid: resume first')
        expected={tuple(k) for k in keys(cohort)}
        if {(r['dataset'],r['query_id'],r['method'],r['budget']) for r in rows}!=expected:
            raise ValueError('Unexpected/duplicate/missing paired membership')
        extra=[p for p in store.path.glob('*.json') if p.name not in {store.file(k).name for k in keys(cohort)}]
        if extra:raise ValueError('Unexpected result shards in post-hoc namespace')
        groups=collections.defaultdict(list)
        for r in rows:
            if r['label']!=LABEL or r['identity_hash']!=f['identity_hash']:raise ValueError('Wrong result label/identity')
            groups[r['dataset'],r['budget'],r['method']].append(r)
        summary=[];evidence=[];comparisons=[];changes=[];utility=[]
        for (d,b,m),rs in sorted(groups.items()):
            name='macro_f1' if d=='SciFact' else 'f1';quality=metric(rs,name);tokens=mean(rs,'evidence_tokens')
            summary.append(dict(label=LABEL,dataset=d,budget=b,method=m,n_queries=len(rs),
                n_groups=len({r['cluster_id'] for r in rs}),metric=name,quality=quality,
                F1=quality if d!='SciFact' else None,macro_F1=quality if d=='SciFact' else None,
                EM=mean(rs,'exact_match') if d!='SciFact' else None,
                accuracy=metric(rs,'accuracy') if d=='SciFact' else None,
                invalid_format=sum(r['prediction'] not in parent['labels'] for r in rs) if d=='SciFact' else None,
                evidence_tokens=tokens,prompt_tokens=mean(rs,'prompt_tokens'),generated_tokens=mean(rs,'generated_tokens'),
                quality_per_1000_evidence_tokens=1000*quality/tokens if tokens else None,
                conceptual_lm_calls=len(rs),new_reader_calls=sum(r['physical_generation_performed'] for r in rs)))
            bs=[r['behavior'] for r in rs]
            evidence.append(dict(label=LABEL,dataset=d,budget=b,method=m,
                evidence_tokens=tokens,unused_budget=mean(rs,'unused_budget'),budget_utilization=mean(rs,'budget_utilization'),
                **{k:mean(bs,k) for k in ['selected_units','represented_parents','mean_pairwise_redundancy',
                    'aspect_count','aspect_coverage','represented_aspects','mean_selected_p_help',
                    'mean_selected_p_harm','mean_transition_IG','mean_transition_U']},
                redundancy_defined_queries=sum(x.get('mean_pairwise_redundancy') is not None for x in bs),
                full_frequency=float(np.mean([x.get('full_units',0)>0 for x in bs])),
                threshold_stop_frequency=float(np.mean([x.get('stop_reason')=='LOW_MARGINAL_VALUE' for x in rs])),
                near_full_frequency=float(np.mean([x['budget_utilization']>=.95 for x in rs])),
                empty_contexts=sum(x['evidence_tokens']==0 for x in rs)))
        for d in DATASETS:
            for b in BUDGETS:
                a=groups[d,b,'BEA_V2']
                for m in METHODS[:-1]:
                    comparisons.append(dict(label=LABEL,dataset=d,budget=b,left='BEA_V2',right=m,
                        **paired(a,groups[d,b,m],'macro_f1' if d=='SciFact' else 'f1',10000,42)))
                lookup={m:{r['query_id']:r for r in groups[d,b,m]} for m in METHODS}
                for member in cohort['selected'][d]:
                    q=member['query_id'];new=lookup['BEA_V2'][q];p=problem(old,v1,d,q)
                    newparents,newsent=represented(new,p)
                    for m in METHODS[:-1]:
                        prior=lookup[m][q];parents,sent=represented(prior,p)
                        changes.append(dict(label=LABEL,dataset=d,budget=b,query_id=q,comparator=m,
                            delta_evidence_tokens=new['evidence_tokens']-prior['evidence_tokens'],
                            delta_utilization=new['budget_utilization']-prior['budget_utilization'],
                            parents_changed=newparents!=parents,sentences_changed=newsent!=sent,
                            parent_jaccard=jaccard(newparents,parents),sentence_jaccard=jaccard(newsent,sent),
                            new_full=new['behavior']['full_units']>0,old_full=prior['behavior']['full_units']>0,
                            new_stop=new['stop_reason'],old_stop=prior['stop_reason']))
                        score='correct' if d=='SciFact' else 'f1';delta=new[score]-prior[score]
                        utility.append(dict(label=LABEL,dataset=d,budget=b,query_id=q,comparator=m,
                            outcome='win' if delta>0 else 'loss' if delta<0 else 'tie',
                            outcome_metric='per-claim correctness' if d=='SciFact' else 'answer F1',
                            delta=delta,**{k:v for k,v in new['behavior'].items() if k.startswith('mean_')}))
        for r,pval in zip(comparisons,holm([r['raw_p'] for r in comparisons])):
            r.update(holm_p=pval,exploratory_family_size=18)
        S,E,C,D,U=map(pd.DataFrame,[summary,evidence,comparisons,changes,utility])
        for name,frame in [('POSTHOC_SUMMARY',S),('POSTHOC_PAIRED_COMPARISONS',C),
            ('POSTHOC_EVIDENCE_BEHAVIOR',E),('POSTHOC_SELECTION_CHANGES',D),('POSTHOC_UTILITY_BEHAVIOR',U)]:
            text(output(c)/(name+'.csv'),frame.to_csv(index=False))
        desc=D.groupby(['dataset','budget','comparator']).agg(
            mean_token_change=('delta_evidence_tokens','mean'),parent_change_fraction=('parents_changed','mean'),
            sentence_change_fraction=('sentences_changed','mean'),mean_parent_jaccard=('parent_jaccard','mean'),
            mean_sentence_jaccard=('sentence_jaccard','mean')).reset_index()
        text(output(c)/'POSTHOC_SELECTION_CHANGE_SUMMARY.csv',desc.to_csv(index=False))
        by={(r['dataset'],r['budget'],r['method']):r for r in summary}
        ev={(r['dataset'],r['budget'],r['method']):r for r in evidence}
        def cell(d,b):
            vals=[]
            for m in METHODS[:-1]:
                r=next(x for x in comparisons if (x['dataset'],x['budget'],x['right'])==(d,b,m))
                vals.append(f"vs {m}: delta {r['point']:+.4f}, nominal 95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}], Holm p={r['holm_p']:.4g}")
            return f"{d} @{b}: "+'; '.join(vals)+'.'
        answers=[]
        answers.append('**HotpotQA recovery:** '+ ' '.join(cell('HotpotQA',b) for b in BUDGETS)+' Recovery must be assessed against both V1 and the simple controls, not V1 alone.')
        answers.append('**QASPER deficit:** '+ ' '.join(cell('QASPER',b) for b in BUDGETS)+' A positive point change alone does not establish recovery.')
        answers.append('**SciFact @1024:** '+cell('SciFact',1024)+' Primary metric is three-class macro-F1; per-query utility win/loss uses correctness only.')
        answers.append('**Exposure versus V1:** '+ ' '.join(f"{d} @{b}: {by[d,b,'BEA_V2']['evidence_tokens']:.1f} vs {by[d,b,'BEA_QBGF_V1']['evidence_tokens']:.1f} mean tokens." for d in DATASETS for b in BUDGETS)+' Per-query token changes are in POSTHOC_SELECTION_CHANGES.csv; association is not a causal test of under-allocation.')
        answers.append('**Token savings:** '+ ' '.join(f"{d} @{b}: "+', '.join(f"{m} {100*(1-by[d,b,'BEA_V2']['evidence_tokens']/by[d,b,m]['evidence_tokens']):+.1f}%" if by[d,b,m]['evidence_tokens'] else f'{m}: undefined (zero tokens)' for m in METHODS[:2])+'.' for d in DATASETS for b in BUDGETS)+' Positive means savings; negative means increased exposure.')
        answers.append('**Quality per token:** POSTHOC_SUMMARY.csv reports task quality / mean evidence tokens ×1,000. This descriptive ratio can favor short contexts despite poor quality; it is not a calibrated utility or latency metric.')
        answers.append('**Selection changes:** Parent and underlying-sentence changes/Jaccard are shown below, including all sentences exposed by FULL. Differences show changed allocation, not that selected evidence is causally necessary.')
        answers.append('**FULL activation:** '+ ' '.join(f"{d} @{b}: BEA_V2 {ev[d,b,'BEA_V2']['full_frequency']:.1%}, V1 {ev[d,b,'BEA_QBGF_V1']['full_frequency']:.1%}." for d in DATASETS for b in BUDGETS))
        answers.append('**Near-full allocation:** '+ ' '.join(f"{d} @{b}: {ev[d,b,'BEA_V2']['near_full_frequency']:.1%} of queries use ≥95% of budget; threshold stops {ev[d,b,'BEA_V2']['threshold_stop_frequency']:.1%}." for d in DATASETS for b in BUDGETS)+' Compare exposure and quality jointly; no calibration rule is reclassified as passed.')
        answers.append('**Downstream usefulness signal:** These combined-method comparisons describe P_help-weighted BEA at one post-hoc operating point. They cannot isolate the causal contribution of P_help from threshold/allocation differences without a matched unweighted ablation. Cross-fitting prevents same-group predictor fitting, but this previously analyzed cohort is not fresh evidence; bootstrap intervals do not include predictor-training uncertainty.')
        content='# BEA_V2 post-hoc engineering results\n\n'+NOTICE+'\n\nOfficial calibration remains **CALIBRATION_FAILED_TERMINATED**. No further threshold search or automatic expansion.\n\n## Quality and cost\n\n'+S.to_markdown(index=False)+'\n\n## Paired contrasts\n\n'+C.to_markdown(index=False)+'\n\nNominal paired cluster-bootstrap 95% intervals; raw p uses the existing two-sided cluster label-swap test. Holm adjustment covers all 18 exploratory comparisons. Grouping is unchanged: QASPER paper, SciFact frozen cited-document component, HotpotQA query. No pooled cross-dataset score.\n\n## Evidence behavior\n\n'+E.to_markdown(index=False)+'\n\n## Selection changes\n\n'+desc.to_markdown(index=False)+'\n\n## Ten requested questions\n\n'+'\n\n'.join(f'{i}. {a}' for i,a in enumerate(answers,1))
        content+='\n\n## Limits\n\n150 previously analyzed development queries, 50/dataset; operating point selected after allocation diagnostics. Historical attribution labels are context-specific observational training targets. P_harm is diagnostic-only. Undefined redundancy for fewer than two underlying sentences is excluded and its denominator is reported. Cache reuse preserves conceptual LM calls; evidence tokens and recorded answer timing do not establish end-to-end latency. Confirmation and QASPER TEST are excluded. The frozen project final policy is unchanged.\n'
        text(output(c)/'POSTHOC_PILOT_RESULTS.md',content)
        text(output(c)/'POSTHOC_DECISION.md','# Post-hoc engineering closeout\n\n'+NOTICE+'\n\nStatus: **EXPLORATORY_EVALUATION_COMPLETE — STOP**.\n\nThe preregistered calibration remains **CALIBRATION_FAILED_TERMINATED**. No threshold tuning, method promotion, new attribution, unseen expansion or further runs are triggered by these results. Interpret all six paired cells and uncertainties in POSTHOC_PILOT_RESULTS.md. This is not a successful continuation of the terminated protocol.\n\n'+'\n\n'.join(cell(d,b) for d in DATASETS for b in BUDGETS)+'\n')
        write(output(c)/'ANALYSIS_COMPLETE.json',dict(label=LABEL,identity_hash=f['identity_hash'],
            logical_rows=1200,rows_hash=digest(rows),official_calibration='CALIBRATION_FAILED_TERMINATED',automatic_expansion=False))
        return dict(summary=S,evidence_behavior=E,paired_comparisons=C,selection_changes=desc)
