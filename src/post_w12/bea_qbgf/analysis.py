"""Exploratory paired VAL metrics, prespecified pilot gate and cautious reports."""
import collections,json
import numpy as np
import pandas as pd
from .protocol import load,freeze,output,save,text,read,LABEL,METHODS,digest
from src.post_w12.reporting import record as record_reporting_repair
from .runner import logical,stores
from src.week12.statistics import metric,paired,holm

def mean(rows,key):
    values=[r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None

def analyze(c=None,stage='pilot'):
    c=c or load();parent,s,cohort,f=freeze(c)
    record_reporting_repair(c,f)
    if stage=='expanded':cohort=read(output(c)/'expanded/EXPANDED_COHORT.json')
    _,_,store,_=stores(c,f);keys=logical(cohort,c);rows=[store.get(k) for k in keys]
    if any(r is None for r in rows):raise ValueError('Analysis requires every exact paired row; resume missing work first')
    expected={(k[0],k[1],k[2],k[3]) for k in keys}
    if {(r['dataset'],r['query_id'],r['method'],r['budget']) for r in rows}!=expected:raise ValueError('Paired membership mismatch')
    groups=collections.defaultdict(list)
    for r in rows:groups[r['dataset'],r['budget'],r['method']].append(r)
    summary=[];evidence=[];comparisons=[]
    for (d,b,m),rs in sorted(groups.items()):
        name='macro_f1' if d=='SciFact' else 'f1';ev=[dict(r['behavior'],evidence_tokens=r['evidence_tokens'],unused_budget=r['unused_budget'],budget_utilization=r['budget_utilization']) for r in rs]
        summary.append(dict(label=LABEL,dataset=d,budget=b,method=m,n_queries=len(rs),n_groups=len({r['cluster_id'] for r in rs}),metric=name,
                            quality=metric(rs,name),EM=mean(rs,'exact_match') if d!='SciFact' else None,
                            accuracy=metric(rs,'accuracy') if d=='SciFact' else None,
                            invalid_format=sum(r['prediction'] not in parent['labels'] for r in rs) if d=='SciFact' else None,
                            evidence_tokens=mean(rs,'evidence_tokens'),prompt_tokens=mean(rs,'prompt_tokens'),generated_tokens=mean(rs,'generated_tokens'),
                            physical_generation_calls_original_receipts=sum(r['physical_generation_performed'] for r in rs),conceptual_lm_calls=len(rs)))
        evidence.append(dict(label=LABEL,dataset=d,budget=b,method=m,n_queries=len(rs),
                             **{k:sum(r.get(k,0) for r in ev) for k in ['full_eligible_query','full_considered','full_chosen','full_rejected_cost','full_rejected_low_gain','full_rejected_threshold','full_lost_competition','full_blocked_one_sentence']},
                             **{k:mean(ev,k) for k in ['evidence_tokens','unused_budget','budget_utilization','selected_units','represented_parents','sentence_units','full_units','mean_pairwise_redundancy','aspect_coverage','represented_aspects','tokens_per_represented_aspect','fidelity_upgrades','empty_evidence','stopped_with_unused_budget']},
                             pairwise_redundancy_defined_queries=sum(r['mean_pairwise_redundancy'] is not None for r in ev),
                             full_representation_frequency=float(np.mean([r['full_units']>0 for r in ev])),
                             upgrade_cell_fraction=float(np.mean([r['fidelity_upgrades']>0 for r in ev]))))
    for d in cohort['selected']:
        for b in c['budgets']:
            for other in ['FIXED_SENTENCE','QBGF_FULL']:
                comparisons.append(dict(label=LABEL,dataset=d,budget=b,left='BEA_QBGF',right=other,
                     **paired(groups[d,b,'BEA_QBGF'],groups[d,b,other],'macro_f1' if d=='SciFact' else 'f1',c['statistics']['resamples'],c['statistics']['seed'])))
    adjusted=holm([r['raw_p'] for r in comparisons])
    for r,p in zip(comparisons,adjusted):r.update(exploratory_family_size=len(comparisons),holm_p=p)
    directory=output(c)/stage
    frames=dict(summary=summary,paired_comparisons=comparisons,evidence_behavior=evidence)
    for name,values in frames.items():text(directory/(name+'.csv'),pd.DataFrame(values).to_csv(index=False))
    # SciFact class F1 is aggregated, never a mean of per-claim F1.
    from src.week11.evaluation import classification_metrics
    classes=[]
    for (d,b,m),rs in groups.items():
        if d=='SciFact':classes.append(dict(dataset=d,budget=b,method=m,**classification_metrics([r['gold'] for r in rs],[r['prediction'] for r in rs])))
    text(directory/'scifact_class_metrics.json',json.dumps(dict(label=LABEL,rows=classes),indent=2))
    decision=classify(c,rows,summary,evidence,comparisons)
    if stage=='pilot':
        save(output(c)/'PILOT_DECISION.json',dict(decision,identity_hash=f['identity_hash'],rows_hash=digest(rows)))
        text(output(c)/'PILOT_DECISION.md','# BEA-QBGF pilot decision\n\nPOST_W12_EXPLORATORY; not confirmation.\n\n```json\n'+json.dumps(decision,indent=2)+'\n```\n\nExpansion requires both PROMISING and a separate manual notebook flag. No final W12 method changes.\n')
    plots(directory,summary,evidence)
    report(c,stage,summary,evidence,comparisons,decision,rows)
    save(directory/'analysis_complete.json',dict(label=LABEL,identity_hash=f['identity_hash'],rows_hash=digest(rows),logical_rows=len(rows)))
    return dict(summary=pd.DataFrame(summary),paired_comparisons=pd.DataFrame(comparisons),evidence_behavior=pd.DataFrame(evidence),decision=decision)

def classify(c,rows,summary,evidence,comparisons):
    gate=c['pilot_gate'];quality={(r['dataset'],r['budget'],r['method']):r for r in summary};ev={(r['dataset'],r['budget'],r['method']):r for r in evidence}
    cells=sorted({(r['dataset'],r['budget']) for r in summary});degraded=[];meaningful=[]
    for d,b in cells:
        margin=gate['macro_f1_margin'] if d=='SciFact' else gate['qa_margin'];cp=[r for r in comparisons if (r['dataset'],r['budget'])==(d,b)]
        if any(r['ci_high']< -margin for r in cp):degraded.append([d,b])
        a=ev[d,b,'BEA_QBGF'];q=quality[d,b,'BEA_QBGF']['quality']
        point_competitive=all(q>=quality[d,b,m]['quality']-margin for m in ['FIXED_SENTENCE','QBGF_FULL'])
        if any(r['point']>=margin and r['ci_low']>0 for r in cp):meaningful.append(dict(dataset=d,budget=b,dimension='quality',note='nominal exploratory paired CI'))
        if point_competitive:
            for other in ['FIXED_SENTENCE','QBGF_FULL']:
                old=ev[d,b,other];reduction=1-a['evidence_tokens']/old['evidence_tokens'] if old['evidence_tokens'] else 0
                red=(old['mean_pairwise_redundancy']-a['mean_pairwise_redundancy']) if all(x is not None for x in [old['mean_pairwise_redundancy'],a['mean_pairwise_redundancy']]) else 0
                cov=a['aspect_coverage']-old['aspect_coverage']
                comparable_cost=a['evidence_tokens']<=1.05*old['evidence_tokens']
                if reduction>=gate['meaningful_token_reduction'] or (comparable_cost and (red>=gate['meaningful_redundancy_reduction'] or cov>=gate['meaningful_coverage_increase'])):
                    meaningful.append(dict(dataset=d,budget=b,dimension='efficiency/coverage/redundancy',comparator=other,
                         token_reduction=reduction,redundancy_reduction=red,coverage_change=cov,note='quality point estimates competitive; not proven statistical noninferiority'))
    bea=[r for r in rows if r['method']=='BEA_QBGF'];fixed={(r['dataset'],r['query_id'],r['budget']):r for r in rows if r['method']=='FIXED_SENTENCE'}
    different=float(np.mean([r['evidence_hash']!=fixed[r['dataset'],r['query_id'],r['budget']]['evidence_hash'] for r in bea]))
    upgrades=float(np.mean([r['behavior']['fidelity_upgrades']>0 for r in bea]));empty=float(np.mean([r['behavior']['empty_evidence'] for r in bea]))
    conditions=dict(not_clearly_degraded_in_most_cells=len(degraded)<=gate['max_clearly_degraded_cells'],
                    meaningful_dimension_present=bool(meaningful),different_from_fixed=different>=gate['min_different_context_fraction'],
                    escalation_active=upgrades>=gate['min_upgrade_cell_fraction'],not_degenerate_empty=empty<=gate['max_empty_cell_fraction'])
    return dict(label=LABEL,decision='PROMISING' if all(conditions.values()) else 'NOT PROMISING',conditions=conditions,
                clearly_degraded_cells=degraded,meaningful_dimensions=meaningful,different_context_fraction=different,
                upgrade_cell_fraction=upgrades,empty_cell_fraction=empty,
                interpretation='Global preregistered exploratory gate; no dataset-specific promotion, no proof of noninferiority, no final-method replacement')

def plots(directory,summary,evidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (directory/'figures').mkdir(parents=True,exist_ok=True)
    for field,records in [('quality',summary),('evidence_tokens',evidence),('aspect_coverage',evidence),('mean_pairwise_redundancy',evidence)]:
        indexed={}
        for row in records:
            key=(row['dataset'],row['budget'],row['method'])
            if key in indexed:raise ValueError('Duplicate plotting cell: '+str(key))
            indexed[key]=row
        required={(d,b,m) for d in ['HotpotQA','QASPER','SciFact'] for b in [256,1024] for m in METHODS}
        missing=required-indexed.keys()
        if missing:raise ValueError('Missing plotting cells: '+str(sorted(missing)))
        fig,axs=plt.subplots(2,3,figsize=(13,7),squeeze=False)
        for j,d in enumerate(['HotpotQA','QASPER','SciFact']):
            for i,b in enumerate([256,1024]):
                values=[indexed[d,b,m].get(field) for m in METHODS]
                axs[i,j].bar(range(3),[v if v is not None else float('nan') for v in values]);axs[i,j].set_xticks(range(3),['FIXED','QBGF','BEA']);axs[i,j].set_title(f'{d} · {b}');axs[i,j].set_ylabel(field)
        fig.suptitle('POST-W12 exploratory VAL · task metrics remain separate');fig.tight_layout();fig.savefig(directory/'figures'/(field+'.png'),dpi=160);plt.close(fig)

def report(c,stage,summary,evidence,comparisons,decision,rows):
    title='PILOT_REPORT.md' if stage=='pilot' else 'EXPANDED_REPORT.md'
    content=f'# BEA-QBGF {stage} — POST_W12_EXPLORATORY\n\nDecision: **{decision["decision"]}**. Frozen Week12 remains SFR top10 → FIXED_SENTENCE → Qwen3-8B BASE_FP16. These are reused VAL examples, not fresh confirmation or OOD.\n\n'
    content+='## All quality and cost cells\n\n'+pd.DataFrame(summary).to_markdown(index=False)+'\n\n## Paired differences\n\n'+pd.DataFrame(comparisons).to_markdown(index=False)+'\n\n## Evidence behavior\n\n'+pd.DataFrame(evidence).to_markdown(index=False)+'\n\n'
    content+='## Questions answered\n\n'
    for number,(question,answer) in enumerate([
        ('Better than QBGF?','See BEA−QBGF paired differences for every dataset/budget; mixed cells do not establish universal superiority.'),
        ('Competitive with FIXED_SENTENCE?','See BEA−FIXED paired intervals. Point competitiveness in the pilot gate is not a formal noninferiority result.'),
        ('Less redundancy?','Compare mean_pairwise_redundancy with its defined-query counts; coverage of FULL uses its constituent sentences for comparability.'),
        ('More query-aspect coverage?','Compare aspect_coverage under identical query-only aspects. Cosine coverage is a surrogate, not verified fact coverage.'),
        ('Stops with unused tokens?','See unused_budget, stopped_with_unused_budget and saved per-step stop reasons.'),
        ('Fidelity escalation active?',f"Upgrade-cell fraction: {decision['upgrade_cell_fraction']:.4f}; inspect source accumulation and FULL actions in cache/selections."),
        ('Does expansion itself improve answers?','Unresolved: frequency is measured, but there is no separate BEA-without-expansion primary arm. Do not infer causal benefit from activation or between-method scores.'),
        ('Gains at 256 or 1024?','The complete paired table reports both budgets; do not choose a budget after outcomes.'),
        ('Dataset dependence?','Report HotpotQA/QASPER answer F1 and SciFact aggregate macro-F1 separately; never pool them.'),
        ('Top-down relevance-mass hypothesis supported?','Mechanism diagnostics can support or weaken its plausibility, but without new LOO this remains unresolved as a claim about neutral downstream evidence. Relevance/coverage/redundancy proxies are not downstream utility labels.')],1):
        content+=f'{number}. **{question}** {answer}\n\n'
    content+='## Scope and costs\n\nAll three methods share exact inputs, frozen serializer, reader, prompt/parser, precision and decoding. Physical cache reuse is recorded per row; every system still costs one conceptual answer call. Embedding lookup/physical/original preprocessing costs and query/aspect computation are recorded in cache/problems; they are distinct from generation. No end-to-end latency or serving-throughput claim is made. No gold/LOO/answer-score fields enter allocation. Novelty is diagnostic only, not a duplicate reward. FULL utility sums sequentially recomputed positive marginal sentence surrogates, updating redundancy and coverage after hypothetical additions; it remains a surrogate rather than learned downstream contextual utility.\n\n'
    content+='Optional attribution is not run automatically. Neutral LOO means zero score change under removal, not useless or unobserved evidence. Expanded VAL is blocked unless the pilot passes and the user enables it; it is not independent confirmation. Historical W12 confirmation/final freeze remain unchanged.\n'
    content+='Physical call totals in summary are original creation receipts, including pilot rows reused in expanded VAL. For physical calls in the latest invocation, use that stage’s cache_manifest.csv; reused rows are never presented as newly generated.\n'
    text(output(c)/stage/title,content)
    text(output(c)/'FINAL_POST_W12_REPORT.md',content)
