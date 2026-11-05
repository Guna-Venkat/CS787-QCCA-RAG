"""Five-method matched comparison; 24 fixed exploratory paired contrasts."""
import collections
import numpy as np
import pandas as pd
from .protocol import *
from src.week12.statistics import metric,paired,holm

NAMES={'FIXED_SENTENCE':'FIXED','QBGF_FULL':'QBGF','BEA_QBGF_V1':'BEA-V1','BEA_V2':'BEA-V2',METHOD:'MiniLM Reranker'}
CONCEPTUAL='''| Method | Evidence value | Allocation behavior |
| --- | --- | --- |
| FIXED_SENTENCE | embedding query–sentence relevance | static global ranking + packing |
| MiniLM reranker | cross-encoder query–sentence relevance | static global ranking + packing |
| QBGF | hierarchical passage/sentence relevance mass | joint DROP/SENT/FULL budget optimization |
| BEA-V1 | relevance + coverage − redundancy | dynamic marginal selection + stopping |
| BEA-V2 | P_help × positive semantic IG | dynamic learned-utility selection + stopping |
'''
CLOSEOUT=('We began with query-conditioned evidence quantity, moved to evidence identity and granularity, '
 'then tested relevance-aware and downstream-utility-aware allocation. The MiniLM cross-encoder baseline '
 'provides a conventional reranking control over the same SFR candidate pool. Together, the comparison '
 'distinguishes simple relevance reranking from budgeted granularity allocation and dynamic marginal-utility selection. '
 'The official frozen policy remains SFR top-10 → FIXED_SENTENCE → Qwen3-8B BASE_FP16 → no LoRA. '
 'This is a final POST-W12 exploratory baseline, not a replacement for the frozen method.')

def average(rs,field):
    vals=[r[field] for r in rs if r.get(field) is not None]
    return float(np.mean(vals)) if vals else None

def comparison_table(summaries):
    by={(r['dataset'],r['budget'],r['method']):r for r in summaries}
    result=[];wins={m:0 for m in METHODS}
    for d in DATASETS:
        for b in BUDGETS:
            if any((d,b,m) not in by for m in METHODS):raise ValueError('Incomplete five-method summary')
            best=max(by[d,b,m]['quality'] for m in METHODS)
            winning=[m for m in METHODS if abs(by[d,b,m]['quality']-best)<=1e-12]
            row={'Dataset':d,'Budget':b}
            for m in METHODS:
                value=by[d,b,m]['quality'];row[NAMES[m]]=f'**{value:.4f}**' if m in winning else f'{value:.4f}'
            row['Winner']=' / '.join(NAMES[m] for m in winning)
            result.append(row)
            for m in winning:wins[m]+=1
    return pd.DataFrame(result),wins

def analyze():
    with lock():
        ctx=context();_,_,rows,_=stores(ctx);complete=read(OUT/'full/COMPLETE.json')
        expected=[[d,m['query_id'],METHOD,b] for d in DATASETS for m in ctx['cohort']['selected'][d] for b in BUDGETS]
        if (complete['rows']!=300 or complete['identity_hash']!=ctx['record']['identity_hash']
            or complete['keys_hash']!=digest(expected)):
            raise ValueError('Analysis requires the exact complete 300-answer MiniLM surface')
        if {p.name for p in rows.path.glob('*.json')}!={rows.file(k).name for k in expected}:
            raise ValueError('Missing/unexpected MiniLM result membership')
        groups=collections.defaultdict(list);allrows=[]
        for d in DATASETS:
            for member in ctx['cohort']['selected'][d]:
                q=member['query_id']
                for b in BUDGETS:
                    for m in METHODS:
                        r=rows.get([d,q,m,b]) if m==METHOD else ctx['source'].get([d,q,m,b])
                        if r is None or r['cluster_id']!=member['group']:raise ValueError('Unpaired/missing group')
                        groups[d,b,m].append(r);allrows.append(r)
        summaries=[];eff=[];contrasts=[];classes=[]
        for (d,b,m),rs in sorted(groups.items()):
            primary='macro_f1' if d=='SciFact' else 'f1';quality=metric(rs,primary)
            bs=[r['behavior'] for r in rs]
            summaries.append(dict(label=LABEL,dataset=d,budget=b,method=m,n_queries=len(rs),n_groups=len({r['cluster_id'] for r in rs}),
                metric=primary,quality=quality,EM=average(rs,'exact_match') if d!='SciFact' else None,
                accuracy=metric(rs,'accuracy') if d=='SciFact' else None,
                invalid_format=sum(r['prediction'] not in ctx['parent']['labels'] for r in rs) if d=='SciFact' else None,
                mean_prompt_tokens=average(rs,'prompt_tokens'),mean_generated_tokens=average(rs,'generated_tokens'),
                conceptual_reader_calls=len(rs)))
            eff.append(dict(label=LABEL,Dataset=d,Budget=b,Method=m,Quality=quality,
                **{'Mean Evidence Tokens':average(rs,'evidence_tokens'),'Budget Utilization':average(rs,'budget_utilization'),
                   'Selected Units':average(bs,'selected_units'),'Represented Parents':average(bs,'represented_parents')},
                mean_redundancy=average(bs,'mean_pairwise_redundancy')))
            if d=='SciFact':
                from src.week11.evaluation import classification_metrics
                classes.append(dict(label=LABEL,budget=b,method=m,**classification_metrics([r['gold'] for r in rs],[r['prediction'] for r in rs])))
        for d in DATASETS:
            for b in BUDGETS:
                for m in CONTROLS:
                    contrasts.append(dict(label=LABEL,dataset=d,budget=b,left=METHOD,right=m,
                        **paired(groups[d,b,METHOD],groups[d,b,m],'macro_f1' if d=='SciFact' else 'f1',10000,42)))
        for r,p in zip(contrasts,holm([r['raw_p'] for r in contrasts])):
            r.update(holm_p=p,exploratory_family_size=24)
        table,wins=comparison_table(summaries);S,E,C=pd.DataFrame(summaries),pd.DataFrame(eff),pd.DataFrame(contrasts)
        for name,frame in [('SUMMARY',S),('EFFICIENCY',E),('PAIRED_COMPARISONS',C),('FINAL_COMPARISON',table)]:
            save(OUT/'analysis'/(name+'.csv'),frame.to_csv(index=False),replace=True)
        save(OUT/'analysis/SCIFACT_CLASS_METRICS.json',dict(label=LABEL,rows=classes),replace=True)
        save(OUT/'analysis/POINT_ESTIMATE_WINS.json',dict(label=LABEL,scope='descriptive point-estimate wins only; exact ties credit all tied methods',wins=wins),replace=True)
        report='# MiniLM cross-encoder: matched-cohort comparison\n\n'+NOTICE+'\n\n'+table.to_markdown(index=False)+'\n\nHotpotQA/QASPER: answer F1; SciFact: three-class macro-F1. Bold indicates highest point estimate, not statistical superiority.\n\n## Efficiency\n\n'+E.to_markdown(index=False)+'\n\n## Descriptive point-estimate wins only\n\n'+'\n'.join(f'- {m}: {wins[m]} / 6' for m in METHODS)+'\n\nExact ties credit all tied methods, so counts can sum above six.\n\n## Paired comparisons\n\n'+C.to_markdown(index=False)+'\n\nExisting paired cluster-bootstrap 95% intervals and cluster label-swap raw p-values; one exploratory Holm family of all 24. QASPER paper groups and SciFact cited-document components remain unchanged. These intervals omit model-training uncertainty of historical learned controls. No per-query macro-F1, pooled task score, confirmation or new method selection.\n\n'
        report+='## Defensive interpretation\n\n'
        for letter,m in zip('ABCD',CONTROLS):
            cs=[r for r in contrasts if r['right']==m]
            higher=sum(r['point']>1e-12 for r in cs);ties=sum(abs(r['point'])<=1e-12 for r in cs)
            significant=[f"{r['dataset']} @{r['budget']}" for r in cs if r['holm_p']<.05 and r['ci_low']>0]
            report+=f'**Question {letter} — MiniLM versus {m}:** higher point quality in {higher}/6 cells, exact point ties in {ties}/6. Positive nominal-CI plus Holm-p<.05 cells: '+(', '.join(significant) or 'none')+'. Inspect all signs, magnitudes and uncertainty; failure to reject is not equivalence.\n\n'
        report+='**Question E — “Your method is only reranking”:** this control changes the relevance scorer while holding the candidate pool, source sentences, packing budget and reader fixed. Similar scores would be compatible with relevance ranking explaining substantial performance; they would not prove equivalence. MiniLM wins would require allocation methods to justify their added complexity. MiniLM losses would show this particular static relevance scorer does not reproduce those observed gains, but would not isolate state-dependent utility as their cause. This is a pretrained English MS-MARCO model applied to three different tasks, not every possible cross-encoder. Inspect exact contexts and task-specific errors rather than selecting a favorable cell.\n\n'
        report+='## Cost and diagnostic limits\n\nCPU CE scoring is query-time computation and remains conceptual cost on cache hits. Score-cache metadata retains original measured scoring seconds and pair/truncation counts. Reader prompt/generated-token costs remain counted on reused outputs. Evidence-token savings do not imply end-to-end latency improvements. Gold retention is computed only after selection; partial/unmapped annotations are explicit. No P_help, LOO labels or downstream quality enters ranking.\n\n'+CONCEPTUAL+'\n'+CLOSEOUT+'\n'
        save(OUT/'reports/FINAL_COMPARISON.md',report,replace=True)
        save(OUT/'analysis/COMPLETE.json',dict(label=LABEL,identity_hash=ctx['record']['identity_hash'],
            minilm_rows=300,all_method_rows=1500,contrasts=24,automatic_expansion=False),replace=True)
        return dict(final_table=table,summary=S,efficiency=E,contrasts=C,wins=wins,report=OUT/'reports/FINAL_COMPARISON.md')
