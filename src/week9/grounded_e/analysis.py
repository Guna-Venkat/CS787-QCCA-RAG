"""Paired paper-cluster evaluation and grounding/cost diagnostics, never tuning."""
import json
import numpy as np
import pandas as pd
from ..evaluation import paired_paper_bootstrap
from ..caching import atomic_json
from .prompts import POLICIES

PAIRS=(('E0','D0'),('E0','D1'),('E1-Q','D0'),('E1-QD','D0'),('E1-QD','E1-Q'),('E1-QD','D3'),('E2','D0'),('E2','E1-QD'))


def quality(rows):
    result=[]
    for name in POLICIES:
        group=[r for r in rows if r['policy']==name]
        if not group:continue
        f=pd.DataFrame([r['evidence'] for r in group])
        supported=sum(r['evidence']['num_supported_items'] for r in group)
        proposed=sum(r['evidence']['num_supported_items']+sum(x.get('reason')=='unsupported_span_or_source' or x.get('reason')=='item_not_object' for x in r['evidence']['rejected_items']) for r in group)
        row=dict(policy=name,queries=len(group),exact_span_support_rate=supported/proposed if proposed else None,
                 accepted_exact_span_support_rate=1.0 if f.num_accepted_items.sum() else None,
                 mean_evidence_tokens=float(f.evidence_tokens.mean()),median_evidence_tokens=float(f.evidence_tokens.median()),p90_evidence_tokens=float(f.evidence_tokens.quantile(.9)),
                 compression_ratio=float(np.mean([r['compression_ratio'] for r in group])))
        for key in ('empty_extraction','invalid_format','fail_closed','reached_extraction_token_limit','zero_exact_support','num_supported_items','num_accepted_items','num_rejected_items','fact_tokens','support_tokens','num_accepted_facts','num_rejected_facts'):
            row[key]=float(f[key].mean())
        result.append(row)
    return result


def warnings(quality_rows):
    messages=[]
    for r in quality_rows:
        for key,limit,label in [('invalid_format',.2,'malformed'),('empty_extraction',.2,'empty'),('fail_closed',.5,'fail-closed'),('reached_extraction_token_limit',.2,'generation-cap hits'),('compression_ratio',.25,'insufficient compression')]:
            if r[key]>limit:messages.append(f"{r['policy']}: {label} rate/value {r[key]:.3f} exceeds preregistered warning {limit}.")
        if r['exact_span_support_rate'] is not None and r['exact_span_support_rate']<.8:
            messages.append(f"{r['policy']}: proposed exact-span support below .8. Rejected evidence must never reach the generator.")
    return messages


def summarize(rows,baselines,directory,*,complete,seed=42):
    directory.mkdir(parents=True,exist_ok=True)
    q=quality(rows);pd.DataFrame(q).to_csv(directory/'evidence_quality.csv',index=False)
    policies={p:[r for r in rows+baselines if r['policy']==p] for p in ('D0','D1','D2','D3')+POLICIES}
    summaries=[]
    for p,group in policies.items():
        if not group:continue
        row=dict(policy=p,queries=len(group))
        for key in ('f1','exact_match','delta_f1','help','hurt','answer_changed','final_prompt_tokens','total_query_prompt_tokens','total_query_generated_tokens','num_lm_calls','retrieval_query_time','selection_latency','extraction_latency','revision_latency','measured_end_to_end_latency'):
            values=[r.get(key) for r in group if r.get(key) is not None]
            row[key]=float(np.mean(values)) if values else None
        denominator=(row.get('help') or 0)+(row.get('hurt') or 0)
        row['descriptive_help_over_help_plus_hurt']=row['help']/denominator if denominator else None
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(directory/'answer_and_cost_summary.csv',index=False)
    pairs={}
    if complete:
        for a,b in PAIRS:pairs[a+' - '+b]=paired_paper_bootstrap(policies[a],policies[b],n=10000,seed=seed)
        atomic_json(directory/'paired_paper_bootstrap.json',pairs)
    notes=[]
    if complete:
        positive=lambda k:pairs[k]['ci95'][0]>0
        if positive('E0 - D0') and positive('E0 - D1'):notes.append('E0 improves over both D0 and D1 with positive paired intervals: compact copied spans help under this controlled protocol.')
        if positive('E1-QD - E1-Q'):notes.append('Draft conditioning improves over query-only structured extraction with a positive paired interval.')
        if positive('E2 - E1-QD'):notes.append('Blueprint organization improves over E1-QD with a positive paired interval.')
        if not notes:notes.append('No requested superiority conclusion is supported by positive paired intervals. Inconclusive intervals do not establish equivalence or failure.')
        if all(pairs[k+' - D0']['ci95'][1]<0 for k in ('E0','E1-Q','E1-QD','E2')):
            notes.append('All E interventions underperform D0 with negative paired intervals. This weakens the representation-only hypothesis for this implementation, subject to usable-evidence and failure-rate diagnostics.')
    warn=warnings(q)
    (directory/'REPORT.md').write_text('# W9-E compact grounded evidence\n\n'+('Complete FIT diagnostic; not held-out confirmation.\n\n' if complete else 'Small FIT preflight only; no method selection or superiority inference.\n\n')+'Primary policy stays K=0. Exact-span support is mechanical, not semantic entailment. Facts use the conservative copied-subspan variant. No outcome-driven prompt tuning. All extraction calls are charged; cached initial generation is counted, with its latency unknown. End-to-end latency remains unmeasured.\n\n'+'\n'.join(notes+warn)+'\n')
    return dict(quality=q,warnings=warn,paired=pairs,interpretation=notes)
