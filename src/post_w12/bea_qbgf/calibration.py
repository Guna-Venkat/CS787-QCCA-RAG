"""Pre-pilot allocation-only threshold calibration. No reader or scorer is imported."""
import copy,subprocess,time
import numpy as np
import pandas as pd
from .protocol import ROOT,load,preregister,output,Store,save,text,read,digest,exclusive
from .allocator import allocate,behavior

def choose_threshold(rows,c):
    """Global deterministic choice on allocation signals only, never answer outcomes."""
    base={ (r['dataset'],r['query_id'],r['budget']):r['coverage'] for r in rows if r['tau']==min(c['calibration']['tau_stop_grid']) }
    summaries=[];cfg=c['calibration']
    for tau in cfg['tau_stop_grid']:
        xs=[r for r in rows if r['tau']==tau]
        retention=[r['coverage']/base[r['dataset'],r['query_id'],r['budget']] if base[r['dataset'],r['query_id'],r['budget']]>0 else 1. for r in xs]
        row=dict(tau=tau,n=len(xs),threshold_stop_fraction=float(np.mean([r['stop_reason']=='LOW_MARGINAL_VALUE' for r in xs])),
                 budget_stop_fraction=float(np.mean([r['stop_reason']=='BUDGET_EXHAUSTED' for r in xs])),
                 mean_utilization=float(np.mean([r['utilization'] for r in xs])),
                 near_full_fraction=float(np.mean([r['utilization']>=.95 for r in xs])),
                 empty_fraction=float(np.mean([r['tokens']==0 for r in xs])),coverage_retention=float(np.mean(retention)))
        row['eligible']=(row['threshold_stop_fraction']>=cfg['minimum_threshold_stop_fraction'] and
                         row['near_full_fraction']<=cfg['maximum_near_full_fraction'] and
                         row['empty_fraction']<=cfg['maximum_empty_fraction'] and
                         row['coverage_retention']>=cfg['minimum_coverage_retention'])
        summaries.append(row)
    eligible=[r['tau'] for r in summaries if r['eligible']]
    return min(eligible) if eligible else None,summaries

def tests(c):
    _,_,_,draft=preregister(c);path=output(c)/'calibration/TESTS.json'
    if path.exists():
        receipt=read(path)
        if receipt['protocol_hash']==draft['identity_hash'] and receipt['passed']:return receipt
        raise ValueError('Incompatible test receipt; choose a fresh version')
    cmd=[str(ROOT/'sara_env/bin/python'),'-m','pytest','W12-POST/tests','tests/week12','-q']
    excluded=['tests/week12/test_c_extension.py::test_safe_notebook_defaults_and_freeze_gate',
              'tests/week12/test_confirmation_m.py::MetadataTests::test_notebook_defaults',
              'tests/week12/test_framework.py::ProtocolTests::test_notebook_defaults']
    for node in excluded:cmd.extend(['--deselect',node])
    p=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    text(output(c)/'calibration/tests.log',p.stdout)
    if p.returncode:raise RuntimeError('Tests failed; see calibration/tests.log. No smoke or pilot started.')
    return save(path,dict(protocol_hash=draft['identity_hash'],passed=True,command=cmd,output=p.stdout,excluded_tests=excluded,
                          exclusion_reason='User approved excluding three stale historical notebook-default assertions on 2026-11-04; original files unchanged'))

@exclusive
def run(c=None):
    c=c or load();tests(c) # Always tests before allocation, including direct CLI calls.
    parent,s,cohort,draft=preregister(c);directory=output(c)/'calibration'
    done=directory/'CALIBRATION.json'
    if done.exists():
        receipt=read(done)
        if receipt['protocol_hash']!=draft['identity_hash']:raise ValueError('Calibration identity changed')
        return receipt
    small=copy.deepcopy(cohort)
    small['selected']={d:sorted(v,key=lambda x:x['query_id'])[:c['calibration']['queries_per_dataset']] for d,v in cohort['selected'].items()}
    save(directory/'COHORT.json',small)
    from .resources import load_val,IsolatedLab
    from .runner import build_problem
    from transformers import AutoTokenizer
    from src.week12.algorithm import validate_spans
    tok=AutoTokenizer.from_pretrained(parent['models']['Qwen']['id'],revision=parent['models']['Qwen']['revision'],local_files_only=True)
    problems=Store(output(c)/'cache/problems',draft['identity']);allocations=Store(directory/'allocations',draft['identity'])
    rows=[];steps=[];lab=None;start=time.time()
    try:
        for number,original in enumerate(load_val(parent,s,small),1):
            # Strip all evaluation payloads before any retrieval or allocation call.
            e={k:v for k,v in original.items() if k not in ('evaluation','gold','gold_answer','gold_label')}
            d,q=e['dataset'],e['query_id'];problem=problems.get([d,q])
            if problem is None:
                if lab is None:lab=IsolatedLab(parent,tok,c)
                problem=build_problem(lab,e);problems.put([d,q],problem)
            if problem['source_hash']!=digest(e['inference']):raise ValueError('Calibration source identity changed')
            for budget in c['budgets']:
                for tau in c['calibration']['tau_stop_grid']:
                    key=[d,q,budget,tau];item=allocations.get(key)
                    if item is None:
                        params=dict(c['parameters'],tau_stop=tau,tau_expand=max(tau,c['parameters']['tau_expand']))
                        sel=allocate(problem['pool'],problem['sentences'],problem['scores'],tok,budget,params)
                        validate_spans(sel['units'],e['inference']['documents'])
                        item=allocations.put(key,dict(selection=sel,behavior=behavior(sel,problem['sentences'],problem['scores'])))
                    sel=item['selection'];bh=item['behavior']
                    rows.append(dict(dataset=d,query_id=q,budget=budget,tau=tau,tokens=sel['evidence_tokens'],utilization=sel['evidence_tokens']/budget,
                                     coverage=bh['aspect_coverage'],stop_reason=sel['stop_reason'],final_marginal_score=sel['final_marginal_score'],
                                     **{k:v for k,v in bh.items() if k.startswith('full_')}))
                    for step in sel['step_scores']:
                        values=step['marginal_scores']
                        steps.append(dict(dataset=d,query_id=q,budget=budget,tau=tau,step=step['step'],candidate_count=step['candidate_count'],
                                          p10=float(np.quantile(values,.1)) if values else None,median=step['median'],
                                          p90=float(np.quantile(values,.9)) if values else None,maximum=step['maximum']))
            print(f'Allocation-only calibration {number}/9 queries; no answer generation; elapsed={time.time()-start:.0f}s',flush=True)
    finally:
        if lab is not None:lab.close()
    selected,summaries=choose_threshold(rows,c)
    for name,data in [('allocations',rows),('step_marginal_distribution',steps),('threshold_summary',summaries)]:
        text(directory/(name+'.csv'),pd.DataFrame(data).to_csv(index=False))
    df=pd.DataFrame(rows)
    counts=[k for k in df if k.startswith('full_')]
    full=df.groupby(['dataset','budget','tau'])[counts].sum().reset_index()
    text(directory/'full_diagnostics.csv',full.to_csv(index=False))
    params=dict(c['parameters'])
    if selected is not None:params.update(tau_stop=selected,tau_expand=max(selected,params['tau_expand']))
    receipt=dict(protocol_hash=draft['identity_hash'],status='READY' if selected is not None else 'NEEDS_REVIEW',
                 parameters=params,selected_threshold=selected,queries=sum(len(x) for x in small['selected'].values()),
                 answer_generations=0,quality_metrics_used=False,cohort_hash=digest(small),summary=summaries)
    text(directory/'ALLOCATION_ONLY_REPORT.md','# Allocation-only pre-pilot calibration\n\n'
         'Nine frozen VAL pilot queries (three per dataset); both budgets. No Qwen generation or answer-quality scoring. '
         'Native VAL adapters parse their existing records, but evaluation payloads are discarded before allocation. '
         'The threshold rule was registered before observing these traces. FULL eligibility remains a two-positive-sentence design assumption.\n\n'
         +pd.DataFrame(summaries).to_markdown(index=False)+'\n\nStatus: **'+receipt['status']+'**; selected global threshold: '+str(selected)+'.\n\n'
         'FULL counts are candidate-step events, not distinct sources. `full_eligible_query` counts queries with any eligible candidate.\n\n'+full.to_markdown(index=False)+'\n')
    save(done,receipt)
    return receipt
