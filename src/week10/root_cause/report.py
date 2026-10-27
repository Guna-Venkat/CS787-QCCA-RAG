"""Reports measured development results only; separate manual review/freeze."""
import csv
import json
from collections import defaultdict
import numpy as np
from .storage import ROOT,Store,atomic,digest
from .protocol import freeze
from .analysis import oracle,paired

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)

def report(cfg):
    base=ROOT/cfg['output_root']; summary=[];groups={};boot=[];oracles={}
    for path in sorted((base/'runs').glob('*/identity.json')):
        if path.parent.name.startswith(('R0-','D-')):continue
        if not (path.parent/'complete.json').exists():continue
        store=Store(path.parent,json.loads(path.read_text()));rows=store.rows()
        if not rows or 'f1' not in rows[0]:continue
        by=defaultdict(list)
        for r in rows:by[r['action']].append(r)
        baseline={r['question_id']:r for r in by.get('no_evidence',[])}
        for a,rs in by.items():
            key=path.parent.name+'|'+a;groups[key]=rs
            summary.append(dict(run=path.parent.name,action=a,n_queries=len(rs),f1=float(np.mean([r['f1'] for r in rs])),
                exact_match=float(np.mean([r['exact_match'] for r in rs])),
                mean_prompt_tokens=float(np.mean([r.get('cumulative_prompt_tokens',r['generation']['prompt_tokens']) for r in rs])),
                HELP=float(np.mean([r['f1']-baseline[r['question_id']]['f1']>.01 for r in rs])) if baseline else None,
                HURT=float(np.mean([r['f1']-baseline[r['question_id']]['f1']<-.01 for r in rs])) if baseline else None,
                delta_vs_no_evidence=float(np.mean([r['f1']-baseline[r['question_id']]['f1'] for r in rs])) if baseline else None,
                answer_changed=float(np.mean([r['prediction']!=baseline[r['question_id']]['prediction'] for r in rs])) if baseline else None,
                lm_calls=1,mean_generated_tokens=float(np.mean([r['generation'].get('generated_tokens',0) for r in rs])),
                mean_generation_seconds=float(np.mean([r['generation'].get('latency',0) for r in rs])),
                measured_end_to_end_seconds=None,answer_nll=None,grounding='unmeasured'))
        if not path.parent.name.startswith('D-'):oracles[path.parent.name]=oracle(rows)
    for a,ar in groups.items():
        for b,br in groups.items():
            if a<=b:continue
            # Paired bootstrap requires identical query AND paper identities.
            if {(r['question_id'],r['paper_id']) for r in ar}!={(r['question_id'],r['paper_id']) for r in br}:continue
            aa=[dict(question_id=r['question_id'],paper_id=r['paper_id'],f1=r['f1']) for r in ar]
            bb=[dict(question_id=r['question_id'],paper_id=r['paper_id'],f1=r['f1']) for r in br]
            result=paired(aa,bb)
            boot.append(dict(a=a,b=b,point=result['point_estimate'],ci_low=result['ci95'][0],ci_high=result['ci95'][1]))
    interactions=[]
    for method in ('MiniLM','SFR'):
        for k in cfg.get('k_values',[2,4,6,8]):
            names=[f'R-{m}-{i}|{m}/{i}/k{k}' for i in ('full','sara') for m in (method,'BM25')]
            if not all(name in groups for name in names):continue
            indexed=[{r['question_id']:r for r in groups[name]} for name in names]
            if any(set(x)!=set(indexed[0]) for x in indexed):raise ValueError('Interaction query membership mismatch')
            contrasts=[]
            for left,right in ((0,1),(2,3)):
                contrasts.append([dict(question_id=q,paper_id=indexed[left][q]['paper_id'],
                    f1=indexed[left][q]['f1']-indexed[right][q]['f1']) for q in sorted(indexed[0])])
            result=paired(*contrasts)
            interactions.append(dict(retriever=method,k=k,contrast='(method-BM25)FULL - (method-BM25)SARA',
                point=result['point_estimate'],ci_low=result['ci95'][0],ci_high=result['ci95'][1]))
    write_csv(base/'retrieval_interface_interactions.csv',interactions)
    write_csv(base/'summary.csv',summary);write_csv(base/'bootstrap.csv',boot);atomic(base/'oracle_diagnostics.json',oracles)
    retrieval=[]
    for path in sorted((base/'runs').glob('R0-*/identity.json')):
        store=Store(path.parent,json.loads(path.read_text())); rows=store.rows()
        for k in (1,2,4,6,8,10):
            ms=[m for r in rows for m in r['retrieval_metrics'] if m['k']==k]
            if not ms:continue
            metrics={}
            for name in ('any_recall','all_recall','coverage','mrr','retrieved_tokens','duplicate_rate','candidate_diversity','unmapped_gold_spans'):
                vs=[m[name] for m in ms if m[name] is not None];metrics[name]=float(np.mean(vs)) if vs else None
            retrieval.append(dict(retriever=path.parent.name[3:],k=k,n_queries=len(ms),**metrics))
    write_csv(base/'retrieval_summary.csv',retrieval)
    required=[base/'runs'/f'R-{m}-{i}'/'complete.json' for m in ('BM25','MiniLM','SFR') for i in ('full','sara')]
    atomic(base/'decision_R.json',dict(GO_NEXT=False,ready_for_review=all(p.exists() for p in required),
        reason='Review matched recall, downstream deltas, confidence intervals and resource costs; no automatic bottleneck claim',
        preregistered_material_f1=.01,preregistered_material_recall=.03,required_repeatability='at least two k values with positive paper CI',
        execution_freeze_hash=digest(freeze(cfg))))
    print('Development reports saved. Manual evidence-based phase review required.')

def review(cfg,phase,retriever,reason):
    if phase not in ('R','C','G'):raise ValueError('Only completed upstream phases can be frozen')
    base=ROOT/cfg['output_root'];report(cfg)
    if phase=='R':
        decision=json.loads((base/'decision_R.json').read_text())
        if not decision['ready_for_review']:raise RuntimeError('All matched Phase R arms must finish')
    else:
        needed=['C-Mistral'] if phase=='C' else ['G-'+m for m in cfg['reader_panel']]
        if not all((base/'runs'/n/'complete.json').exists() for n in needed):raise RuntimeError('Phase incomplete')
    if phase!='R':
        from .runner import phase_gate
        previous=phase_gate(cfg,phase)
        if previous['retriever']!=retriever:raise ValueError('Cannot change the frozen retriever downstream')
    obj=dict(GO_NEXT=True,phase=phase,retriever=retriever,reason=reason,execution_freeze_hash=digest(freeze(cfg)))
    path=base/f'freeze_{phase}.json'
    if path.exists() and json.loads(path.read_text())!=obj:raise ValueError('Phase already frozen differently')
    if not path.exists():atomic(path,obj)
