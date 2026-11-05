"""Optional 20/dataset BEA-only LOO after a promising pilot; never automatic."""
import json,collections
import numpy as np
import pandas as pd
from .protocol import load,freeze,output,Store,read,save,text,digest,LABEL,ROOT,exclusive
from .resources import load_val
from .runner import stores
from src.week12.algorithm import serialize

@exclusive
def run(c=None):
    c=c or load();parent,s,cohort,f=freeze(c);decision=read(output(c)/'PILOT_DECISION.json')
    if decision['identity_hash']!=f['identity_hash'] or decision['decision']!='PROMISING':raise ValueError('Optional attribution requires a compatible PROMISING pilot')
    subset=dict(cohort,selected={d:sorted(rs,key=lambda r:digest([42,d,r['query_id']]))[:20] for d,rs in cohort['selected'].items()})
    save(output(c)/'attribution/COHORT.json',subset)
    from transformers import AutoTokenizer
    from src.week12.reader import Reader,prompt_ids
    from src.week11.evaluation import evaluate
    tok=AutoTokenizer.from_pretrained(parent['models']['Qwen']['id'],revision=parent['models']['Qwen']['revision'],local_files_only=True)
    _,selections,results,generations=stores(c,f);out=Store(output(c)/'attribution/rows',f['identity']);reader=None;records=[];baselines=[]
    try:
        for e in load_val(parent,s,subset):
            for b in c['budgets']:
                key=[e['dataset'],e['query_id'],'BEA_QBGF',b];r=results.get(key);item=selections.get(key)
                if r is None or item is None:raise ValueError('Complete pilot required before attribution')
                units=item['selection']['units'];actions=[('LOO:'+str(i),[j for j in range(len(units)) if j!=i],i) for i in range(len(units))]
                for pid in sorted({u['passage_id'] for u in units}):actions.append(('GROUP_REMOVE:'+pid,[j for j,u in enumerate(units) if u['passage_id']!=pid],None))
                for action,keep,index in actions:
                    akey=key+[action];row=out.get(akey)
                    if row is None:
                        selection=serialize([units[j] for j in keep],tok);gkey=[e['dataset'],e['query_id'],digest(selection['evidence_ids']),'BASE_FP16'];answer=generations.get(gkey);physical=answer is None
                        if answer is None:
                            if reader is None:
                                import torch
                                if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<18*1024**3:raise RuntimeError('Wait for a free GPU; optional attribution is resumable')
                                reader=Reader(parent,'BASE_FP16')
                            answer=reader.generate(e,selection);generations.put(gkey,answer)
                        score=evaluate(answer['prediction'],e)['task_score'];delta=r['task_score']-score
                        row=dict(label=LABEL,dataset=e['dataset'],query_id=e['query_id'],budget=b,method='BEA_QBGF',action=action,
                                 importance=delta,neutral=abs(delta)<=1e-12,
                                 unit_tokens=len(tok.encode(units[index]['source_text'],add_special_tokens=False)) if index is not None else None,
                                 source_hash=r['source_hash'],original_evidence_hash=r['evidence_hash'],physical_generation_performed=physical,
                                 score=score,answer=answer['prediction'],kept_indices=keep)
                        out.put(akey,row)
                    records.append(row)
                # Existing baseline LOO may be imported, but is NEVER regenerated here.
                for m in ['FIXED_SENTENCE','QBGF_FULL']:
                    full=results.get([e['dataset'],e['query_id'],m,b]);sel=selections.get([e['dataset'],e['query_id'],m,b])
                    directory=ROOT/'results/week12/attribution/v1/perturbations';ident=directory/'identity.json'
                    if full is None or sel is None or not ident.exists():continue
                    oldid=read(ident)
                    original_study=read(ROOT/'results/week12/attribution/ATTRIBUTION_QUERY_FREEZE.json')['identity']
                    if oldid!=original_study or oldid.get('parent')!=f['identity']['parent_identity']:continue
                    for i,u in enumerate(sel['selection']['units']):
                        oldkey=[e['dataset'],e['query_id'],m,b,'LOO:'+str(i)];p=directory/(digest(oldkey)+'.json')
                        if not p.exists():continue
                        v=read(p);old=v['row']
                        if v['key']!=oldkey or v['identity']!=digest(oldid) or v['checksum']!=digest(old):raise ValueError('Legacy LOO checksum mismatch')
                        expected_keep=[j for j in range(len(sel['selection']['units'])) if j!=i]
                        if old.get('original_evidence_hash')!=full['evidence_hash'] or old.get('source_hash')!=full['source_hash'] or old.get('unit_indices')!=expected_keep:continue
                        remaining=serialize([u for j,u in enumerate(sel['selection']['units']) if j!=i],tok)
                        if old.get('prompt_hash')!=digest(prompt_ids(tok,e['question'],e['task_type'],remaining['evidence_ids'],parent['labels'])):continue
                        delta=full['task_score']-old['task_score']
                        baselines.append(dict(label=LABEL,dataset=e['dataset'],query_id=e['query_id'],budget=b,method=m,action='LOO:'+str(i),importance=delta,
                                              neutral=abs(delta)<=1e-12,unit_tokens=len(tok.encode(u['source_text'],add_special_tokens=False)),legacy_source=str(p.relative_to(ROOT))))
            print('Optional BEA attribution saved',e['dataset'],e['query_id'],flush=True)
    finally:
        if reader is not None:reader.close()
    allrows=records+baselines;text(output(c)/'attribution/per_unit_and_group.csv',pd.DataFrame(allrows).to_csv(index=False))
    groups=collections.defaultdict(list)
    for r in allrows:
        if r['action'].startswith('LOO:'):groups[r['dataset'],r['budget'],r['method']].append(r)
    summary=[]
    for (d,b,m),rs in sorted(groups.items()):
        total=sum(r['unit_tokens'] for r in rs)
        summary.append(dict(dataset=d,budget=b,method=m,n_queries=len({r['query_id'] for r in rs}),n_units=len(rs),
                            neutral_unit_fraction=float(np.mean([r['neutral'] for r in rs])),
                            neutral_token_fraction=sum(r['unit_tokens'] for r in rs if r['neutral'])/total if total else None,
                            token_definition='Exact source-text Qwen tokens per exposed unit; excludes framing; not additive serialized budget',
                            coverage='Legacy baselines only where exact LOO cache exists; missing rows are not regenerated'))
    text(output(c)/'attribution/summary.csv',pd.DataFrame(summary).to_csv(index=False))
    text(output(c)/'attribution/REPORT.md','# Optional BEA attribution — POST_W12_EXPLORATORY\n\n'+pd.DataFrame(summary).to_markdown(index=False)+'\n\nNeutral means unchanged task score under this removal, not useless evidence or internal inattention. Parent-group removals in per_unit_and_group.csv probe redundancy across units but do not prove causality. Baseline caches may cover only subsets: denominators are explicit and unmatched fractions are descriptive. No historical LOO results were modified.\n')
    return summary
