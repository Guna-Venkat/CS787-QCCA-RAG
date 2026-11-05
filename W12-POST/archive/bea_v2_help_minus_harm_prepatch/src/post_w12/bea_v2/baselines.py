"""Read-only reuse; missing controls restored with canonical code into V2 only."""
from .protocol import *

def baseline(c,v1,f,d,q,method,b):
    original='BEA_QBGF' if method=='BEA_QBGF_V1' else method;key=[d,q,original,b]
    path=old(c)/'cache/results'/(digest(key)+'.json')
    if path.exists():return legacy(c,'cache/results',key,v1['identity_hash'])
    st=Store(root(c)/'pilot/restored_controls',f['identity']);r=st.get([d,q,method,b])
    if r is None:raise ValueError('Missing control; run restore_controls with unchanged historical implementation')
    return r

@exclusive
def restore_controls(c=None):
    c=c or load();parent,v1,s,cohort,f=freeze(c);missing=[]
    st=Store(root(c)/'pilot/restored_controls',f['identity'])
    for d,members in cohort['selected'].items():
        for member in members:
            for b in c['budgets']:
                for m in c['methods'][:-1]:
                    original='BEA_QBGF' if m=='BEA_QBGF_V1' else m
                    path=old(c)/'cache/results'/(digest([d,member['query_id'],original,b])+'.json')
                    if path.exists():legacy(c,'cache/results',[d,member['query_id'],original,b],v1['identity_hash'])
                    elif st.get([d,member['query_id'],m,b]) is None:missing.append((d,member,m,b))
    if not missing:return dict(reused_controls=900,restored=0)
    # This path is manual-only; never regenerate originals into the V1 namespace.
    from .data import problem
    from .stages import tokenizer
    from src.post_w12.bea_qbgf.resources import load_val
    from src.post_w12.bea_qbgf.allocator import allocate as v1_allocate,behavior
    from src.week12.algorithm import choose
    from src.week12.reader import Reader
    from src.week11.evaluation import evaluate
    import torch
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<18*1024**3:raise RuntimeError('Missing frozen controls need a free reader GPU; no substitution or V1 interruption')
    xs={(e['dataset'],e['query_id']):e for e in load_val(parent,s,cohort)};tok=tokenizer(parent);reader=Reader(parent,'BASE_FP16')
    try:
        for d,member,m,b in missing:
            q=member['query_id'];e=xs[d,q];p=problem(c,v1,d,q)
            sel=(v1_allocate(p['pool'],p['sentences'],p['scores'],tok,b,v1['identity']['parameters']) if m=='BEA_QBGF_V1' else
                 choose(p['pool'],p['sentences'],p['props'],p['passage_scores'],p['sentence_scores'],tok,b,m))
            answer=reader.generate(e,sel)
            if answer['reader']!='BASE_FP16' or answer['model_hash']!=digest([parent['models']['Qwen'],'BASE_FP16',None]):raise ValueError('Wrong historical reader identity')
            row=dict(dataset=d,query_id=q,cluster_id=member['group'],method=m,budget=b,**evaluate(answer['prediction'],e),**answer,
                     gold=e['evaluation']['gold_label'] if d=='SciFact' else e['evaluation']['gold_answer'],correct=int(answer['prediction']==e['evaluation']['gold_label']) if d=='SciFact' else None,
                     evidence_tokens=sel['evidence_tokens'],unused_budget=b-sel['evidence_tokens'],budget_utilization=sel['evidence_tokens']/b,
                     behavior=behavior(sel,p['sentences'],p['scores']),stop_reason=sel.get('stop_reason','FROZEN_BASELINE'),evidence_hash=digest([sel['evidence_ids'],sel['text']]),source_hash=p['source_hash'],
                     physical_generation_performed=True,conceptual_lm_calls=1,restoration='Canonical historical implementation; isolated V2 output only')
            st.put([d,q,m,b],row)
    finally:reader.close()
    return dict(restored=len(missing))
