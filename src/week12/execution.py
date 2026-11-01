"""Execution subsets of existing preregistration; no scientific/split config edits."""
CORE_C=('NONE','RAW_SKIP','FIXED_SENTENCE','C2','H3','QBGF_FULL')
CORE_D=('QBGF_FULL','QBGF_NO_QUERY','QBGF_SENTENCE_ONLY','QBGF_PASSAGE_ONLY')

def methods(phase,core_final=True):
    from .runner import MAIN,ABLATIONS
    if phase=='C':return list(CORE_C if core_final else MAIN)
    if phase=='D':return list(CORE_D if core_final else ABLATIONS)
    return ['LLMLINGUA2'] if phase=='E' else ['RAW_SKIP','QBGF_FULL'] if phase=='G' else MAIN

def budgets(cfg,phase,method,core_final=True,real_smoke=False):
    if method in ('STRONG_SFR_RAG','BM25_QWEN_REFERENCE'):return ['NATURAL']
    if real_smoke:return cfg['budgets'][:1]
    bs=cfg['budgets']
    if core_final:
        if bs!=[256,1024]:raise ValueError('CORE_FINAL budgets must be preregistered 256/1024')
        if phase=='C' and method=='C2':return [256]
    return bs

def cost_plan(cfg,counts):
    n=sum(counts.values());rows=[]
    for phase in ['C','D']:
        for core in [True,False]:
            per=sum(len(budgets(cfg,phase,m,core)) for m in methods(phase,core))
            # C: NONE's duplicate budget row uses one cached baseline call.
            # D: QBGF already exists from C, leaving only new ablation calls.
            calls=per-1 if phase=='C' else per-2
            rows.append(dict(phase=phase,CORE_FINAL=core,queries=n,logical_rows=n*per,additional_call_upper_bound=n*calls,methods=methods(phase,core),note='Warm generation reuse of compatible C QBGF in D; other identical/empty evidence may reduce calls further. No empirical full-run time estimate.'))
    return rows
