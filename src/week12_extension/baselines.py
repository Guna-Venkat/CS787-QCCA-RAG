"""Budget-matched passage controls. Never edits original Part C or split membership."""
import json, copy, time
from pathlib import Path
from src.week12.protocol import ROOT, root, identity, digest, sha, Store, atomic, immutable, cohort
METHODS={'BM25_PLAIN_RAG':'BM25','SFR_PLAIN_RAG':'SFR'}

def extension_identity(cfg, synthetic=False):
    return dict(parent=identity(cfg),extension_hash=digest({p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}),synthetic=synthetic,protocol='C-EXT-v1',methods=METHODS,budgets=[256,1024],allocation='RAW_SKIP whole passages; same serialization and source order as C',reader='BASE_FP16')

def folder(cfg,synthetic=False):
    return root(cfg)/(('smoke/c_extension/'+extension_identity(cfg,True)['extension_hash'][:12]) if synthetic else 'c_extension/v1')

def prepare(cfg,synthetic=False):
    from src.week12.runner import examples
    from src.week12.laboratory import Lab
    from src.week12.algorithm import choose,validate_spans,serialize
    if synthetic:
        from src.week11.budgets import FixtureTokenizer
        tok=FixtureTokenizer();legacy={'Qwen':tok}
    else:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(cfg['models']['Qwen']['id'],revision=cfg['models']['Qwen']['revision'],local_files_only=True);legacy={'Qwen':tok}
    assert cfg['budgets']==[256,1024]
    st=Store(folder(cfg,synthetic)/'prepared',extension_identity(cfg,synthetic));lab=Lab(cfg,tok,synthetic)
    xs=examples(cfg,synthetic,3 if synthetic else None)
    try:
        for i,e in enumerate(xs):
            for method,retriever in METHODS.items():
                if all(st.get([e['dataset'],e['query_id'],method,b]) is not None for b in cfg['budgets']):continue
                pool,sent,props,ps,ss,lineage=lab.prepare(e,retriever)
                for b in cfg['budgets']:
                    key=[e['dataset'],e['query_id'],method,b]
                    if st.get(key) is not None:continue
                    s=choose(pool,sent,props,ps,ss,tok,b,'RAW_SKIP',legacy)
                    validate_spans(s['units'],e['inference']['documents'])
                    s.update(method=method,lineage=lineage,raw_reference_tokens=serialize(pool,tok)['evidence_tokens'])
                    assert s['evidence_tokens']<=b
                    st.put(key,dict(dataset=e['dataset'],query_id=e['query_id'],selection=s))
            if (i+1)%10==0:print(f'C-EXT prepared {i+1}/{len(xs)} queries',flush=True)
    finally:lab.close()
    return st.rows()

def generate(cfg,synthetic=False):
    from src.week12.runner import examples,evaluate_result
    from src.week12.reader import Reader,parse_label
    from src.week11.evaluation import evaluate
    from src.week11.readers import FixtureReader
    ident=extension_identity(cfg,synthetic);f=folder(cfg,synthetic)
    st=Store(f/'rows',ident);prepared=Store(f/'prepared',ident).rows()
    # Exact existing shared generation identity: reuse original C answers read-only.
    oldpath=root(cfg)/'generation';old=None
    if not synthetic and (oldpath/'identity.json').exists():
        oldident=json.loads((oldpath/'identity.json').read_text())
        if oldident!=identity(cfg,dict(synthetic=False,real_smoke=False,adapter_hash=None)):raise ValueError('Original C generation identity mismatch')
        old=Store(oldpath,oldident)
    gen=Store(f/'generation',ident)
    xs={(e['dataset'],e['query_id']):e for e in examples(cfg,synthetic,3 if synthetic else None)}
    reader=None
    try:
        for i,item in enumerate(prepared):
            s=item['selection'];e=xs[item['dataset'],item['query_id']];key=[e['dataset'],e['query_id'],s['method'],s['budget'],'BASE_FP16']
            if st.get(key) is not None:continue
            def call(sel):
                nonlocal reader
                k=[e['dataset'],e['query_id'],digest(sel['evidence_ids']),'BASE_FP16']
                result=(old.get(k) if old else None) or gen.get(k)
                if result is not None:return result,False
                if reader is None:reader=FixtureReader(dict(readers=['Qwen']),'Qwen') if synthetic else Reader(cfg)
                if synthetic:
                    result=reader.generate(e['question'],sel['text'],e['task_type']);result.update(reader='BASE_FP16',raw_prediction=result['prediction'],model_hash='SYNTHETIC',prompt_hash=digest([e['question'],sel['evidence_ids']]))
                    if e['task_type']=='verification':result['prediction']=parse_label(result['prediction'],cfg['labels'])
                else:result=reader.generate(e,sel)
                gen.put(k,result);return result,True
            base,_=call(dict(s,text='',evidence_ids=[],method='NONE',evidence_tokens=0))
            answer,physical=call(s)
            row=evaluate_result(cfg,e,s,answer,dict(score=evaluate(base['prediction'],e)['task_score'],prediction=base['prediction']))
            row.update(synthetic=synthetic,real_smoke=False,physical_lm_calls=int(physical),comparison_scope='budget-matched BASE_FP16; whole-passage skip allocator',extension_identity=digest(ident))
            st.put(key,row)
            if (i+1)%10==0:print(f'C-EXT saved {i+1}/{len(prepared)} rows',flush=True)
    finally:
        if reader:reader.close()
    rows=st.rows();atomic(f/'status.json',dict(complete=len(rows)==len(prepared)==len(xs)*4,rows=len(rows),expected=len(xs)*4,synthetic=synthetic))
    return rows

def merged_analysis(cfg,synthetic=False):
    """Supplement H/I without modifying historical analysis, registry or final policy."""
    import pandas as pd
    from src.week12.analysis import read_rows
    from src.week12.statistics import metric,paired,holm
    rows=read_rows(cfg,synthetic);f=folder(cfg,synthetic);p=f/'rows/identity.json'
    ext=Store(p.parent,json.loads(p.read_text())).rows() if p.exists() else []
    if ext and json.loads(p.read_text())!=extension_identity(cfg,synthetic):raise ValueError('C-EXT identity mismatch')
    expected={d:{e['query_id'] for e in es} for d,es in cohort(cfg).items()} if not synthetic else None
    for r in ext:
        if not synthetic and (r['query_id'] not in expected[r['dataset']] or r['split']!='VALIDATION'):raise ValueError('Non-VAL extension row')
    parent_sources={}
    for r in rows:
        sig=(r['paper_id'],r['cluster_id'],r['source_hash'])
        key=(r['dataset'],r['query_id'])
        if key in parent_sources and parent_sources[key]!=sig:raise ValueError('C source identity disagreement')
        parent_sources[key]=sig
    for r in ext:
        key=(r['dataset'],r['query_id'])
        if key in parent_sources and parent_sources[key]!=(r['paper_id'],r['cluster_id'],r['source_hash']):raise ValueError('Extension source identity differs from C')
    rows+=ext;groups={} 

    for r in rows:
        k=(r['dataset'],r['method'],str(r['budget']),r['reader']);groups.setdefault(k,[]).append(r)
    summaries=[];comparisons=[]
    for (d,m,b,reader),g in sorted(groups.items()):
        if len({r['query_id'] for r in g})!=len(g):raise ValueError('Duplicate merged results')
        summaries.append(dict(dataset=d,method=m,budget=b,reader=reader,n_queries=len(g),quality=metric(g,cfg['statistics']['primary_metric'][d]),prompt_tokens=sum(r['prompt_tokens'] for r in g)/len(g),evidence_tokens=sum(r['evidence_tokens'] for r in g)/len(g),scope='matched Qwen/budget' if reader=='BASE_FP16' and b in ('256','1024') else 'natural-budget or separate precision/adaptation reference'))
    for d in cfg['datasets']:
        for b in cfg['budgets']:
            for m in METHODS:
                a=groups.get((d,'QBGF_FULL',str(b),'BASE_FP16'),[]);z=groups.get((d,m,str(b),'BASE_FP16'),[])
                ok=bool(a and z) and {r['query_id'] for r in a}=={r['query_id'] for r in z} and (synthetic or {r['query_id'] for r in a}==expected[d])
                result=paired(a,z,cfg['statistics']['primary_metric'][d],cfg['bootstrap_resamples'],cfg['seed']) if ok else {}
                comparisons.append(dict(dataset=d,budget=b,left='QBGF_FULL',right=m,status='COMPLETE' if ok else 'PENDING',**result))
    for c,adj in zip(comparisons,holm([c.get('raw_p',1.) for c in comparisons])):c['extension_holm_p']=adj if c['status']=='COMPLETE' else None;c['extension_family_size']=12;c['claim_scope']='supplemental exploratory; original 39/51 families unchanged'
    out=f/'analysis';out.mkdir(parents=True,exist_ok=True)
    sf=pd.DataFrame(summaries);cf=pd.DataFrame(comparisons);sf.to_csv(out/'merged_validation_summary.csv',index=False);cf.to_csv(out/'paired_extension_comparisons.csv',index=False)
    refs=[]
    for p in (f/'specialists').glob('*/canonical_references/*/identity.json'):
        for row in Store(p.parent,json.loads(p.read_text())).rows():
            refs.append(dict(row,split='VALIDATION',budget='SPECIALIST_NATURAL',comparison_scope='QASPER specialist cross-reader/adapted Mistral; not matched to Qwen 256/1024'))
    rf=pd.DataFrame(refs);rf.to_csv(out/'specialist_references.csv',index=False)
    return dict(summary=sf,comparisons=cf,specialist_references=rf)

def readiness(cfg):
    """Availability is explicit; historical k6 never enters matched tables."""
    f=folder(cfg);status=f/'status.json'
    out=[dict(method=m,status='COMPLETE' if status.exists() and json.loads(status.read_text()).get('complete') else 'PENDING',scope='same frozen VAL IDs, BASE_FP16, 256/1024 whole-passage RAW_SKIP') for m in METHODS]
    from .specialists import SYSTEMS,config
    for system in SYSTEMS:
        sc=config(cfg,system);p=root(sc)/'canonical_references'/system/'identity.json'
        expected={r['query_id'] for r in cohort(cfg)['QASPER']}
        rs=Store(p.parent,json.loads(p.read_text())).rows() if p.exists() else []
        done={r['query_id'] for r in rs}==expected
        out.append(dict(method=system,status='COMPLETE' if done else 'PENDING_QUALIFICATION_OR_FULL',scope='QASPER only; adapted Mistral/SARA k5/n10; natural specialist reference, not budget matched',n_queries=len(rs)))
    out.append(dict(method='HISTORICAL_SFR_SARA_K6',status='HISTORICAL_ONLY_NOT_WEEK12_PAIRED',scope='Existing W10 findings only. Different cohorts/prompt/checkpoint conditions must not enter fresh VAL tables or CIs. No new result fabricated.'))
    return out

def analyze_with_extension(cfg,synthetic=False):
    from src.week12.analysis import analyze
    result=analyze(cfg,synthetic)
    result['c_extension']=merged_analysis(cfg,synthetic)
    if not synthetic:
        import pandas as pd
        table=pd.DataFrame(readiness(cfg));table.to_csv(folder(cfg)/'analysis/baseline_availability.csv',index=False);result['c_extension']['availability']=table
    return result

def require_complete(cfg):
    f=folder(cfg);p=f/'rows/identity.json'
    if not p.exists() or json.loads(p.read_text())!=extension_identity(cfg):raise ValueError('Missing or stale C-EXT identity')
    rs=Store(p.parent,json.loads(p.read_text())).rows()
    for d,es in cohort(cfg).items():
        ids={e['query_id'] for e in es}
        for m in METHODS:
            for budget in cfg['budgets']:
                g=[r for r in rs if r['dataset']==d and r['method']==m and r['budget']==budget]
                if len(g)!=len(ids) or {r['query_id'] for r in g}!=ids:raise ValueError('Incomplete matched C-EXT surface')
    checks=readiness(cfg)
    required=set(METHODS)|{'SARA_CANONICAL_BM25_K5_N10'}
    if any(r['status']!='COMPLETE' for r in checks if r['method'] in required):raise ValueError('Complete budget-matched C-EXT and canonical BM25 SARA QASPER reference before final freeze; optional SFR-SARA remains explicitly pending if unqualified')
