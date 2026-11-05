"""Manual notebook entry points; automatic missing work, no hidden full expansion."""
import csv,io,json,time,shutil
from pathlib import Path
from src.week12.protocol import ROOT,digest
from .protocol import load,freeze,preregister,output,Store,save,text,LABEL,METHODS,read,exclusive
from .allocator import aspects,signals,allocate,behavior
from .resources import IsolatedLab,load_val,compatible_prediction

def stores(c,f):
    out=output(c);ident=f['identity']
    return Store(out/'cache/problems',f.get('input_identity',ident)),Store(out/'cache/selections',ident),Store(out/'cache/results',ident),Store(out/'cache/generations',ident)

def logical(cohort,c):
    return [[d,r['query_id'],m,b] for d,rows in cohort['selected'].items() for r in rows for b in c['budgets'] for m in METHODS]

def plan(c=None,stage='pilot'):
    c=c or load()
    if not (output(c)/'calibration/CALIBRATION.json').exists():
        parent,s,cohort,f=preregister(c)
        return dict(calibration_required=True,automatic_run_missing=True,logical_rows=len(logical(cohort,c)),
                    compatible_local_rows=0,missing_or_unverified=len(logical(cohort,c)))
    parent,s,cohort,f=freeze(c)
    if stage=='expanded':cohort=read(output(c)/'expanded/EXPANDED_COHORT.json')
    _,selections,results,_=stores(c,f);keys=logical(cohort,c);rows=[results.get(k) for k in keys]
    missing=[k for k,v in zip(keys,rows) if v is None]
    legacy=ROOT/parent['output_root'];folders=['validation/C/rows','validation/C/prepared','generation']
    inventory={p:dict(exists=(legacy/p).exists(),identity_present=(legacy/p/'identity.json').exists(),
                      files=sum(1 for q in (legacy/p).glob('*.json') if q.name!='identity.json')) for p in folders}
    obj=dict(label=LABEL,stage=stage,queries=sum(len(v) for v in cohort['selected'].values()),logical_rows=len(keys),
             compatible_local_rows=len(keys)-len(missing),missing_or_unverified=len(missing),
             prepared_selection_hits=sum(selections.get(k) is not None for k in keys),
             automatic_run_missing=bool(missing),physical_generation_upper_bound=len(missing),
             legacy_inventory=inventory,legacy_rows_not_yet_verified=True,identity_hash=f['identity_hash'],
             note='Legacy presence alone is not a hit. Exact current pool/evidence/prompt is verified after preparation. Entirely absent caches trigger all missing pilot rows, never historical reruns or automatic expansion.')
    save(output(c)/stage/'execution_plan.json',obj,replace=True)
    audit='# BEA-QBGF runtime preflight\n\nPOST_W12_EXPLORATORY; VAL only; final W12 policy unchanged.\n\n```json\n'+json.dumps(obj,indent=2)+'\n```\n\n'
    audit+='Canonical definitions: src/week12/algorithm.py (FIXED_SENTENCE/QBGF_FULL/serializer), src/week12/laboratory.py (SFR top10), src/week11/hierarchy.py, src/week12/reader.py. Models, revisions, precision, tokenizer, prompt/parser/source hashes and exact cohort are in BEA_QBGF_V1_FREEZE.json and PILOT_COHORT.json. Missing embedding/hierarchy caches are reconstructed only under this new output root. Frozen SQLite sources are read-only.\n'
    text(output(c)/'AUDIT_RUNTIME.md',audit)
    return obj

def build_problem(lab,e):
    pool,sent,props,ps,ss,lineage=lab.prepare(e,'SFR');a=aspects(e['question'])
    if sent:
        sv,scost=lab.encoder.encode([u['source_text'] for u in sent],'document')
        qv,qcost=lab.encoder.encode([e['question']],'query');av,acost=lab.encoder.encode(a,'query')
        score=signals(sv,qv[0],av)
        costs=dict(sentence=scost,query=qcost,aspects=acost)
    else:score=dict(relevance=[],aspect_scores=[[] for _ in a],pairwise=[]);costs={}
    return dict(pool=pool,sentences=sent,props=props,passage_scores=ps,sentence_scores=ss,lineage=lineage,
                 scores=score,aspects=a,embedding_costs=costs,source_hash=digest(e['inference']))

@exclusive
def prepare(c=None,stage='pilot',run_missing=True):
    c=c or load();parent,s,cohort,f=freeze(c)
    if stage=='expanded':cohort=read(output(c)/'expanded/EXPANDED_COHORT.json')
    problems,selections,results,_=stores(c,f)
    xs=load_val(parent,s,cohort)
    if not run_missing:
        missing=[k for k in logical(cohort,c) if results.get(k) is None]
        if missing:raise ValueError(f'{len(missing)} outputs missing; enable automatic missing work')
        return
    from transformers import AutoTokenizer
    from src.week12.algorithm import choose,validate_spans,serialize
    tok=AutoTokenizer.from_pretrained(parent['models']['Qwen']['id'],revision=parent['models']['Qwen']['revision'],local_files_only=True)
    lab=None;start=time.time()
    try:
        for number,e in enumerate(xs,1):
            d,q=e['dataset'],e['query_id'];keys=[[d,q,m,b] for b in c['budgets'] for m in METHODS]
            if all(results.get(k) is not None for k in keys):continue
            problem=problems.get([d,q])
            if problem is None:
                if lab is None:lab=IsolatedLab(parent,tok,c)
                problem=build_problem(lab,e)
                problems.put([d,q],problem)
            if problem['source_hash']!=digest(e['inference']):raise ValueError('Current source differs from prepared cache')
            p=problem
            for b in c['budgets']:
                for m in METHODS:
                    key=[d,q,m,b]
                    if selections.get(key) is not None:continue
                    sel=(allocate(p['pool'],p['sentences'],p['scores'],tok,b,f['identity']['parameters']) if m=='BEA_QBGF' else
                         choose(p['pool'],p['sentences'],p['props'],p['passage_scores'],p['sentence_scores'],tok,b,m))
                    validate_spans(sel['units'],e['inference']['documents']);check=serialize(sel['units'],tok)
                    if check['evidence_ids']!=sel['evidence_ids'] or check['evidence_tokens']!=sel['evidence_tokens']:raise ValueError('Serializer does not reproduce selection')
                    sel.update(lineage=p['lineage'])
                    selections.put(key,dict(selection=sel,behavior=behavior(sel,p['sentences'],p['scores']),source_hash=p['source_hash']))
            print(f'BEA preparation {number}/{len(xs)} queries; elapsed={time.time()-start:.0f}s; saved selections reused',flush=True)
    finally:
        if lab is not None:lab.close() # SFR must leave GPU before reader loads.

def manifest(c,stage,f,cohort,events=None):
    _,_,results,_=stores(c,f);out=[]
    for key in logical(cohort,c):
        r=results.get(key)
        if r is None:continue
        event=(events or {}).get(tuple(key),dict(cache_hit=True,cache_source='saved local result; not newly executed',physical_generation_performed=False))
        out.append(dict(label=LABEL,method=r['method'],dataset=r['dataset'],query_id=r['query_id'],budget=r['budget'],
                        status='COMPLETE',**event,identity_hash=f['identity_hash'],
                        result_path=str((output(c)/stage/'rows'/(digest(key)+'.json')).relative_to(ROOT))))
    if out:
        stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
        text(output(c)/stage/'cache_manifest.csv',stream.getvalue())
        if stage=='pilot':text(output(c)/'cache_manifest.csv',stream.getvalue())
    return out

@exclusive
def generate(c=None,stage='pilot',run_missing=True):
    c=c or load();parent,s,cohort,f=freeze(c)
    if stage=='expanded':cohort=read(output(c)/'expanded/EXPANDED_COHORT.json')
    _,selections,results,gen=stores(c,f);xs={(e['dataset'],e['query_id']):e for e in load_val(parent,s,cohort)}
    from transformers import AutoTokenizer
    from src.week12.reader import Reader,prompt_ids
    from src.week11.evaluation import evaluate
    tok=AutoTokenizer.from_pretrained(parent['models']['Qwen']['id'],revision=parent['models']['Qwen']['revision'],local_files_only=True)
    stage_store=Store(output(c)/stage/'rows',f['identity']);reader=None;start=time.time();rejections=[];events={}
    try:
        for number,key in enumerate(logical(cohort,c),1):
            d,q,m,b=key;e=xs[d,q];old=results.get(key)
            if old is not None:
                stage_store.put(key,old)
                events[tuple(key)]=dict(cache_hit=True,cache_source='POST-W12 local completed row',physical_generation_performed=False)
                continue
            if not run_missing:raise ValueError('Missing result and RUN_MISSING=False: '+str(key))
            item=selections.get(key)
            if item is None:raise ValueError('Run preparation before generation: '+str(key))
            if item['source_hash']!=digest(e['inference']):raise ValueError('Source changed')
            sel=item['selection'];gkey=[d,q,digest(sel['evidence_ids']),'BASE_FP16']
            answer=gen.get(gkey);physical=False;source='POST-W12 exact-prompt cache' if answer is not None else None;reasons=[]
            if answer is None and c['cache']['reuse_compatible']:
                answer,source,reasons=compatible_prediction(parent,read(ROOT/'results/week12/W13_FINAL_METHOD_FREEZE.json'),e,sel,tok)
            if reasons:rejections.append(dict(key=key,reasons=reasons))
            if answer is None:
                if shutil.disk_usage(output(c)).free<c['minimum_free_disk_gib']*1024**3:raise RuntimeError('Insufficient free disk; saved rows can resume')
                if reader is None:
                    import torch
                    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<18*1024**3:
                        raise RuntimeError('Qwen BASE_FP16 needs a free GPU (18 GiB headroom). Wait for POST-W9/other work to finish; saved preparation remains resumable. No precision/model substitution.')
                    reader=Reader(parent,'BASE_FP16')
                answer=reader.generate(e,sel);physical=True;source='new canonical BASE_FP16 generation'
            expected=digest(prompt_ids(tok,e['question'],e['task_type'],sel['evidence_ids'],parent['labels']))
            if answer['prompt_hash']!=expected or answer['reader']!='BASE_FP16' or answer['model_hash']!=digest([parent['models']['Qwen'],'BASE_FP16',None]):raise ValueError('Generation identity mismatch')
            gen.put(gkey,answer);metrics=evaluate(answer['prediction'],e)
            row=dict(label=LABEL,dataset=d,query_id=q,paper_id=e['paper_id'],cluster_id=e['cluster_id'],split='VAL',
                     method=m,budget=b,**metrics,**answer,gold=e['evaluation']['gold_label'] if d=='SciFact' else e['evaluation']['gold_answer'],
                     correct=int(answer['prediction']==e['evaluation']['gold_label']) if d=='SciFact' else None,
                     evidence_tokens=sel['evidence_tokens'],unused_budget=b-sel['evidence_tokens'],budget_utilization=sel['evidence_tokens']/b,
                     evidence_ids=sel['evidence_ids'],evidence_text=sel['text'],evidence_hash=sel['evidence_hash'],
                     source_hash=item['source_hash'],candidate_ids=sel['lineage']['candidate_ids'],candidate_content_hashes=sel['lineage']['candidate_content_hashes'],
                     selected_unit_ids=sel['selected_unit_ids'],behavior=item['behavior'],stop_reason=sel.get('stop_reason','FROZEN_BASELINE'),
                     cache_hit=not physical,cache_source=source,physical_generation_performed=physical,conceptual_lm_calls=1,
                     identity_hash=f['identity_hash'],serializer='additive-token-id-blocks-v1')
            results.put(key,row);stage_store.put(key,row)
            events[tuple(key)]=dict(cache_hit=not physical,cache_source=source,physical_generation_performed=physical)
            # Commit a receipt after EACH row, not only at notebook completion.
            save(output(c)/stage/'manifest_rows'/(digest(key)+'.json'),dict(key=key,cache_hit=not physical,cache_source=source,
                 physical_generation_performed=physical,identity_hash=f['identity_hash'],result_path=str(stage_store.file(key).relative_to(ROOT))))
            if number%5==0:print(f'BEA {stage} saved {number}/{len(logical(cohort,c))} rows; elapsed={time.time()-start:.0f}s; method={m}',flush=True)
    finally:
        if reader is not None:reader.close()
        manifest(c,stage,f,cohort,events)
        if rejections:save(output(c)/stage/('cache_rejections_'+digest(rejections)+'.json'),dict(label=LABEL,rejections=rejections))
    save(output(c)/stage/'complete.json',dict(label=LABEL,rows=len(logical(cohort,c)),identity_hash=f['identity_hash'],cohort_hash=digest(cohort)))
    return [results.get(k) for k in logical(cohort,c)]
