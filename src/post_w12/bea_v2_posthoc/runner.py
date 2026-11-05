"""Manual, durable post-hoc runner; original predictors and science are read-only."""
import shutil
import time
from .protocol import *
from src.post_w12.bea_v2.protocol import legacy, old as historical_root
from src.post_w12.bea_v2.data import problem
from src.post_w12.bea_v2.stages import allocation, tokenizer

def stores(c, record):
    return tuple(Store(output(c)/p,record['identity']) for p in
                 ['selections','rows','cache/generations'])

def historical(old,v1,d,q,m,b):
    key=[d,q,'BEA_QBGF' if m=='BEA_QBGF_V1' else m,b]
    path=historical_root(old)/'cache/results'/(digest(key)+'.json')
    if not path.exists():
        return None
    return legacy(old,'cache/results',key,v1['identity_hash'])

def check_row(r,d,q,m,b,member,parent,p,sel=None,tok=None,e=None):
    from src.week12.reader import prompt_ids
    original_m='BEA_QBGF' if m=='BEA_QBGF_V1' else m
    if (r['dataset']!=d or r['query_id']!=q or r['method'] not in (m,original_m)
        or r['budget']!=b or r['cluster_id']!=member['group']
        or r['source_hash']!=p['source_hash'] or r['evidence_tokens']>b
        or r['reader']!='BASE_FP16'
        or r['model_hash']!=digest([parent['models']['Qwen'],'BASE_FP16',None])):
        raise ValueError('Incompatible result/source/model/group: '+str([d,q,m,b]))
    if sel is not None:
        expected=digest(prompt_ids(tok,e['question'],e['task_type'],sel['evidence_ids'],parent['labels']))
        if r['prompt_hash']!=expected or r['evidence_ids']!=sel['evidence_ids']:
            raise ValueError('Historical evidence or prompt differs from frozen selection')

def selection(old,v1,member,d,b,m,tok):
    if m=='BEA_V2':
        return allocation(old,v1,member,d,b,tok,0.00008)
    # Canonical historical functions, only for a genuinely absent control.
    from src.post_w12.bea_qbgf.allocator import allocate as v1_allocate, behavior
    from src.week12.algorithm import choose
    p=problem(old,v1,d,member['query_id'])
    sel=(v1_allocate(p['pool'],p['sentences'],p['scores'],tok,b,v1['identity']['parameters'])
         if m=='BEA_QBGF_V1' else choose(p['pool'],p['sentences'],p['props'],
             p['passage_scores'],p['sentence_scores'],tok,b,m))
    sel['lineage']=p['lineage']
    return dict(selection=sel,behavior=behavior(sel,p['sentences'],p['scores']),source_hash=p['source_hash'])

def preflight(c=None):
    c,old,parent,v1,s,cohort,f=context(c)
    with lock(c):
        context(c,persist=True)
        selections,rows,_=stores(c,f)
        completed=0; reused=0; missing_controls=[]; missing_v2=0; missing_problems=[]
        for d in DATASETS:
            for member in cohort['selected'][d]:
                q=member['query_id']
                try:p=problem(old,v1,d,q)
                except ValueError:missing_problems.append([d,q]);continue
                for b in BUDGETS:
                    for m in METHODS:
                        key=[d,q,m,b];r=rows.get(key)
                        if r is not None:
                            check_row(r,d,q,m,b,member,parent,p);completed+=1
                        elif m=='BEA_V2':missing_v2+=1
                        else:
                            r=historical(old,v1,d,q,m,b)
                            if r is None:missing_controls.append(key)
                            else:check_row(r,d,q,m,b,member,parent,p);reused+=1
        result=dict(label=LABEL,notice=NOTICE,identity_hash=f['identity_hash'],
            official_calibration='CALIBRATION_FAILED_TERMINATED',tau_stop=0.00008,
            queries={d:len(cohort['selected'][d]) for d in DATASETS},logical_rows=1200,
            completed_posthoc_rows=completed,reusable_control_rows=reused,
            missing_v2_answer_upper_bound=missing_v2,missing_controls=missing_controls,
            missing_problem_caches=missing_problems,
            physical_call_upper_bound=missing_v2+len(missing_controls),
            free_disk_gib=shutil.disk_usage(output(c)).free/1024**3,
            reader_loaded=False,predictor_retraining=False,automatic_expansion=False)
        # Status is a derived view; result shards remain immutable.
        text(output(c)/'STATUS.json',__import__('json').dumps(result,indent=2)+'\n')
        return result

def prepare(c=None, allow_missing_control_restore=False):
    c,old,parent,v1,s,cohort,f=context(c)
    with lock(c):
        context(c,persist=True)
        tok=tokenizer(parent);selections,rows,_=stores(c,f)
        count=0
        for d in DATASETS:
            for member in cohort['selected'][d]:
                q=member['query_id'];p=problem(old,v1,d,q)
                for b in BUDGETS:
                    for m in METHODS:
                        key=[d,q,m,b]
                        if rows.get(key) is not None:continue
                        r=historical(old,v1,d,q,m,b) if m!='BEA_V2' else None
                        if r is not None:
                            check_row(r,d,q,m,b,member,parent,p)
                            rows.put(key,dict(r,label=LABEL,study=LABEL,method=m,
                                split='DEVELOPMENT',source_split='VAL',notice=NOTICE,
                                source_result_identity=r.get('identity_hash'),
                                identity_hash=f['identity_hash'],cache_source='read-only frozen control',
                                original_physical_generation_performed=r.get('physical_generation_performed'),
                                physical_generation_performed=False,conceptual_lm_calls=1))
                            continue
                        if m!='BEA_V2' and not allow_missing_control_restore:
                            raise ValueError('Genuinely missing control '+str(key)+
                                '; inspect preflight and explicitly allow canonical restoration if desired')
                        if selections.get(key) is None:
                            item=selection(old,v1,member,d,b,m,tok)
                            selections.put(key,dict(item,label=LABEL))
                        count+=1
                print('Post-hoc prepared/reused:',d,q,flush=True)
        return dict(label=LABEL,missing_answer_selections=count,reader_generations=0)

def generate(c=None, *, explicit=False, allow_missing_control_restore=False):
    if not explicit:
        raise ValueError('Set RUN_FULL=True explicitly; no implicit reader run')
    c,old,parent,v1,s,cohort,f=context(c)
    from src.post_w12.bea_qbgf.resources import load_val
    from src.week12.reader import Reader,prompt_ids
    from src.week11.evaluation import evaluate
    with lock(c):
        context(c,persist=True)
        selections,rows,gen=stores(c,f)
        xs={(e['dataset'],e['query_id']):e for e in load_val(parent,s,cohort)}
        tok=tokenizer(parent);reader=None;start=time.time();done=0
        try:
            for d in DATASETS:
                for member in cohort['selected'][d]:
                    q=member['query_id'];e=xs[d,q];p=problem(old,v1,d,q)
                    if e['cluster_id']!=member['group'] or digest(e['inference'])!=p['source_hash']:
                        raise ValueError('Source or frozen development group changed')
                    for b in BUDGETS:
                        for m in METHODS:
                            key=[d,q,m,b];r=rows.get(key)
                            if r is not None:
                                check_row(r,d,q,m,b,member,parent,p);done+=1;continue
                            if m!='BEA_V2' and not allow_missing_control_restore:
                                raise ValueError('Missing control must be reused or explicitly restored')
                            item=selections.get(key)
                            if item is None:raise ValueError('Run preparation before generation: '+str(key))
                            sel=item['selection']
                            if item['source_hash']!=p['source_hash']:raise ValueError('Prepared source changed')
                            expected=digest(prompt_ids(tok,e['question'],e['task_type'],sel['evidence_ids'],parent['labels']))
                            gkey=[d,q,expected];answer=gen.get(gkey);physical=False
                            source='post-hoc exact-prompt cache'
                            if answer is None:
                                for oldm in METHODS[:-1]:
                                    for oldb in BUDGETS:
                                        candidate=historical(old,v1,d,q,oldm,oldb)
                                        if candidate is not None and candidate['prompt_hash']==expected and candidate['source_hash']==p['source_hash']:
                                            fields=['prediction','raw_prediction','reader','model_hash','prompt_hash','prompt_tokens','generated_tokens','generation_seconds','generation_cap_reached']
                                            answer={k:candidate[k] for k in fields if k in candidate}
                                            source='read-only historical exact-prompt '+oldm;break
                                    if answer is not None:break
                            if answer is None:
                                if shutil.disk_usage(output(c)).free<old['minimum_free_disk_gib']*1024**3:
                                    raise RuntimeError('Insufficient disk; completed rows can resume')
                                if reader is None:
                                    import torch
                                    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<18*1024**3:
                                        raise RuntimeError('Frozen BASE_FP16 needs a free GPU (18 GiB). Do not interrupt other studies; resume later.')
                                    reader=Reader(parent,'BASE_FP16')
                                answer=reader.generate(e,sel);physical=True;source='new canonical BASE_FP16 generation'
                            if (answer['prompt_hash']!=expected or answer['reader']!='BASE_FP16'
                                or answer['model_hash']!=digest([parent['models']['Qwen'],'BASE_FP16',None])):
                                raise ValueError('Reader/model/prompt identity mismatch')
                            gen.put(gkey,answer)
                            r=dict(label=LABEL,study=LABEL,notice=NOTICE,dataset=d,query_id=q,
                                paper_id=e['paper_id'],cluster_id=member['group'],split='DEVELOPMENT',source_split='VAL',
                                method=m,budget=b,**evaluate(answer['prediction'],e),**answer,
                                gold=e['evaluation']['gold_label'] if d=='SciFact' else e['evaluation']['gold_answer'],
                                correct=int(answer['prediction']==e['evaluation']['gold_label']) if d=='SciFact' else None,
                                evidence_tokens=sel['evidence_tokens'],unused_budget=b-sel['evidence_tokens'],
                                budget_utilization=sel['evidence_tokens']/b,evidence_ids=sel['evidence_ids'],
                                evidence_text=sel['text'],evidence_hash=sel['evidence_hash'],selected_unit_ids=sel['selected_unit_ids'],
                                behavior=item['behavior'],stop_reason=sel.get('stop_reason','FROZEN_BASELINE'),
                                source_hash=p['source_hash'],predictor=item.get('predictor'),tau_stop=0.00008 if m=='BEA_V2' else None,
                                physical_generation_performed=physical,conceptual_lm_calls=1,
                                cache_source=source,identity_hash=f['identity_hash'])
                            check_row(r,d,q,m,b,member,parent,p,sel,tok,e)
                            rows.put(key,r)
                            write(output(c)/'receipts'/(digest(key)+'.json'),dict(label=LABEL,
                                key=key,cache_source=source,physical_generation_performed=physical,
                                conceptual_lm_calls=1,identity_hash=f['identity_hash']))
                            done+=1
                            print(f'POST-HOC saved {done}/1200 logical rows; {d} {m} @{b}; elapsed={time.time()-start:.0f}s',flush=True)
        finally:
            if reader is not None:reader.close()
        expected_keys=keys(cohort)
        if any(rows.get(k) is None for k in expected_keys):raise ValueError('Incomplete post-hoc grid')
        write(output(c)/'COMPLETE.json',dict(label=LABEL,identity_hash=f['identity_hash'],
            logical_rows=1200,keys_hash=digest(expected_keys),automatic_expansion=False))
    return dict(label=LABEL,complete=True,logical_rows=1200)
