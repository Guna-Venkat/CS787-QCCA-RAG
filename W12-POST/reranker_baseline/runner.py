"""Read-only frozen inputs, CPU scoring, manual reader execution and durable resume."""
import shutil
import time
from .protocol import *
from .minilm_reranker import Scorer,pack,rank_scores

def check_frozen_identity(p,e,member,parent,d,q):
    """Top-10 is a ceiling; preserve a short frozen pool without padding it."""
    checks=dict(source=p['source_hash']==digest(e['inference']),group=e['cluster_id']==member['group'],
        retriever=p['lineage']['retriever']=='SFR',available_frozen_width=0<len(p['pool'])<=10,
        candidate_ids=p['lineage']['candidate_ids']==[u['unit_id'] for u in p['pool']],
        candidate_content_hashes=p['lineage']['candidate_content_hashes']==[digest(u['source_text']) for u in p['pool']],
        encoder=p['lineage']['encoder']==parent['models']['SFR'])
    failed=[name for name,ok in checks.items() if not ok]
    if failed:
        raise ValueError('Frozen SFR top-10/source/group changed: '+str([d,q])+
            '; failed checks='+','.join(failed)+'; actual frozen pool size='+str(len(p['pool'])))

def load_inputs(ctx,mode):
    from src.post_w12.bea_qbgf.resources import load_val
    from src.post_w12.bea_v2.data import problem
    small=members(ctx,mode);cohort=dict(ctx['cohort'],selected=small)
    examples={(e['dataset'],e['query_id']):e for e in load_val(ctx['parent'],ctx['splits'],cohort)}
    result=[]
    for d in DATASETS:
        for member in small[d]:
            q=member['query_id'];e=examples[d,q];p=problem(ctx['old'],ctx['v1'],d,q)
            check_frozen_identity(p,e,member,ctx['parent'],d,q)
            from src.week11.cache import digest as unit_digest
            if any(u['hash']!=unit_digest(u['source_text']) for u in p['pool']+p['sentences']):
                raise ValueError('Frozen unit text/hash changed')
            docs={u['document_id']:u['text'] for u in e['inference']['documents']}
            if len({u['unit_id'] for u in p['sentences']})!=len(p['sentences']):raise ValueError('Duplicate frozen sentences')
            for u in p['sentences']:
                if u['unit_type']!='SENTENCE' or docs[u['document_id']][u['start_char']:u['end_char']]!=u['source_text']:
                    raise ValueError('Frozen sentence boundary/source changed')
            # Match all source controls to this exact current source (not new retrieval).
            for b in BUDGETS:
                for m in CONTROLS:
                    if ctx['source'].get([d,q,m,b])['source_hash']!=p['source_hash']:
                        raise ValueError('Baseline source pool differs from reranker source')
            result.append(dict(dataset=d,query_id=q,member=member,example=e,problem=p))
    if mode=='smoke' and len(result)!=3:raise ValueError('Smoke is exactly three total queries')
    return result

def token_counter(ctx):
    from src.post_w12.bea_v2.stages import tokenizer
    return tokenizer(ctx['parent'])

def score_key(item):return [item['dataset'],item['query_id']]

def source_fingerprint(item):
    p=item['problem']
    return digest(dict(query=item['example']['inference']['question'],lineage=p['lineage'],
        sentences=[{k:u[k] for k in ['unit_id','passage_id','source_text','start_char','end_char','hash','document_order','token_count']} for u in p['sentences']]))

def preflight():
    with lock():
        ctx=context(persist=True);data=load_inputs(ctx,'full');scores,sels,rows,_=stores(ctx)
        expected=[[x['dataset'],x['query_id'],METHOD,b] for x in data for b in BUDGETS]
        extras={p.name for p in rows.path.glob('*.json')}-{rows.file(k).name for k in expected}
        if extras:raise ValueError('Unexpected reranker answer IDs')
        result=dict(label=LABEL,notice=NOTICE,identity_hash=ctx['record']['identity_hash'],
            cohorts=[dict(dataset=d,n_queries=len(ctx['cohort']['selected'][d]),n_groups=len({m['group'] for m in ctx['cohort']['selected'][d]}),budgets=BUDGETS) for d in DATASETS],
            matched_control_rows=1200,expected_minilm_rows=300,reranker_queries_scored=sum(scores.get(score_key(x)) is not None for x in data),
            configured_retrieval_width=10,
            short_frozen_pools=[dict(dataset=x['dataset'],query_id=x['query_id'],available_passages=len(x['problem']['pool']))
                for x in data if len(x['problem']['pool'])<10],
            completed_answer_rows=sum(rows.get(k) is not None for k in expected),
            missing_answer_upper_bound=sum(rows.get(k) is None for k in expected),
            free_disk_gib=shutil.disk_usage(HOME).free/1024**3,model=MODEL,revision=REVISION,
            dependencies=versions(),reader_loaded=False,full_run_enabled=False)
        save(OUT/'protocol/STATUS.json',result,replace=True)
        return result

def _prepare(ctx,data,mode,allow_download):
    from src.post_w12.bea_qbgf.allocator import behavior
    scores_store,sels,rows,_=stores(ctx,mode);tok=token_counter(ctx);scorer=None;prepared=[]
    try:
        for number,item in enumerate(data,1):
            d,q=item['dataset'],item['query_id'];p=item['problem'];key=score_key(item)
            fingerprint=source_fingerprint(item);cached=scores_store.get(key)
            if cached is not None and cached['source_fingerprint']!=fingerprint:
                raise ValueError('Cached CE scores have different query/source identity')
            if cached is None:
                if scorer is None:scorer=Scorer(allow_download=allow_download)
                # No gold/outcomes/LOO/P_help/semantic coverage scores are passed.
                cached=scorer.score(item['example']['inference']['question'],p['sentences'],[u['unit_id'] for u in p['pool']])
                for r in cached['scores']:r.update(query_id=q,dataset=d)
                cached.update(dataset=d,query_id=q,source_fingerprint=fingerprint)
                scores_store.put(key,cached)
            for b in BUDGETS:
                k=[d,q,METHOD,b];old=sels.get(k)
                if old is None:
                    sel=pack(p['sentences'],cached['scores'],tok,b)
                    old=sels.put(k,dict(label=LABEL,selection=sel,source_hash=p['source_hash'],
                        source_fingerprint=fingerprint,behavior=behavior(sel,p['sentences'],p['scores']),
                        ranking_scores_hash=digest(cached['scores'])))
                if old['source_hash']!=p['source_hash'] or old['source_fingerprint']!=fingerprint:
                    raise ValueError('Incompatible prepared evidence')
                prepared.append(dict(dataset=d,query_id=q,budget=b,evidence_tokens=old['selection']['evidence_tokens'],
                    selected_units=len(old['selection']['units']),represented_parents=len(old['selection']['selected_passage_ids'])))
            print(f'MiniLM {mode}: prepared/reused {number}/{len(data)} queries; scores shared by both budgets',flush=True)
    finally:
        if scorer is not None:scorer.close() # CPU model released before any reader generation.
    return prepared

def prepare_full(*,explicit=False,allow_download=False):
    if not explicit:raise ValueError('RUN_FULL=True is required for the 150-query scoring surface')
    with lock():
        ctx=context(persist=True);data=load_inputs(ctx,'full')
        return _prepare(ctx,data,'full',allow_download)

def smoke(*,allow_download=False,reader=False):
    with lock():
        ctx=context(persist=True);data=load_inputs(ctx,'smoke')
        prepared=_prepare(ctx,data,'smoke',allow_download);score_store,sels,rows,_=stores(ctx,'smoke')
        if reader:_generate(ctx,data,'smoke',budgets=[256]) # At most 3, only on explicit manual request.
        report=dict(label=LABEL,smoke_queries=3,device='cpu',model=MODEL,revision=REVISION,
            reader_requested=reader,reader_generations_executed=sum(bool(rows.get([x['dataset'],x['query_id'],METHOD,256]) and rows.get([x['dataset'],x['query_id'],METHOD,256])['physical_generation_performed']) for x in data) if reader else 0,
            preparations=prepared,examples=[])
        for x in data:
            key=score_key(x);scored=score_store.get(key)
            report['examples'].append(dict(dataset=x['dataset'],query_id=x['query_id'],question=x['example']['question'],
                gold=x['example']['evaluation'],model_info=scored['model'],top10=rank_scores(scored['scores'])[:10],
                all_sentence_scores=scored['scores'],allocation={str(b):sels.get([x['dataset'],x['query_id'],METHOD,b])['selection'] for b in BUDGETS},
                reader_result=rows.get([x['dataset'],x['query_id'],METHOD,256]) if reader else None))
        save(OUT/'smoke/SMOKE_REPORT.json',report,replace=True)
        return report

def _generate(ctx,data,mode,budgets=BUDGETS):
    from src.week12.reader import Reader,prompt_ids
    from src.week11.evaluation import evaluate
    from src.post_w12.final_case_study.gold import retention
    _,sels,rows,answers=stores(ctx,mode);tok=token_counter(ctx);reader=None;start=time.time();done=0
    if mode=='smoke' and (len(data)>3 or list(budgets)!=[256]):raise ValueError('Reader smoke limited to three total answers @256')
    try:
        for x in data:
            d,q=x['dataset'],x['query_id'];e=x['example'];p=x['problem']
            for b in budgets:
                key=[d,q,METHOD,b];existing=rows.get(key)
                if existing is not None:
                    if existing['source_hash']!=p['source_hash']:raise ValueError('Completed answer source changed')
                    done+=1;continue
                item=sels.get(key)
                if item is None:raise ValueError('Prepare missing evidence first')
                sel=item['selection'];expected=digest(prompt_ids(tok,e['question'],e['task_type'],sel['evidence_ids'],ctx['parent']['labels']))
                gkey=[d,q,expected];answer=answers.get(gkey);source='MiniLM exact-prompt cache';physical=False
                if answer is None:
                    for m in CONTROLS:
                        for oldb in BUDGETS:
                            r=ctx['source'].get([d,q,m,oldb])
                            if r['prompt_hash']==expected and r['source_hash']==p['source_hash']:
                                fields=['prediction','raw_prediction','reader','model_hash','prompt_hash','prompt_tokens','generated_tokens','generation_seconds','generation_cap_reached']
                                answer={k:r[k] for k in fields if k in r};source='read-only control exact-prompt '+m;break
                        if answer is not None:break
                if answer is None:
                    if shutil.disk_usage(HOME).free<3*1024**3:raise RuntimeError('Disk below 3 GiB; completed rows resumable')
                    if reader is None:
                        import torch
                        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<18*1024**3:
                            raise RuntimeError('Canonical Qwen BASE_FP16 requires a free GPU with 18 GiB headroom; no model/precision substitution')
                        reader=Reader(ctx['parent'],'BASE_FP16')
                    answer=reader.generate(e,sel);physical=True;source='new canonical BASE_FP16 generation'
                answer={k:v for k,v in answer.items() if k!='label'}
                if (answer['prompt_hash']!=expected or answer['reader']!='BASE_FP16'
                    or answer['model_hash']!=digest([ctx['parent']['models']['Qwen'],'BASE_FP16',None])):
                    raise ValueError('Frozen reader/prompt identity mismatch')
                answers.put(gkey,dict(answer,label=LABEL))
                evaluation=evaluate(answer['prediction'],e)
                correct=int(answer['prediction']==e['evaluation']['gold_label']) if d=='SciFact' else None
                ret=retention(e,sel['units'],tok)
                row=dict(label=LABEL,notice=NOTICE,dataset=d,query_id=q,group_id=x['member']['group'],cluster_id=x['member']['group'],
                    budget=b,method=METHOD,question=e['question'],gold=e['evaluation']['gold_label'] if d=='SciFact' else e['evaluation']['gold_answer'],
                    gold_answer=e['evaluation']['gold_answer'],gold_label=e['evaluation']['gold_label'],retriever='SFR',retrieval_width=10,
                    retrieved_passage_count=len(p['pool']),
                    reranker_model=MODEL,reranker_revision=REVISION,selected_sentence_ids=sel['selected_unit_ids'],
                    selected_unit_ids=sel['selected_unit_ids'],selected_passage_ids=sel['selected_passage_ids'],
                    selected_units=len(sel['units']),represented_parents=len(sel['selected_passage_ids']),evidence_tokens=sel['evidence_tokens'],
                    unused_budget=b-sel['evidence_tokens'],budget_utilization=sel['evidence_tokens']/b,evidence_text=sel['text'],
                    evidence_ids=sel['evidence_ids'],evidence_hash=sel['evidence_hash'],**answer,**evaluation,
                    generated_answer=answer['prediction'],F1=evaluation.get('f1') if d!='SciFact' else None,
                    EM=evaluation.get('exact_match') if d!='SciFact' else None,correct=correct,accuracy=correct,
                    format_valid=answer['prediction'] in ctx['parent']['labels'] if d=='SciFact' else True,
                    behavior=item['behavior'],retrospective_gold=ret,source_hash=p['source_hash'],
                    source_fingerprint=item['source_fingerprint'],candidate_ids=p['lineage']['candidate_ids'],
                    candidate_content_hashes=p['lineage']['candidate_content_hashes'],stop_reason=sel['stop_reason'],
                    cache_source=source,physical_generation_performed=physical,conceptual_lm_calls=1,
                    identity_hash=ctx['record']['identity_hash'])
                rows.put(key,row)
                save(OUT/(mode+'/receipts')/(digest(key)+'.json'),dict(label=LABEL,key=key,cache_source=source,
                    physical_generation_performed=physical,conceptual_lm_calls=1,identity_hash=ctx['record']['identity_hash']))
                done+=1;print(f'MiniLM {mode} durable answers {done}/{len(data)*len(budgets)}; elapsed={time.time()-start:.0f}s',flush=True)
    finally:
        if reader is not None:reader.close()
    expected=[[x['dataset'],x['query_id'],METHOD,b] for x in data for b in budgets]
    if any(rows.get(k) is None for k in expected):raise ValueError('Incomplete MiniLM answer surface')
    save(OUT/(mode+'/COMPLETE.json'),dict(label=LABEL,rows=len(expected),keys_hash=digest(expected),identity_hash=ctx['record']['identity_hash']))
    return dict(label=LABEL,completed=len(expected),mode=mode)

def generate_full(*,explicit=False):
    if not explicit:raise ValueError('Explicit RUN_FULL=True required before any full reader generation')
    with lock():
        ctx=context(persist=True);data=load_inputs(ctx,'full')
        return _generate(ctx,data,'full')
