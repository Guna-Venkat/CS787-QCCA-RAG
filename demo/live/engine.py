"""Observable live evidence operations; inference receives no gold/reference fields."""
import re,time,hashlib,json
METHODS=('NONE','RAW_SKIP','FIXED_SENTENCE','QBGF_FULL')
STAGES=('Query','Passages','Retriever','Granularity','Allocation','Reader','Answer')
from .segmentation import DEFAULT as SEGMENTATION,documents as source_documents,annotate,inventory as sentence_inventory
from .protocol import NATIVE,CONTROLLED,VERSION,scorer_identity,invariants
MAX_PASSAGES=20

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def parse_passages(text):
    if not isinstance(text,str):raise ValueError('Paste passages as text.')
    blocks=re.split(r'(?m)^\s*---+\s*$',text)
    return [re.sub(r'^\s*Passage\s+\d+\s*:\s*','',b,flags=re.I).strip() for b in blocks]

def validate(query,passages,method,budget,retriever):
    if not isinstance(query,str) or not query.strip():raise ValueError('Enter a non-empty query.')
    if not passages:raise ValueError('Add at least one evidence passage.')
    if len(passages)>MAX_PASSAGES:raise ValueError('Use at most 20 passages.')
    if any(not isinstance(p,str) or not p.strip() for p in passages):raise ValueError('Fill every passage or remove empty passages before running.')
    if len(query)>2000:raise ValueError('Keep the query below 2,000 characters.')
    if any(len(p)>8000 for p in passages) or sum(map(len,passages))>30000:raise ValueError('Use at most 8,000 characters per passage and 30,000 in total.')
    if method not in METHODS or budget not in (256,1024) or retriever not in ('BM25','SFR'):raise ValueError('Choose a supported method, retriever and budget.')
    return query.strip(),[p.strip() for p in passages]

def input_key(query,passages,budget,retriever,segmentation=SEGMENTATION,scoring_mode=NATIVE):return digest([VERSION,query,passages,budget,retriever,segmentation,scoring_mode,scorer_identity(scoring_mode,retriever)])

def safe_answer(text):
    # Never display a returned hidden-thinking block, even if a model ignores thinking=False.
    if '</think>' in text:text=text.rsplit('</think>',1)[1]
    elif '<think>' in text:return None
    return text.strip()


def option_label(option):
    if option['name']=='DROP':return 'DROP'
    if option['name']=='FULL':return 'FULL PASSAGE'
    k=option.get('actual_k') or int(option['name'].split('_')[1])
    return f'{k} selected evidence unit'+('s' if k!=1 else '')

def packing_trace(pool,sentences,ps,ss,tok,budget,method,selection):
    from src.week12.algorithm import serialize,order
    if method=='NONE':return []
    if method=='QBGF_FULL':
        rank={p['unit_id']:i+1 for i,p in enumerate(pool)}
        return [dict(passage_id=x['passage_id'],rank=rank[x['passage_id']],decision=option_label(x),option_id=x['name'],cost=x['cost'],relevance=ps[rank[x['passage_id']]-1],utility=x['utility'],remaining_budget=None) for x in selection['choices']]
    ranked=pool if method=='RAW_SKIP' else sorted(sentences,key=lambda u:(-ss[u['unit_id']],order(u)))
    chosen=[];trace=[]
    for u in ranked:
        before=serialize(chosen,tok)['evidence_tokens'];trial=serialize(chosen+[u],tok)['evidence_tokens'];fits=trial<=budget
        if fits:chosen.append(u)
        trace.append(dict(unit_id=u['unit_id'],passage_id=u['passage_id'],decision='INCLUDED' if fits else 'DOES NOT FIT — SKIPPED',standalone_tokens=serialize([u],tok)['evidence_tokens'],incremental_cost=trial-before,remaining_budget=budget-(trial if fits else before)))
    if [u['unit_id'] for u in serialize(chosen,tok)['units']]!=selection['selected_unit_ids']:raise ValueError('Visual packing trace differs from actual selected context.')
    return trace

def explanation(result):
    count=len(result['passages']);sel=result['selection'];method=result['method'];units=sel['units'];sources=len({u['document_id'] for u in units})
    lines=[f'{count} passages were provided.']
    if method=='NONE':lines+=['Retrieval and evidence allocation were skipped.','No external evidence was supplied; the reader is configured for query-only generation.']
    else:
        lines += [f"{result['retriever']} ranked the passages; the top {len(result['pool'])} candidates were considered.", 'Controlled retriever mode: common SFR downstream scoring.' if result.get('scoring_mode')==CONTROLLED else 'Pipeline-native mode: downstream scoring follows the selected retrieval pipeline.']
        if method in ('FIXED_SENTENCE','QBGF_FULL'):lines.append(f"The candidate sources contained {len(result['sentences'])} exact source sentence/structural segments.")
        if method=='QBGF_FULL':lines.append('QBGF chose one fidelity option per passage jointly under the shared budget.')
        lines += [f'{len(units)} evidence units from {sources} sources were selected under the {result["budget"]}-token evidence budget.',f'{sel["evidence_tokens"]} evidence tokens were prepared for Qwen3-8B.']
    lines.append('The reader generated the displayed answer.' if result.get('answer') is not None else 'Reader generation was unavailable; the exact context is still ready for inspection.')
    return lines

def run(query,passages,method,budget,retriever,resources,enable_models=False,notify=None,prepared_cache=None,generate_answer=True,segmentation=SEGMENTATION,scoring_mode=NATIVE):
    query,passages=validate(query,passages,method,budget,retriever)
    scoring=scorer_identity(scoring_mode,retriever)
    if method=='QBGF_FULL' and retriever=='BM25' and scoring_mode==NATIVE:
        raise ValueError('BM25-native QBGF is disabled: raw lexical scores are incompatible with the cosine-oriented softmax scale. Select Controlled retriever-only comparison with common SFR scoring.')
    needs_sfr=method!='NONE' and (retriever=='SFR' or scoring_mode==CONTROLLED)
    if needs_sfr and not enable_models:raise ValueError('Enable Live Models for the common SFR scorer or SFR retrieval.')
    states={s:dict(state='pending',detail='') for s in STAGES};started=time.perf_counter();durations={}
    def status(stage,state,detail):
        states[stage]=dict(states[stage],state=state,detail=detail)
        if notify:notify(stage,state,detail,states.copy())
    def begin(stage,detail):status(stage,'processing',detail);durations[stage]=time.perf_counter()
    def end(stage,detail,state='completed'):
        durations[stage]=time.perf_counter()-durations[stage] if isinstance(durations.get(stage),float) else 0.;status(stage,state,detail)
    cache=prepared_cache if prepared_cache is not None else {}
    with resources.lock:
        try:
            begin('Query','Processing query…');tok=resources.tokenizer();query_tokens=len(tok.encode(query,add_special_tokens=False))
            if query_tokens>512:raise ValueError('Keep the query below 512 reader tokens.')
            qvec=None
            if needs_sfr:
                if not enable_models:raise ValueError('Enable Live Models to use SFR.')
                status('Query','processing','Loading the local SFR encoder and encoding query…');qvec=resources.embed([query],'query')[0]
            states['Query']['preview']=dict(query=query,reader_tokens=query_tokens,embedding_model='Salesforce/SFR-Embedding-Mistral' if qvec is not None else None)
            end('Query',f'Query encoded with SFR · {query_tokens} reader tokens' if qvec is not None else f'Query tokenized · {query_tokens} reader tokens')
            begin('Passages','Preparing passages…')
            from src.week11.hierarchy import build
            from src.week12.algorithm import choose,serialize,cosine,validate_spans
            from src.week9.retrieval import bm25
            documents=source_documents(passages,segmentation)
            all_units=annotate(build(documents,tok,dict(max_passage_tokens=256)),segmentation);audit_sentences=sentence_inventory(passages,all_units,tok,segmentation);all_passages=[u for u in all_units if u['unit_type']=='PASSAGE'];source_tokens=sum(p['token_count'] for p in all_passages)
            if source_tokens>16000:raise ValueError('Use no more than 16,000 source tokens for this live presentation.')
            states['Passages']['preview']=dict(passages=len(passages),source_tokens=source_tokens,sentences=sum(u['unit_type']=='SENTENCE' for u in all_units))
            end('Passages',f'{len(passages)} passages · {source_tokens} source tokens')
            pool=[];sent=[];ps=[];ss={};ranking=[]
            if method=='NONE':
                for stage in ('Retriever','Granularity'):
                    status(stage,'skipped','Skipped for query-only generation')
            else:
                begin('Retriever',f'Running {retriever} retrieval…')
                key=digest([query,passages,retriever,getattr(resources,'device','cuda'),segmentation,scoring_mode,scoring]);prepared=cache.get(key)
                if prepared is None:
                    if retriever=='BM25':scores=bm25(query,[p['source_text'] for p in all_passages]).tolist()
                    else:scores=cosine(qvec,resources.embed([p['source_text'] for p in all_passages],'document'))
                    ix=sorted(range(len(all_passages)),key=lambda i:(-scores[i],i))
                    ranking=[dict(unit=all_passages[i],rank=j+1,score=float(scores[i]),in_pool=j<10) for j,i in enumerate(ix)]
                    pool=[r['unit'] for r in ranking[:10]];ps=[r['score'] for r in ranking[:10]];ids={p['unit_id'] for p in pool}
                    sent=[u for u in all_units if u['unit_type']=='SENTENCE' and u['passage_id'] in ids]
                    # Fine relevance is computed only when a sentence-based method requests it.
                    if scoring_mode==CONTROLLED:
                        ps=cosine(qvec,resources.embed([p['source_text'] for p in pool],'document'))
                    prepared=dict(pool=pool,ps=ps,ranking=ranking,sentences=sent,ss=None);cache[key]=prepared
                    if len(cache)>20:cache.pop(next(iter(cache)))
                pool,ps,ranking,sent=prepared['pool'],prepared['ps'],prepared['ranking'],prepared['sentences']
                states['Retriever']['preview']=[dict(rank=r['rank'],passage=r['unit']['document_id'],score=r['score'],tokens=r['unit']['token_count'],candidate=r['in_pool']) for r in ranking]
                end('Retriever',f'{len(pool)} candidates ranked by {retriever}')
                begin('Granularity','Creating evidence units…')
                if method in ('FIXED_SENTENCE','QBGF_FULL'):
                    if prepared['ss'] is None:
                        texts=[u['source_text'] for u in sent]
                        vals=bm25(query,texts).tolist() if retriever=='BM25' and scoring_mode==NATIVE else cosine(qvec,resources.embed(texts,'document')) if texts else []
                        prepared['ss']=dict(zip([u['unit_id'] for u in sent],vals))
                    ss=prepared['ss']
                if method=='QBGF_FULL':detail=f'{len(sent)} source segments · DROP / SENT_1 / SENT_2 / SENT_4 / SENT_8 / FULL'
                elif method=='FIXED_SENTENCE':detail=f'{len(sent)} source segments ranked'
                else:detail=f'{len(pool)} complete passage units'
                states['Granularity']['preview']=[dict(passage=u['document_id'],sentence=u['source_text'],relevance=ss.get(u['unit_id'])) for u in sent] if method in ('FIXED_SENTENCE','QBGF_FULL') else dict(representation='Whole passages')
                end('Granularity',detail)
            begin('Allocation','Allocating evidence under budget…')
            if method=='QBGF_FULL' and scoring.get('kind')!='SFR cosine':raise ValueError('QBGF requires SFR cosine scores in the corrected live protocol')
            selection=choose(pool,sent,[],ps,ss,tok,budget,method);validate_spans(selection['units'],documents)
            trace=packing_trace(pool,sent,ps,ss,tok,budget,method,selection)
            qbgf_options=[]
            if method=='QBGF_FULL':
                from src.week12.algorithm import options
                selected_by_passage={x['passage_id']:x for x in selection['choices']}
                for rank,group in enumerate(options(pool,sent,ps,ss,tok),1):
                    for option in group:
                        chosen=selected_by_passage[option['passage_id']]
                        qbgf_options.append(dict(passage_id=pool[rank-1]['document_id'],parent_id=option['passage_id'],retriever_rank=rank,downstream_relevance=ps[rank-1],option_id=option['name'],label=option_label(option),cost=option['cost'],utility=option['utility'],selected=option['name']==chosen['name'],evidence_text='\n\n'.join(u['source_text'] for u in option['units']),unit_types=[u.get('display_kind',u['unit_type']) for u in option['units']]))
            states['Allocation']['preview']=dict(evidence_tokens=selection['evidence_tokens'],budget=budget,decisions=trace)
            end('Allocation',f'{selection["evidence_tokens"]} / {budget} evidence tokens')
            from src.week12.reader import prompt_ids
            prompt=prompt_ids(tok,query,'freeform_qa',selection['evidence_ids'])
            result=dict(result_type='LIVE_CUSTOM',protocol_version=VERSION,scoring_mode=scoring_mode,scorer_identity=scoring,invariants=invariants(query,passages,method,budget,segmentation,scoring_mode,getattr(resources,'device','cuda'),retriever),allocation_passage_scores=ps,query=query,passages=passages,segmentation=segmentation,sentence_inventory=[dict(row,selected=row['sentence_id'] in selection['selected_unit_ids'],covered_by_full_passage=row['parent_passage_id'] in selection['selected_unit_ids']) for row in audit_sentences],method=method,budget=budget,retriever=retriever,query_tokens=query_tokens,source_tokens=source_tokens,ranking=ranking,pool=pool,sentences=sent,relevance=ss,selection=selection,allocation_trace=trace,qbgf_options=qbgf_options,answer=None,reader='Qwen/Qwen3-8B · BASE_FP16 · '+getattr(resources,'device','cuda').upper(),device=getattr(resources,'device','cuda'),prompt_tokens=len(prompt),generated_tokens=None,reader_error=None,input_key=input_key(query,passages,budget,retriever,segmentation,scoring_mode))
            states['Reader']['preview']=dict(reader='Qwen3-8B',prompt_tokens=len(prompt),evidence_tokens=selection['evidence_tokens'])
            begin('Reader','Loading Qwen3-8B and generating answer…' if enable_models else 'Reader generation disabled; preparing context only…')
            if enable_models and generate_answer:
                try:
                    response=resources.generate(query,selection);answer=safe_answer(response['prediction'])
                    if answer is None:raise RuntimeError('The reader returned no displayable final answer; hidden thinking was suppressed.')
                    result.update(answer=answer,prompt_tokens=response['prompt_tokens'],generated_tokens=response['generated_tokens'],generation_seconds=response.get('generation_seconds'))
                    end('Reader','Answer generated from query only' if method=='NONE' else 'Answer generated from the selected context')
                except Exception as exc:
                    resources.release_gpu();result['reader_error']=f'{type(exc).__name__}: {exc}'
                    end('Reader','Reader generation unavailable in this environment.','failed')
            else:
                result['reader_error']='Live models were not enabled.' if not enable_models else 'Reader generation was not requested.';end('Reader','Reader generation unavailable in this environment.','skipped')
            begin('Answer','Preparing result view…');end('Answer','Pipeline complete' if result['answer'] is not None else 'Evidence pipeline complete · reader unavailable')
            result.update(stages=states,durations=durations,measured_run_seconds=time.perf_counter()-started)
            result['explanation']=explanation(result)
            return result
        except Exception:
            active=next((s for s in STAGES if states[s]['state']=='processing'),None)
            if active:status(active,'failed','This stage could not complete; see the error below.')
            raise
        finally:
            if getattr(resources,'device','cuda')=='cpu':resources.release_gpu()
