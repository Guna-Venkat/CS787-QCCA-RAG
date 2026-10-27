"""Frozen inference adapters; use original functions, not reimplemented policies."""
import time
import numpy as np
import pandas as pd
from .history import historical
from .materialize import record_view
from ..final_h import protocol as p
from ..backend import CachedClient
from ..runner import canonical_context, run_intervention
from ..cost_accounting import QueryCost


def static(record, matrix, backend, scorer, system, d0=None):
    from src.week7.w7e.prompt_builder import build_mixed_fidelity_sara_prompt
    from src.week6.routing import route_r1_semantic_topk
    question=str(record['question']).strip(); passages=record['passages'][:10]
    k=4 if system=='D0' else int(system.rsplit('k',1)[1])
    semantic=system!='BM25_k8'
    full=route_r1_semantic_topk(scorer.score_query_passages(question,[x['text'] for x in passages]),k) if semantic else list(range(k))
    heavy=sorted(set(range(10))-set(full))
    prompt,count,compressed,_=build_mixed_fidelity_sara_prompt(question,passages,full,[],heavy,{},backend.tokenizer)
    cap=128 if semantic else 64  # Actual frozen Week-2/5 BM25 protocol, not an invented 128-token baseline.
    if not semantic:
        from .literal import bm25_prompt
        original,_=bm25_prompt()(question,passages,k,backend.tokenizer)
        p.require(prompt==original,'BM25 historical prompt parity failed')
    comp=matrix[compressed].to(device=backend.device,dtype=backend.model.dtype)
    result=backend.generate(prompt,cap,comp)
    canon=dict(question_id=str(record['question_id']),paper_id=str(record['paper_id']),k=k,
        selected_full_indices=full,compressed_indices=compressed,prediction=result['content'],
        context_tokens=count,saved_generated_token_ids=result['generated_token_ids'])
    row=dict(question_id=str(record['question_id']),paper_id=str(record['paper_id']),split='test',system=system,
        prediction=result['content'],initial_draft=result['content'] if system=='D0' else d0['prediction'],
        question=question,canonical=canon,final_prompt_tokens=result['prompt_tokens'],
        total_query_prompt_tokens=result['prompt_tokens'],total_query_generated_tokens=result['generated_tokens'],
        num_lm_calls=1,evidence_tokens=None,retrieval_query_time=None,selection_latency=None,extraction_latency=None,
        revision_latency=result['latency'],verifier_latency=None,measured_end_to_end_latency=None,
        model=backend.identity(),decoding=dict(do_sample=False,max_new_tokens=cap),
        prompt_hash=p.digest(prompt),cost_note='Single pass; reusable retrieval/SFR indexing separate. Actual serialized LM prompt counted, not historical unwrapped context-token estimate.')
    return row


def raw_intervention(record,d0,pool,encoder,backend,calls,embeddings,cfg,extractions):
    spec=dict(policy='D1',selector='DraftSemantic',evidence_k=2,representation='raw',revision=False)
    row=run_intervention(record_view(record,d0),pool,spec,encoder,backend,calls,embeddings,cfg,extractions)
    row.update(system='D1',split='test',evidence_tokens=len(backend.tokenizer.encode(row['evidence_text'],add_special_tokens=False)),
        source_chunk_hashes={c['chunk_id']:p.digest(c['text']) for c in row['source_chunks']})
    return row


def primary(record,d0,d1,backend,calls,embeddings,encoder,encoder_meta,caches):
    from ..senttopk_e3.runner import prepare_query,generate
    view=record_view(record,d0); chunks=d1['source_chunks'];q=view['inference']['question_id']
    ranked,bank,meta=prepare_query(view['inference'],chunks,backend.tokenizer,encoder,encoder_meta,*caches)
    view['_d1']=d1
    bundle=dict(lineage={q:dict(D1=p.digest(d1))},e_lineage={q:{}},encoder=encoder_meta)
    row=generate(view,chunks,ranked,bank,meta,3,backend,calls,embeddings,bundle)
    from ..utilization_f.prompts import definitions
    row.update(system=p.PRIMARY,split='test',policy='F0',prompt_template_hash=definitions()['F0']['prompt_hash'])
    row['row_content_hash']=p.digest(row)
    return row


def safe(record,d0,f0,backend,calls,embeddings):
    from ..utilization_f.runner import generate
    row=generate(record_view(record,d0),f0,'F2-A',backend,calls,dict(embeddings=embeddings))
    row.update(system=p.SAFE,split='test')
    for field in ('sentence_index_conceptual_seconds','query_embedding_conceptual_seconds','embedding_lookup_compute_latency','sentence_ranking_latency'):
        row[field]=f0.get(field)
    return row


def verify(d0,f0,pipeline,schema):
    from ..action_verification_g.features import payload, online_features
    started=time.perf_counter()
    frame=pd.DataFrame([online_features(payload(d0,f0,f0))])[schema]
    probability=float(pipeline.predict_proba(frame)[0,1]);accept=probability>=.70
    result=dict(f0,system=p.VERIFY,policy=p.VERIFY,prediction=f0['prediction'] if accept else d0['prediction'],
        g2_accept=accept,g2_probability=probability,g2_threshold=.70,verifier_latency=time.perf_counter()-started,
        measured_end_to_end_latency=None,cost_note='Every query pays full F0 generation, including KEEP; verifier model loading excluded from online timing.')
    for key in ('f1','exact_match','delta_f1','help','hurt','answer_changed','row_content_hash'):result.pop(key,None)
    return result


def w8(record,d0,matrix,model,tokenizer,controller,scorer,modules,device='cuda'):
    generation,full,dev=modules
    group=dev.dev_routing(record,d0['canonical'],scorer)
    from src.week8.generator_features import canonical_k4_context
    context_started=time.perf_counter()
    _,prompt,tokens,compressed,comp,full_indices=canonical_k4_context(group,record,matrix,tokenizer)
    context_seconds=time.perf_counter()-context_started
    group['base_context_tokens']=tokens
    draft=generation.generate_continuation(model,tokenizer,prompt,comp,device,diagnostics=True)
    p.require(draft['prediction']==d0['prediction'],'W8 historical first pass differs from shared D0; preserve run and investigate, never silently substitute')
    action,inference=controller.decide(record,group,draft['prediction'],draft['diagnostics'])
    base=dict(question_id=str(record['question_id']),paper_id=str(record['paper_id']),draft=draft,decision=action,
        inference=inference,full_indices=full_indices,compressed_indices=compressed,
        base_pipeline_seconds=context_seconds+draft['generation_seconds']+draft['diagnostics_seconds']+inference['decision_seconds'],reference_action_match=None)
    revision=None
    if action['selected_action']=='PROMOTE':
        started=time.perf_counter()
        prompt,comp,full_indices,compressed=generation.promoted_prompt(record,group,matrix,tokenizer,action,'G0',draft['prediction'])
        elapsed=time.perf_counter()-started
        revision=generation.generate_continuation(model,tokenizer,prompt,comp,device)
        revision.update(prompt_build_seconds=elapsed,full_indices=full_indices,compressed_indices=compressed)
    row=full.evaluate_row(record,base,revision,'original_SARA','G0','H05-exact-recovered')
    row.update(system='original_SARA_G0',split='test',initial_draft=d0['prediction'],question=str(record['question']).strip(),
        total_query_prompt_tokens=row['cumulative_prompt_tokens'],total_query_generated_tokens=row['generated_tokens'],
        num_lm_calls=row['num_generation_passes'],evidence_tokens=None,retrieval_query_time=None,
        selection_latency=inference['decision_seconds'],extraction_latency=None,revision_latency=row['revision_generation_seconds'],
        verifier_latency=None,measured_end_to_end_latency=None,
        cost_note='Exact original W8 serialized prompt sums; diagnostics/decision timings separate; cached retrieval/SFR and cold loading excluded. No invented end-to-end latency.')
    return row
