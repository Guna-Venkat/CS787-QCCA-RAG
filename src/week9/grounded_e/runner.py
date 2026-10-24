"""Manual grounded W9-E only. Never runs/freeze-selects F-H or rewrites A-D."""
import argparse
import json
import time
from pathlib import Path
from importlib.metadata import version
from ..provenance import branch_guard,digest,file_hash,assert_inference_clean
from ..caching import RowCache,atomic_json,read_jsonl
from ..backend import SaraBackend,CachedClient,load_tokenizer
from ..cost_accounting import QueryCost
from ..runner import canonical_context
from ..revision import revision_messages
from ..evaluation import annotate_outcome
from . import prompts
from .evidence import extractive,parse,tokens
from .protocol import config,inputs,identity,output_dir,preflight_records,e_sources
from .analysis import summarize


def cache_key(cache,inf,policy,chunks,prompt_hash,budget):
    return cache.key(inf['question_id'],inf['paper_id'],[digest(c['text']) for c in chunks],
        dict(policy=policy,prompt_hash=prompt_hash,max_evidence_tokens=budget,source_ids=[c['chunk_id'] for c in chunks]))


def produce(record,chunks,lineage,policy,backend,calls,extractions,embeddings,cfg,seal):
    inf=record['inference'];assert_inference_clean(inf);assert_inference_clean(chunks)
    budget=cfg['max_evidence_tokens']
    context,comp,initial_tokens=canonical_context(record,embeddings,backend.tokenizer,backend.device,backend.model.dtype)
    cost=QueryCost(initial_draft_prompt_tokens=initial_tokens,
        initial_draft_generated_tokens=len(record['canonical'].get('saved_generated_token_ids',backend.tokenizer.encode(inf['initial_draft'],add_special_tokens=False))),
        final_prompt_tokens=initial_tokens)
    # Source selection/retrieval is imported unchanged from the frozen D branch.
    cost.retrieval_query_time=record['_d_cost']['retrieval_query_time']
    cost.selection_latency=record['_d_cost']['selection_latency']
    messages=prompts.messages(policy,inf,chunks,budget) if policy!='E0' else None
    extraction_prompt=backend.render(messages) if messages is not None else prompts.E0_DEFINITION
    extraction_prompt_hash=digest(extraction_prompt)
    key=cache_key(extractions,inf,policy,chunks,extraction_prompt_hash,budget)
    extracted=extractions.rows.get(key)
    if extracted is None:
        started=time.perf_counter()
        if policy=='E0':
            obj=extractive(inf,chunks,backend.tokenizer,budget)
            extraction_calls=[];extraction_prompt_tokens=0;extraction_generated_tokens=0
            raw='';cpu_latency=time.perf_counter()-started;stage_latency=cpu_latency
        else:
            extraction_cost=QueryCost(num_lm_calls=0)
            client=CachedClient(backend,calls,inf,extraction_cost,'extraction',evidence_ids=[c['chunk_id'] for c in chunks])
            result=client.chat(messages,max_tokens=cfg['extraction_max_tokens'])
            raw=result['content']
            obj=parse(policy,raw,chunks,inf['question_id'],backend.tokenizer,budget,cap_hit=result['generated_tokens']>=cfg['extraction_max_tokens'])
            extraction_calls=extraction_cost.model_calls
            extraction_prompt_tokens=extraction_cost.extraction_prompt_tokens
            extraction_generated_tokens=extraction_cost.extraction_generated_tokens
            cpu_latency=None;stage_latency=client.latency
        extracted=extractions.put(key,dict(question_id=inf['question_id'],paper_id=inf['paper_id'],policy=policy,evidence=obj,raw_output=raw,
            prompt_hash=extraction_prompt_hash,extraction_calls=extraction_calls,extraction_prompt_tokens=extraction_prompt_tokens,
            extraction_generated_tokens=extraction_generated_tokens,extraction_latency=stage_latency,cpu_extraction_latency=cpu_latency))
    evidence=extracted['evidence']
    cost.model_calls.extend(extracted['extraction_calls'])
    cost.num_lm_calls+=len(extracted['extraction_calls'])
    cost.extraction_prompt_tokens=extracted['extraction_prompt_tokens']
    cost.extraction_generated_tokens=extracted['extraction_generated_tokens']
    cost.extraction_latency=extracted['extraction_latency']
    # Exact same D answer behavior (revision=False/F0), context and answer budget.
    answer_client=CachedClient(backend,calls,inf,cost,'revision',context=context,retrieval_embeds=comp,
        evidence_ids=[c['chunk_id'] for c in chunks],revision_mode='F0')
    final_prompt_hash=digest(context);answer_fail_closed=False;answer_generation_called=False
    if evidence['extraction_valid']:
        answer_messages=revision_messages(inf,evidence['text'],False)
        final_prompt_hash=digest(answer_client.prepare_prompt(answer_messages))
        answer_generation_called=True
        prediction=answer_client.chat(answer_messages,max_tokens=cfg['max_new_tokens'])['content'].strip()
        cost.revision_latency=answer_client.latency
        if not prediction:
            prediction=inf['initial_draft'];answer_fail_closed=True
    else:
        # No answer generation at all on extraction failure. Original y0 is the
        # actual output; extraction calls remain fully charged.
        prediction=inf['initial_draft']
    raw_tokens=sum(tokens(backend.tokenizer,c['text']) for c in chunks)
    model=backend.identity()
    tokenizer=dict(name_or_path=backend.tokenizer.name_or_path,tokenizer_class=type(backend.tokenizer).__name__,
        transformers_version=version('transformers'),revision_hash=digest({k:v for k,v in seal['model_hashes'].items() if 'tokenizer' in Path(k).name or k.endswith('.model')}))
    used_ids=sorted({i['source_chunk_id'] for i in evidence['items']})
    row=dict(question_id=inf['question_id'],paper_id=inf['paper_id'],policy=policy,extraction_policy=policy,
        initial_draft=inf['initial_draft'],prediction=prediction,selected_chunk_ids=[c['chunk_id'] for c in chunks],
        source_selected_chunk_ids=[c['chunk_id'] for c in chunks],source_chunk_hashes={c['chunk_id']:digest(c['text']) for c in chunks},
        source_d_row_hashes=lineage,retrieval_config_hash=seal['retrieval_config_hash'],selector_config_hash=seal['selector_config_hash'],
        evidence_k=2,selected_chunk_count=len(chunks),evidence_actual_count=len(used_ids),used_evidence_chunk_ids=used_ids,
        extraction_prompt_hash=extraction_prompt_hash,extraction_object_cache_key=key,evidence=evidence,
        evidence_text_hash=digest(evidence['text']),answer_output_hash=digest(prediction),final_prompt_hash=final_prompt_hash,original_draft_prompt_hash=digest(context),answer_generation_called=answer_generation_called,
        final_output_source='original_draft' if evidence['fail_closed'] or answer_fail_closed else 'new_generation',
        fail_closed=evidence['fail_closed'] or answer_fail_closed,failure_reason=evidence['failure_reason'] or ('empty_answer_generation' if answer_fail_closed else None),
        extraction_valid=evidence['extraction_valid'],answer_fail_closed=answer_fail_closed,
        source_raw_tokens=raw_tokens,compression_ratio=evidence['evidence_tokens']/max(raw_tokens,1),
        extraction_rate=policy!='E0',promotion_rate=evidence['extraction_valid'],evidence_budget=budget,
        extraction_decoding=dict(do_sample=False,max_new_tokens=cfg['extraction_max_tokens']),answer_decoding=dict(do_sample=False,max_new_tokens=128),
        model=model,tokenizer=tokenizer,**cost.to_dict())
    return annotate_outcome(row,inf,record['evaluation'])


def verify_gate(directory,ident,ack_warnings=False):
    path=directory/'preflight/preflight_gate.json'
    if not path.is_file():raise ValueError('Run the deterministic FIT preflight before full execution')
    gate=json.loads(path.read_text())
    if gate['protocol_identity_hash']!=digest(ident) or not gate['implementation_checks_passed']:
        raise ValueError('Preflight is stale/incomplete for this exact source/model/prompt/budget configuration')
    for p,h in gate['artifact_hashes'].items():
        if file_hash(p)!=h:raise ValueError('Preflight artifacts changed')
    if gate['warnings'] and not ack_warnings:
        raise ValueError('Review displayed preflight warnings; only then explicitly acknowledge with ACK_PREFLIGHT_WARNINGS=True / --ack-preflight-warnings. No automatic prompt tuning.')
    return gate


def validate_rows(rows,records,chunks,cfg):
    expected={(r['inference']['question_id'],p) for r in records for p in prompts.POLICIES}
    observed=[(r['question_id'],r['policy']) for r in rows]
    if len(observed)!=len(set(observed)) or set(observed)!=expected:raise ValueError('Incomplete or duplicate E grid')
    for r in rows:
        ev=r['evidence'];source={c['chunk_id']:c for c in chunks[r['question_id']]}
        if r['source_selected_chunk_ids']!=list(source):raise ValueError('E/D source IDs changed')
        if ev['evidence_tokens']>cfg['max_evidence_tokens']:raise ValueError('Evidence budget exceeded')
        if ev['fail_closed'] and (r['prediction']!=r['initial_draft'] or r['answer_generation_called']):raise ValueError('Fail-closed answer changed or model called')
        for item in ev['items']:
            c=source[item['source_chunk_id']]
            if c['text'][item['char_start']:item['char_end']]!=item['exact_source_text'] or item['exact_source_text']!=item['support_span'] or item['source_paper_id']!=c['paper_id']:
                raise ValueError('Accepted span lacks exact canonical source support')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=('prepare','preflight','full','analyze'),required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--ack-preflight-warnings',action='store_true')
    args=parser.parse_args(argv);branch_guard()
    cfg=config(args.config);bundle=inputs(cfg);ident=identity(cfg,bundle);directory=output_dir(cfg)
    atomic_json(directory/'protocol/grounded_e_identity.json',ident)
    atomic_json(directory/'protocol/prompt_templates.json',ident['prompt_templates'])
    if args.stage=='prepare':
        print('Read-only D seal and grounded E protocol prepared; no model loaded.',flush=True);return
    if args.stage=='full':
        gate=verify_gate(directory,ident,args.ack_preflight_warnings)
        print(json.dumps(dict(preflight_quality=gate['quality'],warnings=gate['warnings'])),flush=True)
    records=preflight_records(bundle['records'],cfg) if args.stage=='preflight' else bundle['records']
    stage_dir=directory/('preflight' if args.stage=='preflight' else 'fit')
    cache_ident=dict(stage='preflight' if args.stage=='preflight' else 'full',protocol=ident)
    result_path=stage_dir/'grounded_e_results.jsonl'
    if args.stage=='analyze':
        rows=read_jsonl(result_path)
        if json.loads(result_path.with_suffix('.identity.json').read_text())!=cache_ident:raise ValueError('Stale E results')
        for row in rows:
            if row['cache_fingerprint']!=digest(cache_ident) or row['row_content_hash']!=digest({k:v for k,v in row.items() if k!='row_content_hash'}):raise ValueError('E row checksum mismatch')
    else:
        import torch
        backend=None;embeddings=None;started=time.perf_counter()
        with RowCache(result_path,cache_ident,args.resume) as cache, RowCache(stage_dir/'lm_calls.jsonl',cache_ident,args.resume) as calls, RowCache(stage_dir/'evidence_objects.jsonl',cache_ident,args.resume) as extractions:
            for number,record in enumerate(records,1):
                inf=record['inference'];qid=inf['question_id'];chunks=bundle['chunks'][qid]
                record=dict(record,_d_cost=next(r for r in bundle['baselines'] if r['question_id']==qid and r['policy']=='D1'))
                for policy in prompts.POLICIES:
                    key=cache.key(qid,inf['paper_id'],[digest(c['text']) for c in chunks],dict(policy=policy,budget=cfg['max_evidence_tokens']))
                    if key in cache.rows:continue
                    if backend is None:
                        print('Loading unchanged trained SARA FP16 efficient-SDPA backend.',flush=True)
                        backend=SaraBackend(cfg)
                        if backend.identity()['dtype']!='torch.float16' or backend.identity()['quantization']!='none':raise ValueError('Unexpected runtime precision')
                        embeddings=torch.load(bundle['paths']['sfr'],map_location='cpu',weights_only=False)
                        if 'embeddings' in embeddings:embeddings=embeddings['embeddings']
                    print(f"W9-E {args.stage} query {number}/{len(records)} {qid} {policy}",flush=True)
                    cache.put(key,produce(record,chunks,bundle['lineage'][qid],policy,backend,calls,extractions,embeddings,cfg,bundle['seal']))
                if number%10==0 or args.stage=='preflight':print(f'W9-E saved {number}/{len(records)} queries; rows={len(cache.rows)}; elapsed={time.perf_counter()-started:.0f}s',flush=True)
            rows=list(cache.rows.values())
    validate_rows(rows,records,bundle['chunks'],cfg)
    ids={r['inference']['question_id'] for r in records}
    baselines=[r for r in bundle['baselines'] if r['question_id'] in ids]
    summary=summarize(rows,baselines,stage_dir,complete=args.stage!='preflight',seed=cfg['seed'])
    examples=[]
    for r in rows:
        ev=r['evidence'];known={c['chunk_id']:c for c in bundle['chunks'][r['question_id']]}
        examples.append(dict(question_id=r['question_id'],policy=r['policy'],evidence=ev,
            support_verification=[dict(source_chunk_id=i['source_chunk_id'],char_start=i['char_start'],char_end=i['char_end'],canonical_source_substring=known[i['source_chunk_id']]['text'][i['char_start']:i['char_end']]) for i in ev['items']]))
    atomic_json(stage_dir/'evidence_examples.json',examples[:12])
    atomic_json(stage_dir/'manifest.json',dict(schema='grounded-e-v1',stage=args.stage,complete=True,full_fit=args.stage!='preflight',queries=len(records),rows=len(rows),identity_hash=digest(ident),primary_policy_unchanged='K_E=0',no_method_selected=True))
    if args.stage=='preflight':
        artifacts=[result_path,stage_dir/'evidence_quality.csv',stage_dir/'evidence_examples.json']
        atomic_json(stage_dir/'preflight_gate.json',dict(protocol_identity_hash=digest(ident),implementation_checks_passed=True,
            query_ids=[r['inference']['question_id'] for r in records],quality=summary['quality'],warnings=summary['warnings'],
            artifact_hashes={str(p.resolve()):file_hash(p) for p in artifacts},downstream_f1_not_used_for_tuning=True))
        print(json.dumps(dict(quality=summary['quality'],warnings=summary['warnings'])),flush=True)
    print(f'W9-E {args.stage} complete: {len(rows)} rows; {stage_dir}',flush=True)


if __name__=='__main__':main()
