"""Manual W9-F only: immutable E3-k3, F0 alias, three new revision calls/query."""
import argparse
import json
import random
import time
import re
import numpy as np
from ..provenance import branch_guard,digest
from ..data import resolve
from ..caching import RowCache,read_jsonl,atomic_json
from ..backend import SaraBackend,CachedClient
from ..runner import canonical_context
from ..cost_accounting import QueryCost
from ..evaluation import annotate_outcome
from .prompts import POLICIES,messages,definitions
from .protocol import config,inputs,identity,pin,unchanged,smoke_records,verify_checksums
from .analysis import analyze,validate_rows,lexical

INHERITED=('sentence_index_conceptual_seconds','query_embedding_conceptual_seconds',
           'embedding_lookup_compute_latency','sentence_ranking_latency')


def invalid_answer(answer):
    if not answer.strip():return 'empty_answer_generation'
    # Never repair/expose wrappers, reasoning blocks or leaked instruction envelopes.
    if re.search(r'\[/?INST\]|</?think>|<\|(?:system|user|assistant)\|>|```|(?:^|\n)\s*(?:Reasoning|Chain.of.thought)\s*:',answer,re.I):
        return 'invalid_answer_wrapper'
    return None


def source_fields(src):
    return dict(question=src['question'],evidence_text=src['evidence_text'],
        evidence_hash=src['evidence_hash'],evidence_tokens=src['evidence_tokens'],
        selected_chunk_ids=src['selected_chunk_ids'],source_chunk_hashes=src['source_chunk_hashes'],
        selected_sentence_ids=src['selected_sentence_ids'],
        sentence_text_hashes=[s['text_hash'] for s in src['selected_sentences']],
        selected_sentences=src['selected_sentences'],
        source_e3_row_hash=src['row_content_hash'],sentence_k=3,
        source_support_validation_rate=src['source_support_validation_rate'],
        **{'inherited_'+k:src.get(k) for k in INHERITED})


def alias(src,policy,bundle):
    if policy=='F0':
        row={k:v for k,v in src.items() if k not in ('cache_key','cache_fingerprint','row_content_hash')}
        row.update(source_fields(src),policy='F0',f0_alias=True,answer_fail_closed=False,
                   runtime_source='Saved equivalent E3-k3 result; no new F0 generation.',new_answer_calls=0)
    elif policy=='D0':
        row=dict(bundle['d0'][src['question_id']],question=src['question'],evidence_text='',
            evidence_tokens=0,evidence_hash=digest(''),initial_draft=src['initial_draft'],
            source_e3_row_hash=src['row_content_hash'],answer_generation_called=False,
            runtime_source='Saved original first-pass answer; SKIP.',new_answer_calls=0,
            answer_fail_closed=False,fail_closed=False)
    else:raise ValueError('Only D0/F0 can be reused')
    row.update(experiment='W9-F-E3k3-utilization-v1',
        prompt_template_hash=definitions()['F0']['prompt_hash'] if policy=='F0' else None)
    return row


def generate(record,src,policy,backend,calls,bundle):
    if policy not in ('F1','F2-A','F2-B'):raise ValueError('F0 must use verified saved alias')
    inf=record['inference']
    context,comp,nt=canonical_context(record,bundle['embeddings'],backend.tokenizer,backend.device,backend.model.dtype)
    cost=QueryCost(initial_draft_prompt_tokens=nt,
        initial_draft_generated_tokens=src['initial_draft_generated_tokens'],
        final_prompt_tokens=nt,retrieval_query_time=src['retrieval_query_time'],
        selection_latency=src['selection_latency'],extraction_latency=0.0)
    client=CachedClient(backend,calls,inf,cost,'revision',context=context,retrieval_embeds=comp,
                        evidence_ids=src['selected_sentence_ids'],revision_mode=policy)
    prompt_messages=messages(inf,src['evidence_text'],policy)
    prompt=client.prepare_prompt(prompt_messages)
    before_calls=len(calls.rows)
    result=client.chat(prompt_messages,max_tokens=128)
    new_calls=len(calls.rows)-before_calls
    raw=result['content'].strip();failure=invalid_answer(raw)
    prediction=inf['initial_draft'] if failure else raw
    cost.revision_latency=client.latency
    if len(backend.tokenizer.encode(src['evidence_text'],add_special_tokens=False))!=src['evidence_tokens']:
        raise ValueError('Evidence tokenizer count changed')
    row=dict(question_id=inf['question_id'],paper_id=inf['paper_id'],policy=policy,
        experiment='W9-F-E3k3-utilization-v1',initial_draft=inf['initial_draft'],
        prediction=prediction,raw_generation=raw,**source_fields(src),
        answer_generation_called=True,answer_fail_closed=bool(failure),fail_closed=bool(failure),
        failure_reason=failure,final_output_source='original_draft' if failure else 'new_generation',
        final_prompt_hash=digest(prompt),prompt_template_hash=definitions()[policy]['prompt_hash'],
        answer_output_hash=digest(prediction),model=backend.identity(),
        tokenizer=src['tokenizer'],decoding=dict(do_sample=False,max_new_tokens=128),
        runtime_source='New revision policy; stage timing may be reused on cache resume.',
        new_answer_calls=new_calls,**cost.to_dict())
    row.update(lexical(prediction,inf['initial_draft'],src['evidence_text']))
    return annotate_outcome(row,inf,record['evaluation'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=('prepare','smoke','full','analyze'),required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);branch_guard();cfg=config(args.config)
    random.seed(cfg['seed']);np.random.seed(cfg['seed'])
    bundle=inputs(cfg);ident=identity(cfg,bundle);root=resolve(cfg['output_root'])
    pin(root/'protocol.json',ident);pin(root/'config_snapshot.json',cfg)
    pin(root/'prompts.json',definitions());pin(root/'f0_alias_verification.json',dict(
        equivalent=True,queries=len(bundle['f0_proofs']),proofs=bundle['f0_proofs'],
        upstream_protocol_hash=ident['upstream_protocol_hash'],model=ident['model'],
        checks=['source/config/model code unchanged','checkpoint/tokenizer identities',
                'rendered final prompt hash','sentence/evidence hashes','greedy decoding/128 cap',
                'matching saved LM call/output','canonical latent context and initial token count']))
    pin(root/'source_evidence_identities.json',bundle['evidence_identities'])
    if args.stage=='prepare':
        unchanged(bundle)
        print(f'W9-F protocol pinned; {len(bundle["f0_proofs"])} F0 aliases verified. No generator loaded.');return
    records=smoke_records(bundle['records'],cfg) if args.stage=='smoke' else bundle['records']
    directory=root/('smoke' if args.stage=='smoke' else 'fit')
    result_path=directory/'results.jsonl'
    cache_ident=dict(protocol=ident,stage='smoke' if args.stage=='smoke' else 'full')
    backend=None;started=time.perf_counter();physical_new=0
    if args.stage=='analyze':
        if json.loads(result_path.with_suffix('.identity.json').read_text())!=cache_ident:raise ValueError('Stale W9-F results')
        rows=read_jsonl(result_path);verify_checksums(rows,cache_ident)
    else:
        with RowCache(result_path,cache_ident,args.resume) as results,RowCache(directory/'lm_calls.jsonl',cache_ident,args.resume) as calls:
            for i,record in enumerate(records,1):
                inf=record['inference'];q=inf['question_id'];src=bundle['e3_rows'][q]
                for policy in ('D0',)+POLICIES:
                    key=results.key(q,inf['paper_id'],src['selected_sentence_ids'],dict(policy=policy,evidence_hash=src['evidence_hash']))
                    if key in results.rows:continue
                    if policy in ('D0','F0'):row=alias(src,policy,bundle)
                    else:
                        if backend is None:
                            print('Loading unchanged SARA for revision only. No extraction LM/encoder/controller.',flush=True)
                            backend=SaraBackend(bundle['historical_cfg'])
                            if backend.identity()!=bundle['model_identity']:raise ValueError('Generator identity differs from E3')
                        n_before=len(calls.rows)
                        print(f'W9-F {args.stage}: {i}/{len(records)} query {q}, {policy}',flush=True)
                        row=generate(record,src,policy,backend,calls,bundle)
                        physical_new+=len(calls.rows)-n_before
                    results.put(key,row)
                if args.stage=='smoke' or i%10==0:
                    print(f'W9-F saved {i}/{len(records)} queries; rows={len(results.rows)}; elapsed={time.perf_counter()-started:.0f}s',flush=True)
            rows=list(results.rows.values())
    validate_rows(rows,records,bundle)
    outcome=analyze(rows,cfg,ident,directory,complete=args.stage!='smoke')
    unchanged(bundle)
    atomic_json(directory/'manifest.json',dict(experiment=cfg['experiment'],stage=args.stage,split='fit',
        complete=True,full_fit=args.stage!='smoke',queries=len(records),rows=len(rows),
        f0_reused=True,protocol_hash=digest(ident),source_e3_protocol_hash=ident['upstream_protocol_hash'],
        old_artifacts_unchanged=True,extraction_lm_calls=0,controller_trained_or_run=False,
        physical_new_answer_calls_this_invocation=physical_new,
        physical_call_note='Zero on analysis/resume hits; per-row conceptual policy costs retain original stage costs.',
        measured_end_to_end_latency=None))
    print(f'W9-F {args.stage} complete: {len(rows)} rows. {directory}',flush=True)
    if outcome:print(json.dumps(outcome),flush=True)


if __name__=='__main__':main()
