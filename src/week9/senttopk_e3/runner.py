"""Manual E3-SentTopK; no extraction LM, MMR, dynamic training or old-policy reruns."""
import argparse
import json
import random
import time
import numpy as np
from ..provenance import branch_guard,digest,file_hash
from ..caching import RowCache,read_jsonl,atomic_json
from ..retrieval_cache import RetrievalCache
from ..retrieval import LocalEncoder
from ..backend import SaraBackend,CachedClient,load_tokenizer
from ..runner import canonical_context
from ..revision import revision_messages
from ..cost_accounting import QueryCost
from ..evaluation import annotate_outcome
from .protocol import config,inputs,identity,output,smoke_records,pin_protocol,GRID
from .sentences import inventory,validate,rank,select,evidence,diagnostics
from .embeddings import cached
from .analysis import analyze


class LazyEncoder:
    def __init__(self,cfg):self.cfg=cfg;self.encoder=None
    def encode(self,texts):
        if self.encoder is None:self.encoder=LocalEncoder(self.cfg['encoder'],self.cfg['encoder_device'])
        return self.encoder.encode(texts)


def prepare_query(inf,chunks,tokenizer,encoder,encoder_meta,inventories,scores,sentence_cache,query_cache):
    qid=inf['question_id'];pid=inf['paper_id'];source_hashes=[digest(c['text']) for c in chunks]
    ikey=inventories.key(qid,pid,source_hashes,'source_sentence_inventory')
    old=inventories.rows.get(ikey)
    if old is None:
        started=time.perf_counter();sents=inventory(qid,pid,chunks,tokenizer);elapsed=time.perf_counter()-started
        old=inventories.put(ikey,dict(question_id=qid,paper_id=pid,source_selected_chunk_ids=[c['chunk_id'] for c in chunks],sentences=sents,origin_segmentation_seconds=elapsed))
    sents=old['sentences'];validate(sents,chunks,pid)
    if old['source_selected_chunk_ids']!=[c['chunk_id'] for c in chunks]:raise ValueError('Inventory coarse chunk IDs changed')
    if not sents:
        return [],{},dict(sentence_index_conceptual_seconds=0.0,sentence_index_current_compute_seconds=0.0,sentence_embedding_lookup_seconds=0.0,query_embedding_conceptual_seconds=0.0,query_embedding_current_compute_seconds=0.0,query_embedding_lookup_seconds=0.0,embedding_lookup_compute_latency=0.0,sentence_ranking_latency=0.0,current_sentence_ranking_latency=0.0)
    vectors,sm=cached(sentence_cache,[s['text'] for s in sents],'sentence',encoder,encoder_meta)
    query,qm=cached(query_cache,[inf['question']],'query',encoder,encoder_meta)
    key=scores.key(qid,pid,source_hashes,dict(query_hash=digest(inf['question']),inventory_hash=digest(sents)))
    start=time.perf_counter();scored=scores.rows.get(key);lookup=time.perf_counter()-start;current_rank=0.0
    if scored is None:
        start=time.perf_counter();ranked=rank(inf['question'],sents,query[0],vectors);current_rank=time.perf_counter()-start
        scored=scores.put(key,dict(question_id=qid,paper_id=pid,question=inf['question'],query_hash=digest(inf['question']),ranked_sentences=ranked,origin_ranking_seconds=current_rank))
    ranked=scored['ranked_sentences'];validate(ranked,chunks,pid)
    if {s['sentence_id'] for s in ranked}!={s['sentence_id'] for s in sents}:raise ValueError('Scored inventory changed')
    bank={s['sentence_id']:v for s,v in zip(sents,vectors)}
    meta=dict(sentence_index_conceptual_seconds=sm['conceptual_compute_seconds'],sentence_index_current_compute_seconds=sm['current_compute_seconds'],sentence_embedding_lookup_seconds=sm['lookup_seconds'],
        query_embedding_conceptual_seconds=qm['conceptual_compute_seconds'],query_embedding_current_compute_seconds=qm['current_compute_seconds'],query_embedding_lookup_seconds=qm['lookup_seconds'],
        embedding_lookup_compute_latency=sm['lookup_seconds']+sm['current_compute_seconds']+qm['lookup_seconds']+qm['current_compute_seconds']+lookup,
        sentence_ranking_latency=scored['origin_ranking_seconds'],current_sentence_ranking_latency=current_rank,
        sentence_score_cache_lookup_seconds=lookup,origin_segmentation_seconds=old['origin_segmentation_seconds'])
    return ranked,bank,meta


def generate(record,chunks,ranked,bank,embedding_meta,k,backend,calls,embeddings,bundle):
    inf=record['inference'];selected=select(ranked,k);validate(selected,chunks,inf['paper_id'])
    block=evidence(selected);context,comp,initial_tokens=canonical_context(record,embeddings,backend.tokenizer,backend.device,backend.model.dtype)
    d1=record['_d1']
    cost=QueryCost(initial_draft_prompt_tokens=initial_tokens,initial_draft_generated_tokens=len(record['canonical'].get('saved_generated_token_ids',backend.tokenizer.encode(inf['initial_draft'],add_special_tokens=False))),
        final_prompt_tokens=initial_tokens,retrieval_query_time=d1['retrieval_query_time'],selection_latency=d1['selection_latency'],extraction_latency=0.0)
    answer_hash=digest(context);called=False
    if selected:
        client=CachedClient(backend,calls,inf,cost,'revision',context=context,retrieval_embeds=comp,evidence_ids=[s['sentence_id'] for s in selected],revision_mode='F0')
        messages=revision_messages(inf,block,False);answer_hash=digest(client.prepare_prompt(messages));called=True
        prediction=client.chat(messages,max_tokens=128)['content'].strip();cost.revision_latency=client.latency
    else:prediction=inf['initial_draft']
    n_tokens=len(backend.tokenizer.encode(block,add_special_tokens=False))
    raw_tokens=sum(len(backend.tokenizer.encode(c['text'],add_special_tokens=False)) for c in chunks)
    row=dict(question_id=inf['question_id'],paper_id=inf['paper_id'],question=inf['question'],policy=f'E3-k{k}',experiment='E3-SentTopK',sentence_k=k,coarse_evidence_k=2,
        initial_draft=inf['initial_draft'],prediction=prediction,selected_chunk_ids=[c['chunk_id'] for c in chunks],source_chunk_hashes={c['chunk_id']:digest(c['text']) for c in chunks},
        source_d_row_hashes=bundle['lineage'][inf['question_id']],source_e_row_hashes=bundle['e_lineage'][inf['question_id']],
        selected_sentence_ids=[s['sentence_id'] for s in selected],selection_rank_order=[s['sentence_id'] for s in ranked[:k]],selected_sentences=selected,
        evidence_text=block,evidence_tokens=n_tokens,raw_coarse_tokens=raw_tokens,compression_ratio=n_tokens/max(raw_tokens,1),
        source_support_validation_rate=1.0 if selected else None,fail_closed=not selected,failure_reason=None if selected else 'no_valid_sentences',
        answer_generation_called=called,final_prompt_hash=answer_hash,evidence_hash=digest(block),answer_output_hash=digest(prediction),
        model=backend.identity(),tokenizer=dict(name_or_path=backend.tokenizer.name_or_path,tokenizer_class=type(backend.tokenizer).__name__),
        encoder_identity_hash=digest(bundle['encoder']),decoding=dict(do_sample=False,max_new_tokens=128),
        cost_note='Initial saved draft is counted; index origin computation, actual cache lookup/computation, and conceptual query embedding cost are separate. E3 calls only the answer LM; never an extraction LM.',
        **diagnostics(selected,ranked,bank),**embedding_meta,**cost.to_dict())
    return annotate_outcome(row,inf,record['evaluation'])


def validate_grid(rows,records,chunks):
    expected={(r['inference']['question_id'],r['inference']['paper_id'],k) for r in records for k in GRID}
    actual=[(r['question_id'],r['paper_id'],r['sentence_k']) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('E3 full query/k coverage incomplete')
    for r in rows:
        source=chunks[r['question_id']]
        if r['selected_chunk_ids']!=[c['chunk_id'] for c in source] or r['source_chunk_hashes']!={c['chunk_id']:digest(c['text']) for c in source}:raise ValueError('E3 source identity changed')
        validate(r['selected_sentences'],source,r['paper_id'])
        selected=r['selected_sentences']
        if len(selected)!=r['actual_sentence_k'] or len(selected)>r['sentence_k'] or len({s['sentence_id'] for s in selected})!=len(selected):raise ValueError('Invalid actual k')
        if selected!=sorted(selected,key=lambda s:(s['document_char_start'],s['original_chunk_rank'],s['sentence_order'],s['sentence_id'])):raise ValueError('Source order not restored')
        if r['evidence_text']!=evidence(selected):raise ValueError('Altered sentence evidence')
        if r['fail_closed'] and (r['prediction']!=r['initial_draft'] or r['answer_generation_called'] or r['failure_reason']!='no_valid_sentences'):raise ValueError('Invalid no-sentence fallback')
        if r['extraction_prompt_tokens'] or r['extraction_generated_tokens'] or any(c['stage']!='revision' for c in r['model_calls']):raise ValueError('E3 unexpectedly used an extraction LM')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--stage',required=True,choices=('prepare','smoke','full','analyze'));parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);branch_guard();cfg=config(args.config)
    random.seed(cfg['seed']);np.random.seed(cfg['seed'])
    bundle=inputs(cfg);ident=identity(cfg,bundle);root=output(cfg)
    pin_protocol(root/'protocol.json',ident)
    if args.stage=='prepare':print('E3 inputs pinned read-only; no scoring, model loading or generations run.');return
    records=smoke_records(bundle['records'],cfg) if args.stage=='smoke' else bundle['records']
    directory=root/('smoke' if args.stage=='smoke' else 'fit');result_path=directory/'results.jsonl'
    cache_ident=dict(protocol=ident,stage='smoke' if args.stage=='smoke' else 'full')
    if args.stage=='analyze':
        if json.loads(result_path.with_suffix('.identity.json').read_text())!=cache_ident:raise ValueError('Stale E3 results')
        rows=read_jsonl(result_path)
        for r in rows:
            if r['cache_fingerprint']!=digest(cache_ident) or r['row_content_hash']!=digest({k:v for k,v in r.items() if k!='row_content_hash'}):raise ValueError('E3 row checksum mismatch')
    else:
        import torch
        tokenizer=load_tokenizer(bundle['historical_cfg']);encoder=LazyEncoder(cfg);backend=None;embeddings=None
        d1={r['question_id']:r for r in bundle['baselines'] if r['policy']=='D1'}
        cache_root=root/'sentence_cache';embedding_identity=dict(encoder=bundle['encoder'],sources=ident['sources'],config_hash=digest(cfg))
        started=time.perf_counter()
        with RowCache(result_path,cache_ident,args.resume) as results, RowCache(directory/'lm_calls.jsonl',cache_ident,args.resume) as calls, RowCache(cache_root/'inventories.jsonl',ident,True) as inventories, RowCache(cache_root/'scores.jsonl',ident,True) as scores, RetrievalCache(cache_root/'sentence_embeddings.sqlite3',embedding_identity,True) as sentence_cache, RetrievalCache(cache_root/'query_embeddings.sqlite3',embedding_identity,True) as query_cache:
            for number,record in enumerate(records,1):
                inf=record['inference'];qid=inf['question_id'];chunks=bundle['chunks'][qid]
                keys={k:results.key(qid,inf['paper_id'],[digest(c['text']) for c in chunks],dict(sentence_k=k,query_hash=digest(inf['question']))) for k in GRID}
                if all(key in results.rows for key in keys.values()):continue
                ranked,bank,embedding_meta=prepare_query(inf,chunks,tokenizer,encoder,bundle['encoder'],inventories,scores,sentence_cache,query_cache)
                record=dict(record,_d1=d1[qid])
                for k in GRID:
                    if keys[k] in results.rows:continue
                    if backend is None:
                        print('Loading unchanged trained SARA FP16 efficient-SDPA backend.',flush=True)
                        backend=SaraBackend(bundle['historical_cfg'])
                        if backend.identity()['dtype']!='torch.float16':raise ValueError('Unexpected generator dtype')
                        embeddings=torch.load(bundle['paths']['sfr'],map_location='cpu',weights_only=False)
                        if 'embeddings' in embeddings:embeddings=embeddings['embeddings']
                    print(f'E3 {args.stage}: query {number}/{len(records)} {qid}, k={k}',flush=True)
                    results.put(keys[k],generate(record,chunks,ranked,bank,embedding_meta,k,backend,calls,embeddings,bundle))
                if args.stage=='smoke' or number%10==0:print(f'E3 saved {number}/{len(records)} queries; rows={len(results.rows)}; elapsed={time.perf_counter()-started:.0f}s',flush=True)
            rows=list(results.rows.values())
            # Distinguish unique offline sentence-index computation from query
            # attribution, and physical cache contents from per-action costs.
            index_stats=dict(unique_sentence_embeddings=len(sentence_cache.rows),unique_query_embeddings=len(query_cache.rows),
                sentence_index_unique_origin_seconds=sum(r['origin_amortized_compute_seconds'] for r in sentence_cache.rows.values()),
                query_embeddings_unique_origin_seconds=sum(r['origin_amortized_compute_seconds'] for r in query_cache.rows.values()),
                note='Sentence embeddings are reusable preprocessing/index cost. Query embeddings are online conceptual cost even when cached. Current-run lookup/compute is recorded separately in each row; no cache hit erases origin computation.')
            atomic_json(directory/'embedding_costs.json',index_stats)
    validate_grid(rows,records,bundle['chunks'])
    ids={r['inference']['question_id'] for r in records};baselines=[r for r in bundle['baselines'] if r['question_id'] in ids]
    outcome=analyze(rows,baselines,cfg,ident,directory,complete=args.stage!='smoke')
    if any(file_hash(p)!=h for p,h in ident['input_hashes'].items()):raise ValueError('Historical D/E artifact changed; E3 will not reinterpret it')
    atomic_json(directory/'manifest.json',dict(experiment='E3-SentTopK-v1',stage=args.stage,queries=len(records),rows=len(rows),complete=True,full_fit=args.stage!='smoke',protocol_hash=digest(ident),old_artifacts_unchanged=True,extraction_lm_calls=0,dynamic_controller_run=False))
    print(f'E3 {args.stage} complete: {len(rows)} rows; {directory}; full FIT={args.stage!="smoke"}',flush=True)
    if outcome:print(json.dumps(outcome),flush=True)


if __name__=='__main__':main()
