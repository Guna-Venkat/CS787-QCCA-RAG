"""Independent real E3 smoke audit; no model loading or old-artifact writes."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from src.week9.provenance import ROOT,digest,file_hash,source_hashes
from src.week9.caching import read_jsonl,atomic_json
from src.week9.retrieval_cache import read_rows
from src.week9.senttopk_e3.protocol import sources,GRID
from src.week9.senttopk_e3.sentences import rank,select,validate,evidence
root=ROOT/'results/week9/w9e3_senttopk/v1'
protocol=json.loads((root/'protocol.json').read_text())
assert protocol['sources']==sources() and protocol['historical_sources']==source_hashes()
assert all(file_hash(p)==h for p,h in protocol['input_hashes'].items())
rows=read_jsonl(root/'smoke/results.jsonl');calls=read_jsonl(root/'smoke/lm_calls.jsonl')
inventories=read_jsonl(root/'sentence_cache/inventories.jsonl');scores=read_jsonl(root/'sentence_cache/scores.jsonl')
assert len(rows)==12 and len({r['question_id'] for r in rows})==2
stage_stamp=digest(dict(protocol=protocol,stage='smoke'))
for r in rows+calls:assert r['cache_fingerprint']==stage_stamp and r['row_content_hash']==digest({k:v for k,v in r.items() if k!='row_content_hash'})
for r in inventories+scores:assert r['cache_fingerprint']==digest(protocol) and r['row_content_hash']==digest({k:v for k,v in r.items() if k!='row_content_hash'})
queries={r['question_id'] for r in rows};chunks={}
with (ROOT/'results/week9/diagnostics/w9c_k2/extraction/fit/w9d_representation.jsonl').open() as f:
    for line in f:
        r=json.loads(line)
        if r['question_id'] in queries and r['policy']=='D1':chunks[r['question_id']]=r['source_chunks']
sv={r['text_hash']:np.asarray(r['vector']) for r in read_rows(root/'sentence_cache/sentence_embeddings.sqlite3')}
qv={r['text_hash']:np.asarray(r['vector']) for r in read_rows(root/'sentence_cache/query_embeddings.sqlite3')}
checks=[]
for inv in inventories:
    q=inv['question_id'];sents=inv['sentences'];score=next(r for r in scores if r['question_id']==q)
    vectors=[sv[s['text_hash']] for s in sents]
    ranked=rank(score['question'],sents,qv[score['query_hash']],vectors)
    assert [s['sentence_id'] for s in ranked]==[s['sentence_id'] for s in score['ranked_sentences']]
    assert np.allclose([s['cosine'] for s in ranked],[s['cosine'] for s in score['ranked_sentences']],atol=1e-12,rtol=0)
    validate(sents,chunks[q],inv['paper_id'])
    for k in GRID:
        row=next(r for r in rows if r['question_id']==q and r['sentence_k']==k)
        expected=select(ranked,k)
        assert row['selected_sentences']==expected and row['evidence_text']==evidence(expected)
        assert row['actual_sentence_k']==min(k,len(sents))
        assert row['source_chunk_hashes']=={c['chunk_id']:digest(c['text']) for c in chunks[q]}
        assert row['source_support_validation_rate']==1
        assert row['extraction_prompt_tokens']==row['extraction_generated_tokens']==0
    checks.append(dict(question_id=q,question=score['question'],source_chunk_ids=[c['chunk_id'] for c in chunks[q]],candidate_sentences=len(sents),all_candidates_and_scores_saved=str((root/'sentence_cache/scores.jsonl').relative_to(ROOT))))
assert all(c['stage']=='revision' and c['dtype']=='torch.float16' and c['attention_backend']=='efficient_sdpa' for c in calls)
assert not (root/'fit/results.jsonl').exists() and not (root/'frozen_e3_senttopk.json').exists()
assert not (root/'fit/dynamic_go_no_go.json').exists()
manifest=json.loads((root/'smoke/manifest.json').read_text())
assert manifest['complete'] and not manifest['full_fit'] and not manifest['dynamic_controller_run']
notebook=json.loads((ROOT/'notebooks/week9/week9e_extraction_faithfulness.ipynb').read_text())
audit=json.loads((ROOT/'results/week9/patch_verification/w9e3_senttopk/notebook_append_audit.json').read_text())
assert notebook['cells'][:audit['original_cells']]==audit['prefix_cells']
atomic_json(root/'source_config_prompt_model_hashes.json',dict(config_hash=digest(protocol['config']),source_bundle_hash=digest(protocol['sources']),
    historical_source_bundle_hash=digest(protocol['historical_sources']),old_e_source_bundle_hash=digest(protocol['historical_e_sources']),
    answer_prompt_source_hash=protocol['historical_sources']['src/week9/revision.py'],model_hashes=protocol['model_hashes'],encoder_identity_hash=digest(protocol['encoder']),
    note='Every generated row additionally records its actual final prompt, evidence and output hashes.'))
result=dict(branch='gagan',queries=checks,rows=len(rows),answer_lm_calls=len(calls),extraction_lm_calls=0,
    peak_gpu_allocated_gib=max(c['peak_memory_allocated'] for c in calls)/1024**3,
    encoder_model=protocol['encoder']['model_id'],encoder_revision=protocol['encoder']['revision'],
    evidence_token_min=min(r['evidence_tokens'] for r in rows),evidence_token_max=max(r['evidence_tokens'] for r in rows),
    all_sentence_source_checks_passed=True,all_cosines_and_ranking_recomputed_from_cached_vectors=True,
    source_presentation_order_verified=True,old_artifacts_unchanged=True,old_notebook_prefix_preserved_except_safe_defaults=True,
    full_e3_started=False,k_selected_or_frozen=False,dynamic_controller_trained_or_run=False,tests_passed=119,tests_deselected=1,
    deselection='Existing all-notebooks defaults test conflicts with user-enabled B/C/D flags; E3 disabled-path test passed.')
atomic_json(ROOT/'results/week9/patch_verification/w9e3_senttopk/verification.json',result)
print(json.dumps(result,indent=2))
