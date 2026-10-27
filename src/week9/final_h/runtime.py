"""Reuse frozen E3/F/G inference; callable only from an authorized H2 producer.

The caller supplies a real canonical TEST record and source candidate pool.
This module does not fabricate first-pass/SFR inputs or adapt the W8 baseline.
"""
import time
import joblib
from . import protocol as p
from ..selection import select as coarse_select
from ..senttopk_e3.runner import prepare_query, generate as e3_generate
from ..utilization_f.runner import generate as f_generate
from ..action_verification_g.features import payload, online_features
from ..provenance import digest


def verify_action(d0, f0, src, model_root=None):
    """The gate sees inference fields only and never substitutes chosen-action costs."""
    import pandas as pd
    model_root = model_root or p.G_ROOT/'final_fit'
    schema = p.read(model_root/'feature_schema.json')
    threshold = p.read(model_root/'threshold.json')['threshold']
    p.require(schema['feature_count'] == 32 and threshold == .70, 'Frozen G2 changed')
    model = joblib.load(model_root/'logistic_pipeline.joblib')
    started = time.perf_counter()
    features = online_features(payload(d0, f0, src))
    x = pd.DataFrame([features])[schema['feature_columns']]
    probability = float(model.predict_proba(x)[0, 1])
    accept = probability >= .70
    row = dict(f0, system=p.VERIFY, policy=p.VERIFY,
        prediction=f0['prediction'] if accept else d0['prediction'], g2_accept=accept,
        g2_probability=probability, g2_threshold=.70, verifier_latency=time.perf_counter()-started,
        measured_end_to_end_latency=None, cost_note='Post-generation gate: all F0 LM costs retained even KEEP')
    # Scores copied from F0 are stale when KEEP; the H2 common scorer recomputes them.
    for key in ('f1', 'exact_match', 'delta_f1', 'help', 'hurt', 'answer_changed'):
        row.pop(key, None)
    return row


def generate_w9(root, record, pool, backend, encoder, embeddings, calls, inventories, scores, sentence_cache, query_cache):
    """Only the already frozen E3-k3/F0 and F2-A, with post-generation G2."""
    freeze = p.validate_freeze(root)
    p.require((__import__('pathlib').Path(root)/'h2_claim.json').is_file(), 'Inference requires explicit claimed H2')
    inf = record['inference']
    started = time.perf_counter()
    chunks = coarse_select(inf, pool['candidates'], 2, 'DraftSemantic', encoder)
    selection_time = time.perf_counter()-started
    # Exact inference-only source identities; never substitute development chunk IDs.
    source_identity = dict(question_id=inf['question_id'], paper_id=inf['paper_id'],
        chunk_ids=[c['chunk_id'] for c in chunks], chunk_hashes=[digest(c['text']) for c in chunks])
    ranked, bank, meta = prepare_query(inf, chunks, backend.tokenizer, encoder, freeze['encoder'],
                                     inventories, scores, sentence_cache, query_cache)
    new_record = dict(record, _d1=dict(retrieval_query_time=pool['retrieval_query_time'], selection_latency=selection_time))
    q = inf['question_id']
    bundle = dict(lineage={q: dict(final_test_source_identity=source_identity)},
                  e_lineage={q: {}}, encoder=freeze['encoder'])
    f0 = e3_generate(new_record, chunks, ranked, bank, meta, 3, backend, calls, embeddings, bundle)
    f0.update(system=p.PRIMARY, policy='F0', split='test')
    from ..utilization_f.prompts import definitions
    f0['prompt_template_hash'] = definitions()['F0']['prompt_hash']
    f0['row_content_hash'] = digest(f0)
    safe = f_generate(new_record, f0, 'F2-A', backend, calls, dict(embeddings=embeddings))
    safe.update(system=p.SAFE, split='test')
    d0 = dict(prediction=inf['initial_draft'])
    verified = verify_action(d0, f0, f0)
    return [f0, safe, verified]
