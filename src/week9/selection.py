"""Outcome-independent selectors with explicit variable candidate identities."""
import numpy as np
from .provenance import assert_inference_clean
from .retrieval import bm25, unit_vectors
from .sane_adapter import select_synopses

SELECTORS = ('BM25', 'QuerySemantic', 'DraftSemantic', 'QueryDraft', 'SANE-adapted')


def select(inference, candidates, evidence_k, method, encoder, client=None, alpha=.5):
    assert_inference_clean(inference)
    assert_inference_clean(candidates)
    if evidence_k < 0 or method not in SELECTORS:
        raise ValueError('Invalid selector or evidence_k')
    if any(c['paper_id'] != inference['paper_id'] for c in candidates):
        raise ValueError('Cross-paper selection')
    if len({c['chunk_id'] for c in candidates}) != len(candidates):
        raise ValueError('Duplicate candidate identities')
    if evidence_k == 0:
        return []
    if method == 'SANE-adapted':
        if client is None:
            raise ValueError('SANE selection requires a real LM backend')
        return select_synopses(inference['question'], candidates, evidence_k, client)[0]
    texts = [c['text'] for c in candidates]
    if not texts:
        return []
    if method == 'BM25':
        scores = bm25(inference['question'], texts)
    else:
        vectors = unit_vectors(encoder.encode(texts))
        q, d = unit_vectors(encoder.encode([inference['question'], inference['initial_draft']]))
        query_scores, draft_scores = vectors @ q, vectors @ d
        scores = query_scores if method == 'QuerySemantic' else draft_scores if method == 'DraftSemantic' else alpha*query_scores+(1-alpha)*draft_scores
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]['chunk_id']))
    return [{**candidates[i], 'selection_score': float(scores[i])} for i in order[:evidence_k]]


def validate_selection(candidates, selected, evidence_k):
    known = {c['chunk_id'] for c in candidates}
    ids = [c['chunk_id'] for c in selected]
    if len(ids) > min(evidence_k, len(candidates)) or len(set(ids)) != len(ids) or not set(ids) <= known:
        raise ValueError('Invalid selected chunk identities/budget')
