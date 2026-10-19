"""Evaluation-only paragraph alignment, with explicit coarse-density limitations."""
import math
import re
import numpy as np
from .faithfulness import norm


def alignment(chunk, evidence):
    text = norm(chunk['text'])
    touched = []
    fractions = []
    for i, paragraph in enumerate(evidence):
        paragraph = norm(re.sub(r'BIBREF\d+', '', paragraph))
        # Contiguous exact token overlap is a conservative paragraph proxy.
        words, gold = text.split(), paragraph.split()
        from difflib import SequenceMatcher
        blocks = SequenceMatcher(None, words, gold, autojunk=False).get_matching_blocks()
        count = sum(b.size for b in blocks)
        fraction = count / max(len(gold), 1)
        if count >= min(5, len(gold)) and (paragraph in text or text in paragraph or fraction >= .5):
            touched.append(i)
        fractions.append(count)
    return set(touched), min(len(text.split()), max(fractions, default=0))


def retrieval_metrics(ranked, all_chunks, evidence):
    n = len(evidence)
    labels = [alignment(c, evidence)[0] for c in ranked]
    retrieved = set().union(*labels) if labels else set()
    # Evaluate only answer annotations with evidence; missing annotation != retrieval success.
    any_recall = float(bool(retrieved)) if n else None
    all_recall = float(len(retrieved) == n) if n else None
    relevant_universe = sum(bool(alignment(c, evidence)[0]) for c in all_chunks)
    gains = [int(bool(x)) for x in labels]
    dcg = sum(g/math.log2(i+2) for i, g in enumerate(gains))
    ideal = sum(1/math.log2(i+2) for i in range(min(relevant_universe, len(ranked))))
    overlap = sum(alignment(c, evidence)[1] for c in ranked)
    word_count = sum(len(c['text'].split()) for c in ranked)
    lengths = [c['token_length'] for c in all_chunks]
    return dict(any_gold_evidence_recall=any_recall, all_gold_evidence_recall=all_recall,
        mrr=next((1/(i+1) for i, g in enumerate(gains) if g), 0.) if n else None,
        ndcg=dcg/ideal if ideal else None, annotated_evidence_count=n,
        mean_chunk_tokens=float(np.mean(lengths)) if lengths else 0.,
        median_chunk_tokens=float(np.median(lengths)) if lengths else 0.,
        retrieved_context_tokens=sum(c['token_length'] for c in ranked),
        evidence_overlap_density=overlap/word_count if word_count else 0.,
        evidence_density_note='Coarse paragraph/exact word overlap proxy, not token-level gold attribution; overlapping chunks counted separately.')
