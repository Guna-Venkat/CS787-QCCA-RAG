"""SemRAG-buffered-project-v1: Algorithm 1 adjacency with lossless boundaries."""
import numpy as np
from .chunking import sentences, token_windows, make_chunk
from .retrieval import unit_vectors


def semantic_chunks(document, lineage, tokenizer, encoder, *, buffer=1, threshold=.35, max_tokens=1024, overlap=128):
    if buffer < 0 or not 0 <= threshold <= 2:
        raise ValueError('Invalid semantic buffer/absolute cosine-distance threshold')
    text = document['text']
    spans = sentences(text)
    if not spans:
        return []
    windows = [text[spans[max(0, i-buffer)][0]:spans[min(len(spans)-1, i+buffer)][1]]
               for i in range(len(spans))]
    vectors = unit_vectors(encoder.encode(windows))
    distances = (1 - np.sum(vectors[:-1] * vectors[1:], axis=1)).clip(0, 2)
    breaks = [0] + [i + 1 for i, d in enumerate(distances) if d >= threshold] + [len(spans)]
    cfg = dict(representation='SEM', algorithm='SemRAG-buffered-project-v1', buffer=buffer,
               threshold=threshold, max_tokens=max_tokens, overlap=overlap)
    output = []
    for lo, hi in zip(breaks, breaks[1:]):
        a, b = spans[lo][0], spans[hi-1][1]
        for s, e in token_windows(text[a:b], tokenizer, max_tokens, overlap):
            output.append(make_chunk(document, lineage, a+s, a+e, cfg, tokenizer,
                [dict(sentence_boundary=lo+i+1, cosine_distance=float(d)) for i, d in enumerate(distances[lo:hi])]))
    return output
