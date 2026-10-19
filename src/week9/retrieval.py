"""Paper-local BM25, dense cosine and rank-preserving RRF."""
from collections import Counter
import re
import numpy as np
from .provenance import assert_inference_clean


def unit_vectors(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite embedding matrix')
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def bm25(query, texts, k1=1.5, b=.75):
    docs = [re.findall(r'\w+', t.lower()) for t in texts]
    n = len(docs)
    if not n:
        return np.array([])
    df = Counter(w for d in docs for w in set(d))
    average = max(np.mean([len(d) for d in docs]), 1)
    scores = []
    for words in docs:
        tf = Counter(words)
        scores.append(sum(np.log(1 + (n-df[w]+.5)/(df[w]+.5)) *
            tf[w]*(k1+1)/(tf[w]+k1*(1-b+b*len(words)/average)) for w in re.findall(r'\w+', query.lower())))
    return np.array(scores)


class LocalEncoder:
    def __init__(self, name='sentence-transformers/all-MiniLM-L6-v2', device='cpu'):
        from sentence_transformers import SentenceTransformer
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(name, local_files_only=True)
        self.model = SentenceTransformer(local_path, device=device, local_files_only=True)
        self.tokenizer = self.model.tokenizer
        self.max_input_tokens = self.model.max_seq_length

    def encode(self, texts):
        # Explicit long-text mean pooling over non-overlapping encoder windows.
        # Avoid silently embedding only the first 256 tokens of a 1024-token chunk.
        from .chunking import token_windows
        windows, groups = [], []
        for text in texts:
            spans = token_windows(text, self.tokenizer, self.max_input_tokens-2, 0)
            values = [text[a:b] for a, b in spans] or ['']
            groups.append((len(windows), len(windows)+len(values)))
            windows.extend(values)
        if not windows:
            return np.empty((0, self.model.get_sentence_embedding_dimension()))
        vectors = self.model.encode(windows, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
        return unit_vectors([np.mean(vectors[a:b], axis=0) for a,b in groups])


def rank_candidates(query, paper_id, chunks, encoder, method, width, vectors=None):
    assert_inference_clean(chunks)
    if width < 1 or method not in ('BM25', 'SEMANTIC', 'RRF'):
        raise ValueError('Invalid retrieval method/width')
    if any(c['paper_id'] != paper_id for c in chunks):
        raise ValueError('Cross-paper retrieval forbidden')
    ids = [c['chunk_id'] for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate chunks')
    if not chunks:
        return []
    sparse = bm25(query, [c['text'] for c in chunks])
    dense = None
    if method != 'BM25':
        vectors = unit_vectors(encoder.encode([c['text'] for c in chunks]) if vectors is None else vectors)
        dense = vectors @ unit_vectors(encoder.encode([query]))[0]
    def ranks(scores):
        order = sorted(range(len(chunks)), key=lambda i: (-scores[i], ids[i]))
        return {i: r+1 for r, i in enumerate(order)}
    br = ranks(sparse)
    dr = ranks(dense) if dense is not None else {}
    score = sparse if method == 'BM25' else dense if method == 'SEMANTIC' else np.array([1/(60+br[i])+1/(60+dr[i]) for i in range(len(chunks))])
    order = sorted(range(len(chunks)), key=lambda i: (-score[i], ids[i]))[:width]
    return [{**chunks[i], 'retrieval_rank': r+1, 'retrieval_score': float(score[i]),
        'bm25_rank': br[i], 'dense_rank': dr.get(i), 'bm25_score': float(sparse[i]),
        'dense_score': float(dense[i]) if dense is not None else None} for r, i in enumerate(order)]
