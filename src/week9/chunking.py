"""Offset-preserving chunk records and token windows."""
from __future__ import annotations
import re
from .provenance import digest


def sentences(text):
    # Preserve all whitespace and punctuation; spans tile the source exactly.
    starts = [0] + [m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=\S)', text)]
    return [(a, b) for a, b in zip(starts, starts[1:] + [len(text)]) if b > a]


def normalized_with_offsets(text):
    tokens = list(re.finditer(r'\S+', text))
    return ' '.join(m.group() for m in tokens), tokens


def locate_passages(document, passages):
    norm, tokens = normalized_with_offsets(document['text'])
    boundaries = []
    cursor = 0
    for token in tokens:
        boundaries.append((cursor, cursor + len(token.group()), token.start(), token.end()))
        cursor += len(token.group()) + 1
    result = []
    candidates = {}
    for i, text in enumerate(passages):
        wanted = ' '.join(text.split())
        if not wanted:
            continue
        start = norm.find(wanted)
        if start < 0:
            raise ValueError(f'Cannot align original passage {i} in paper {document["paper_id"]}')
        options = []
        while start >= 0:
            end = start + len(wanted)
            a = next(v[2] + start - v[0] for v in boundaries if v[0] <= start < v[1])
            b = next(v[2] + end - v[0] for v in boundaries if v[0] < end <= v[1])
            options.append((a,b))
            start = norm.find(wanted,start+1)
        candidates[i] = options
    # Original SARA passages have zero overlap. A short passage may occur inside
    # another, longer passage elsewhere (e.g. abstract repeated in introduction).
    # Resolve using the full passage partition, never a first-occurrence guess.
    resolved = {}
    while candidates:
        forced = {i: opts[0] for i,opts in candidates.items() if len(opts)==1}
        if not forced:
            raise ValueError(f'Ambiguous original passage {min(candidates)} lineage')
        resolved.update(forced)
        for i in forced:
            del candidates[i]
        for i,options in candidates.items():
            candidates[i] = [(a,b) for a,b in options if not any(a<e and b>s for s,e in forced.values())]
            if not candidates[i]:
                raise ValueError(f'Overlapping/ambiguous original passage {i} lineage')
    for i,(a,b) in sorted(resolved.items()):
        result.append(dict(original_passage_index=i, char_start=a, char_end=b,
                           passage_id=f'{document["paper_id"]}:orig:{i}'))
    return result


def token_windows(text, tokenizer, limit, overlap=0):
    if limit < 1 or not 0 <= overlap < limit:
        raise ValueError('Require positive token limit and 0 <= overlap < limit')
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False)['offset_mapping']
    offsets = [(a, b) for a, b in offsets if b > a]
    if not offsets:
        return [(0, len(text))] if text else []
    result = []
    start = 0
    while start < len(offsets):
        stop = min(start + limit, len(offsets))
        a = 0 if start == 0 else offsets[start][0]
        # Extend to next token start to preserve intervening whitespace.
        b = len(text) if stop == len(offsets) else offsets[stop][0]
        result.append((a, b))
        if stop == len(offsets):
            break
        start = stop - overlap
    return result


def make_chunk(document, lineage, a, b, cfg, tokenizer, scores=None):
    text = document['text'][a:b]
    sents = sentences(document['text'])
    touched = [p for p in lineage if p['char_start'] < b and p['char_end'] > a]
    indices = [i for i, (s, e) in enumerate(sents) if s < b and e > a]
    chunk_id = digest([document['paper_id'], document['source_sha256'], a, b, cfg])[:24]
    return dict(chunk_id=chunk_id, paper_id=document['paper_id'], source_document_id=document['source_document_id'],
        text=text, char_start=a, char_end=b, sentence_start=min(indices), sentence_end=max(indices) + 1,
        source_section=[s['section'] for s in document['sections'] if s['char_start'] < b and s['char_end'] > a],
        original_passage_index=[p['original_passage_index'] for p in touched],
        original_passage_ids=[p['passage_id'] for p in touched],
        token_length=len(tokenizer.encode(text, add_special_tokens=False, verbose=False)), chunker_config=cfg,
        boundary_scores=scores or [], source_sha256=document['source_sha256'])


def original_chunks(document, lineage, tokenizer):
    return [make_chunk(document, lineage, p['char_start'], p['char_end'], {'representation': 'ORIG'}, tokenizer)
            for p in lineage]


def fixed_chunks(document, lineage, tokenizer, size, overlap=0):
    cfg = dict(representation='FIXED', token_size=int(size), overlap=int(overlap), stride=int(size-overlap))
    return [make_chunk(document, lineage, a, b, cfg, tokenizer)
            for a, b in token_windows(document['text'], tokenizer, size, overlap)]
