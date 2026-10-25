"""Lossless indexed embedding caches; reused computation never becomes zero cost."""
import time
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download
from ..provenance import digest,file_hash
from ..retrieval import unit_vectors


def encoder_identity(model_id,frozen_hashes):
    if model_id!='sentence-transformers/all-MiniLM-L6-v2':raise ValueError('Use frozen canonical MiniLM')
    snapshot=Path(snapshot_download(model_id,local_files_only=True))
    paths=[p for p in snapshot.rglob('*') if p.is_file() and p.suffix in ('.safetensors','.bin','.json','.txt')]
    hashes={str(p):file_hash(p) for p in paths}
    if not hashes or any(frozen_hashes.get(p)!=h for p,h in hashes.items()):raise ValueError('Canonical encoder file/revision changed')
    return dict(model_id=model_id,revision=snapshot.name,file_hashes=hashes,
        normalization='LocalEncoder nonoverlapping token-window mean, then float64 L2 normalization, floor 1e-12',
        text_preprocessing='exact canonical string; no truncation, no draft')


def embedding_key(cache,text,kind,encoder_meta):
    return cache.key('TEXT',kind,[digest(text)],dict(encoder=encoder_meta,normalization=encoder_meta['normalization']))


def cached(cache,texts,kind,encoder,encoder_meta):
    started=time.perf_counter();keys=[embedding_key(cache,t,kind,encoder_meta) for t in texts]
    values={};missing={}
    for key,text in zip(keys,texts):
        row=cache.rows.get(key)
        if row is None:missing.setdefault(key,text)
        else:values[key]=row
    lookup=time.perf_counter()-started;compute=0.0
    if missing:
        start=time.perf_counter();vectors=unit_vectors(encoder.encode(list(missing.values())));compute=time.perf_counter()-start
        for (key,text),vector in zip(missing.items(),vectors):
            values[key]=cache.put(key,dict(text_hash=digest(text),kind=kind,vector=vector.tolist(),
                encoder_identity_hash=digest(encoder_meta),origin_amortized_compute_seconds=compute/len(missing),
                timing_basis='Amortized actual batch computation; conceptually retained on cache hits'))
    vectors=np.asarray([values[k]['vector'] for k in keys])
    for key,text in zip(keys,texts):
        if values[key]['text_hash']!=digest(text) or values[key]['encoder_identity_hash']!=digest(encoder_meta):raise ValueError('Embedding identity mismatch')
    meta=dict(lookup_seconds=lookup,current_compute_seconds=compute,cache_new_entries=len(missing),
        conceptual_compute_seconds=sum(values[k]['origin_amortized_compute_seconds'] for k in keys),
        cache_hits=len(keys)-len(missing))
    return vectors,meta
