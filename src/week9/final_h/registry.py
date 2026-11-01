"""Read historical non-TEST provenance; never relabel it as confirmation data."""
import re
import csv
from pathlib import Path
from functools import lru_cache
from .protocol import ROOT, read, require, E3_ROOT, F_ROOT, G_ROOT, PRIMARY, SAFE, VERIFY
from ..provenance import file_hash as _file_hash, digest

file_hash = lru_cache(maxsize=None)(_file_hash)


FIELDS = ['baseline_id', 'family', 'source_artifact', 'split', 'query_ids', 'paper_ids',
          'retriever', 'generator', 'fidelity_representation', 'budget', 'revision_mode',
          'prompt_hash', 'model_id', 'model_revision', 'final_prompt_tokens',
          'cumulative_query_prompt_tokens', 'generated_tokens', 'lm_calls',
          'latency_semantics', 'artifact_hash', 'provenance_status']


def inventory():
    entries = []
    imported = read(ROOT/'results/week9/baselines/w8_import_manifest.json')['files']
    for v in imported.values():
        p = ROOT/v['imported_path']
        require(p.is_file() and file_hash(p) == v['sha256'], 'Historical W8 import checksum differs')
    for split in ('internal_val', 'dev'):
        path = ROOT/f'results/week9/baselines/{split}/baseline_manifest.json'
        manifest = read(path)
        for old in manifest['entries']:
            source = old['source_artifact']
            original = str(source).removeprefix(str(ROOT)+'/')
            if original in imported:
                source = str(ROOT/imported[original]['imported_path'])
            p = ROOT/source if not str(source).startswith('/') else Path(source)
            valid = p.is_file() and (old.get('artifact_hash') is None or file_hash(p) == old['artifact_hash'])
            budget = re.search(r'_k(\d+)$', old['baseline_id'])
            e = dict(baseline_id=old['baseline_id'], family=old['family'], source_artifact=str(p),
                split=old['split'], query_ids=old.get('query_ids'), paper_ids=old.get('paper_ids'),
                retriever=old.get('retriever'), generator=old.get('generator'),
                fidelity_representation=old.get('fidelity_policy'), budget=int(budget[1]) if budget else None,
                revision_mode=old.get('generation_mode', old.get('revision_mode')),
                prompt_hash=old.get('prompt_version'), model_id=None, model_revision=None,
                final_prompt_tokens=old.get('prompt_tokens'), cumulative_query_prompt_tokens=old.get('cumulative_tokens'),
                generated_tokens=old.get('generated_tokens'), lm_calls=None,
                latency_semantics='Saved stage latency only; historical end-to-end not established',
                artifact_hash=file_hash(p) if valid else None,
                provenance_status=old['status'] if valid else 'MISSING_OR_CHANGED',
                source_registry_hash=file_hash(path), source_registry=str(path),
                original_source_artifact=original,
                final_split_status='NOT_TEST; cannot pair or promote saved scores to TEST',
                provenance_note=old.get('note', ''), final_table='extended')
            if old['baseline_id'] in ('original_SARA_G0', 'semantic_k4', 'semantic_k6', 'semantic_k8', 'ORIGINAL_SARA/SARA'):
                e['final_table'] = 'primary'
            if old.get('frozen_protocol'):
                e['frozen_protocol'] = old['frozen_protocol']
            entries.append(e)
    # Saved W9 diagnostic policies are FIT findings, not confirmation predictions.
    sources = {
        **{p: ROOT/'results/week9/diagnostics/w9c_k2/grounded_faithfulness/v1/budget_128/fit/grounded_e_results.jsonl'
           for p in ('E0', 'E1-Q', 'E1-QD', 'E2')},
        **{p: ROOT/'results/week9/diagnostics/w9c_k2/extraction/fit/w9d_representation.jsonl' for p in ('D1', 'D2', 'D3')},
        'F1': F_ROOT/'fit/results.jsonl', 'D0': F_ROOT/'fit/results.jsonl',
        PRIMARY: F_ROOT/'frozen_w9f_intervention.json', SAFE: F_ROOT/'frozen_w9f_f2.json',
        VERIFY: G_ROOT/'final_fit/training_manifest.json'}
    # Resolve canonical D paths from saved E input lineage rather than assuming a path exists.
    e_protocol = read(E3_ROOT/'protocol.json')
    d_candidates = [Path(p) for p in e_protocol['input_hashes'] if Path(p).name == 'w9d_representation.jsonl']
    for p in ('D1', 'D2', 'D3'):
        if len(d_candidates) == 1:
            sources[p] = d_candidates[0]
    model = read(F_ROOT/'frozen_w9f_intervention.json')['generator_identity']
    with (G_ROOT/'scaffold/fold_assignments.csv').open() as stream:
        membership = list(csv.DictReader(stream))
    for name, p in sources.items():
        e = {k: None for k in FIELDS}
        e.update(baseline_id=name, family='W9_FROZEN', source_artifact=str(p), split='fit',
            query_ids=[r['question_id'] for r in membership], paper_ids=[r['paper_id'] for r in membership],
            retriever='FIXED-5/SEMANTIC@20; DraftSemantic K=2', generator='canonical trained SARA',
            fidelity_representation='E3-k3' if name in (PRIMARY, SAFE, VERIFY, 'F1') else 'see frozen policy',
            budget=3 if name in (PRIMARY, SAFE, VERIFY, 'F1') else None,
            revision_mode='F0' if name in (PRIMARY, VERIFY) else 'F2-A' if name == SAFE else name,
            model_id=model['model_id'], model_revision=model['model_revision'],
            artifact_hash=file_hash(p) if p.is_file() else None,
            provenance_status='FROZEN_DEPLOYMENT' if name in (PRIMARY, SAFE, VERIFY) else 'FIT_DIAGNOSTIC' if p.is_file() else 'MISSING',
            latency_semantics='Stage times only; no invented measured end-to-end latency',
            final_split_status='Requires fresh frozen TEST materialization',
            final_table='primary' if name in ('D0', PRIMARY, SAFE, VERIFY) else 'extended')
        entries.append(e)
    # QCLA was not in the older saved baseline manifest; explicit missing is honest.
    if not any(e['baseline_id'] == 'QCLA1' for e in entries):
        e = {k: None for k in FIELDS}
        e.update(baseline_id='QCLA1', family='W8_SECONDARY', split='internal_val',
                 source_artifact='results/week8/w8f_qcla/full/qcla_1_internal_val.jsonl',
                 provenance_status='NOT_AVAILABLE_FOR_TEST', final_table='extended')
        entries.append(e)
    require(all(set(FIELDS) <= e.keys() for e in entries), 'Registry lacks required provenance fields')
    return dict(schema='W9-H-baseline-registry-v1', entries=entries,
        registry_role='Historical inventory and frozen deployment definitions; no TEST metrics',
        missing_values='null means unestablished, never zero',
        standard_rag_note='Original Standard RAG uses its separate checkpoint; never label it SARA k10',
        no_outcome_based_removal=True, registry_digest=digest(entries))
