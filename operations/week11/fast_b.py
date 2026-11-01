"""B execution optimization: group sentences once and reuse identical cached vectors.

Reference pooling, ranking, metrics, inputs and action IDs are unchanged. Every
previous B ranking is checked before new rows are saved. Supplemental driver
hashes are retained in an execution receipt and in every newly produced row.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np

from src.week11.protocol import load, ROOT, guard, gate, authorization, scientific_config, code_identity
from src.week11.cache import atomic, digest, file_hash, lock, Store
from src.week11.provenance import identity, model_identity
from src.week11.datasets import load_dataset, source_identity
from src.week11.runner import Laboratory, sampled
from src.week11.readers import tokenizers
from src.week11.planning import plan
from src.week11.retrievers import passage_scores, ranked
from src.week11.evaluation import retrieval_metrics
from src.week11.reporting import report


def grouped_positions(units):
    positions = defaultdict(list)
    sentences = []
    for unit in units:
        if unit['unit_type'] == 'SENTENCE':
            positions[unit['passage_id']].append(len(sentences))
            sentences.append(unit)
    return sentences, positions


def execute_b(cfg=None, *, full=False, smoke=False, explicit=False, limit=3, **kwargs):
    cfg = cfg or load()
    guard()
    authorization(full, explicit, smoke, limit)
    if not full or smoke:
        raise ValueError('This driver is for the explicitly authorized real B resume only')
    gate(cfg, 'B')
    root = ROOT / cfg['output_root'] / 'passage_representation'
    saved_manifest = json.loads((root / 'manifest.json').read_text())
    sources = source_identity(cfg)
    models = {name: model_identity(cfg['models'][name], weights=True) for name in ['MiniLM', 'SFR']}
    expected = identity(cfg, 'B', sources, models, False)
    if saved_manifest['identity'] != expected:
        raise ValueError('Frozen scientific/model/source identity changed; refuse optimized resume')
    execution = dict(driver_sha256=file_hash(Path(__file__)), reference_identity=expected,
                     optimization='one-pass passage grouping + same cached vectors; no score/metric changes')
    receipt = root / 'optimized_execution_identity.json'
    if receipt.exists() and json.loads(receipt.read_text()) != execution:
        raise ValueError('Execution optimization changed; refuse silent resume')
    with lock(root / 'run.lock'):
        if not receipt.exists():
            atomic(receipt, execution)
        examples = sampled([e for name in cfg['datasets'] for e in load_dataset(name, cfg)], cfg)
        actual = [dict(dataset=e['dataset'], query_id=e['query_id'], document_id=e['document_id'], cluster_id=e['cluster_id']) for e in examples]
        if actual != json.loads((root / 'cohort.json').read_text())['selected']:
            raise ValueError('Selected query cohort changed')
        toks = tokenizers(cfg)
        shared = dict(config_hash=digest(scientific_config(cfg)), code_hash=digest(code_identity()),
                      dataset_hash=digest(sources), selection_models={k: cfg['models'][k] for k in ['SFR', 'MiniLM']},
                      synthetic=False, tokenizer_hash=digest({k: model_identity(cfg['models'][k], False) for k in cfg['readers']}),
                      embedding_weight_hash=digest({k: models[k] for k in ['SFR', 'MiniLM']}))
        lab = Laboratory(cfg, examples, toks, ROOT / cfg['cache_root'], shared, False)
        store = Store(root / 'rows', expected)
        rank_cache = Store(ROOT / cfg['cache_root'] / 'operational_B_rankings' / execution['driver_sha256'][:16], execution)
        prepared = {}
        reference_verified = 0
        pending = []
        started = time.perf_counter()
        try:
            for query_index, example in enumerate(examples):
                units = lab.units(example)
                passages = [u for u in units if u['unit_type'] == 'PASSAGE']
                corpus = digest([(u['unit_id'], u['hash']) for u in units])
                encoder = lab.encoder('SFR')
                if corpus not in prepared:
                    sentences, positions = grouped_positions(units)
                    # A already prepared these exact model/text/role vectors.
                    pv, _ = encoder.encode([p['source_text'] for p in passages], allow_compute=False)
                    sv, _ = encoder.encode([s['source_text'] for s in sentences], allow_compute=False)
                    grouped = [sv[positions[p['unit_id']]] if positions[p['unit_id']] else np.empty((0, pv.shape[1]), dtype=np.float32) for p in passages]
                    prepared[corpus] = pv, grouped
                pv, sv = prepared[corpus]
                q, _ = encoder.encode([example['question']], 'query', allow_compute=False)
                for action in plan(cfg, 'B', example['dataset']):
                    key = [example['dataset'], example['query_id'], action['action_id']]
                    cached = rank_cache.get(key)
                    if cached is None:
                        scores = passage_scores(q[0], pv, sv, action['representation'], cfg['passage_pooling_beta'])
                        ranking = ranked(passages, scores)
                        rank_cache.put(key, [u['unit_id'] for u in ranking])
                    else:
                        by_id = {p['unit_id']: p for p in passages}
                        ranking = [by_id[uid] for uid in cached]
                    ids = [u['unit_id'] for u in ranking]
                    old = store.get(key)
                    if old is not None:
                        if old['ranking'] != ids:
                            raise ValueError('Optimized driver disagrees with previously saved B ranking: ' + str(key))
                        reference_verified += 1
                    else:
                        # Verify ALL overlaps before writing any newly computed outcome.
                        pending.append((key, example, action, ranking))
                print('B pooling prepared query', query_index + 1, '/', len(examples), example['dataset'], flush=True)
            atomic(root / 'optimization_equivalence.json', dict(previous_rankings_verified=reference_verified,
                all_previous_rankings_exact=True, new_rankings_ready=len(pending),
                driver_sha256=execution['driver_sha256'], timestamp=datetime.now(timezone.utc).isoformat()))
            print('Exact overlap verification passed:', reference_verified, 'rankings; new rows:', len(pending), flush=True)
            for index, (key, example, action, ranking) in enumerate(pending):
                metrics = retrieval_metrics(ranking, example)
                store.put(key, dict(dataset=example['dataset'], query_id=example['query_id'], cluster_id=example['cluster_id'],
                    action_id=action['action_id'], retriever=action['retriever'], evidence_granularity=action['granularity'],
                    passage_representation=action['representation'], metrics=metrics, cost={},
                    ranking=[u['unit_id'] for u in ranking], mean_unit_tokens=float(np.mean([u['token_count'] for u in ranking])) if ranking else None,
                    mean_passage_tokens=float(np.mean([u['token_count'] for u in ranking])) if ranking else None,
                    synthetic=False, execution_driver_sha256=execution['driver_sha256']))
                print('B optimized saved', reference_verified + index + 1, '/600; elapsed', round(time.perf_counter() - started), 's', flush=True)
            rows = store.rows()
            flattened = []
            for row in rows:
                flattened.extend(dict(**{k: v for k, v in row.items() if k not in ['metrics', 'ranking', 'cost']},
                    **metric, top_ids=row['ranking'][:metric['k']], retrieval_latency=None,
                    document_preprocessing_original_seconds=None, document_embedding_lookup_seconds=None,
                    query_embedding_original_seconds=None, query_embedding_lookup_seconds=None) for metric in row['metrics'])
            expected_rows = sum(len(plan(cfg, 'B', e['dataset'])) for e in examples)
            atomic(root / 'status.json', dict(synthetic=False, executable_complete=len(rows) == expected_rows,
                                           completed=len(rows), expected=expected_rows, pending=[]))
            return report('B', flattened, cfg, root, synthetic=False)
        finally:
            lab.close()
