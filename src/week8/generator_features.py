"""Measured W8-D features on canonical k4 drafts; resumable without QA generation."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.week8.attribution import extract_generator_interaction_features
from src.week8.data import PROJECT_ROOT, load_config, load_week8_split
from src.week8.marginal_information import compute_mig_features_for_query
from src.week8.semantic import SemanticScorer
from src.week8.utility_training import load_surface, _validate_saved_routing
from src.week7.w7e.prompt_builder import (
    build_mixed_fidelity_sara_prompt, extract_mixed_fidelity_sara_embeddings,
)

SCHEMA_VERSION = 'w8d_measured_masking_v2_canonical_question'
FEATURE_COLUMNS = [
    'query_relevance_rq', 'draft_relevance_ry', 'redundancy_red',
    'candidate_diversity', 'mig_score', 'a6_draft_sim',
    'masking_delta_logp', 'masking_delta_logp_per_token',
    'masking_target_tokens', 'masking_available',
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(split, root=PROJECT_ROOT):
    """Invalidate resumed features when data, checkpoint, or implementation changes."""
    cfg = load_config()
    checkpoint = root / cfg['sara']['checkpoint_path']
    paths = [root / 'results/week8/surfaces' / f'promotion_surface_{split}.jsonl',
             root / cfg['splits']['train_retrieval'], root / cfg['splits']['train_sfr_embeddings'],
             root / cfg['splits'][f'{split}_manifest']]
    for name in ['adapter_config.json', 'adapter_model.safetensors', 'sara_extras.safetensors',
                 'tokenizer_config.json', 'sara_config.json']:
        paths.append(checkpoint / name)
    tokenizer_files = [checkpoint / name for name in [
        'tokenizer.json', 'tokenizer.model', 'added_tokens.json',
        'special_tokens_map.json', 'chat_template.jinja'] if (checkpoint / name).is_file()]
    if not any(p.name in ('tokenizer.json', 'tokenizer.model') for p in tokenizer_files):
        raise FileNotFoundError('SARA checkpoint has no saved tokenizer vocabulary')
    paths.extend(tokenizer_files)
    for name in ['src/week8/attribution.py', 'src/week8/generator_features.py',
                 'src/week8/semantic.py', 'src/week8/marginal_information.py',
                 'src/week8/utility_training.py', 'src/week7/w7e/prompt_builder.py',
                 'scripts/week8/run_w8d_feature_engineering.py',
                 'Baselines/SARA-main/src/model/loader.py',
                 'Baselines/SARA-main/src/model/xMistral/modeling_xmistral.py']:
        paths.append(root / name)
    sources = {str(p.relative_to(root)): sha256_file(p) for p in paths}
    payload = {'schema': SCHEMA_VERSION, 'split': split, 'sources': sources,
               'semantic_model': 'sentence-transformers/all-MiniLM-L6-v2',
               'alpha': .5, 'beta': .5, 'attribution': 'zero SFR vector before projector; draft only'}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return fingerprint, payload


def validate_feature_rows(rows, group, fingerprint):
    """Validate one complete query cache, including source identity and missing masks."""
    if len(rows) != 6 or {int(r['semantic_rank']) for r in rows} != set(range(5, 11)):
        raise ValueError('W8-D cache must contain all six unique candidate ranks per query')
    expected = {int(r.semantic_rank): r for r in group.itertuples()}
    draft_hash = hashlib.sha256(str(group.iloc[0].draft_answer).encode()).hexdigest()
    for row in rows:
        ref = expected[int(row['semantic_rank'])]
        if (row.get('source_fingerprint') != fingerprint or
                str(row['question_id']) != str(ref.question_id) or
                str(row['paper_id']) != str(ref.paper_id) or
                int(row['candidate_passage_index']) != int(ref.candidate_passage_index) or
                row.get('draft_sha256') != draft_hash):
            raise ValueError('W8-D cache sources/candidate routing differ; use a fresh output directory')
        available = int(row['masking_available'])
        if available != int(bool(str(group.iloc[0].draft_answer).strip())):
            raise ValueError('W8-D masking availability disagrees with canonical draft')
        names = [c for c in FEATURE_COLUMNS if available or c not in
                 ('masking_delta_logp', 'masking_delta_logp_per_token')]
        if not np.isfinite([row[c] for c in names]).all():
            raise ValueError('Non-finite measured W8-D features')
        if available:
            if row['masking_target_tokens'] <= 0 or not np.isclose(
                    row['masking_delta_logp'], row['masking_delta_logp_per_token'] * row['masking_target_tokens']):
                raise ValueError('W8-D masking sum/mean or target count is inconsistent')
        elif row['masking_target_tokens'] != 0 or any(
                row[c] is not None and not pd.isna(row[c])
                for c in ('masking_delta_logp', 'masking_delta_logp_per_token')):
            raise ValueError('Unavailable masking features must be null, not observed zeros')


def canonical_k4_context(group, record, embeddings, tokenizer):
    """Rebuild exactly the question normalization used by W7/W8-A generation."""
    qid = str(group.iloc[0].question_id)
    passages = record['passages'][:10]
    heavy = sorted(group.candidate_passage_index.astype(int).tolist())
    full = sorted(set(range(10)) - set(heavy))
    if len(passages) != 10 or len(set(heavy)) != 6 or len(full) != 4:
        raise ValueError(f'{qid}: invalid canonical k4 allocation')
    question = str(record['question']).strip()
    if question != str(group.iloc[0].get('question', question)).strip():
        raise ValueError(f'{qid}: retrieval question differs from canonical surface')
    if embeddings.ndim != 2 or len(embeddings) != 10 or not torch.isfinite(embeddings).all():
        raise ValueError(f'{qid}: expected ten finite cached SFR embeddings')
    prompt, tokens, compressed, _ = build_mixed_fidelity_sara_prompt(
        question, passages, full, [], heavy, {}, tokenizer, hybrid_medium=False)
    if compressed != heavy:
        raise ValueError('Compressed-slot ordering differs from original passage indices')
    if 'base_context_tokens' in group and tokens != int(group.iloc[0].base_context_tokens):
        raise ValueError(f'{qid}: rebuilt prompt token count differs from canonical k4')
    comp_embeds = extract_mixed_fidelity_sara_embeddings(embeddings, compressed)
    return question, prompt, tokens, compressed, comp_embeds, full


def compute_query_features(group, record, embeddings, model, tokenizer, scorer, fingerprint, device):
    qid = str(group.iloc[0].question_id)
    passages = record['passages'][:10]
    texts = [p['text'] for p in passages]
    sims = scorer.score_query_passages(record['question'], texts).cpu().numpy()
    expected_full = _validate_saved_routing(group, texts, sims)
    _, prompt, tokens, compressed, comp_embeds, full = canonical_k4_context(
        group, record, embeddings, tokenizer)
    if full != expected_full:
        raise ValueError('Saved semantic routing changed while rebuilding canonical prompt')
    candidates = {int(r.semantic_rank): int(r.candidate_passage_index) for r in group.itertuples()}
    candidate_text = {rank: texts[idx] for rank, idx in candidates.items()}
    slot_map = {rank: compressed.index(idx) for rank, idx in candidates.items()}
    draft = str(group.iloc[0].draft_answer)
    start = time.perf_counter()
    mig = compute_mig_features_for_query(record['question'], draft,
                                        [texts[i] for i in full], candidate_text, scorer)
    interaction = extract_generator_interaction_features(
        model, tokenizer, prompt, draft, comp_embeds, candidate_text, slot_map, scorer, device)
    elapsed = time.perf_counter() - start
    rows = []
    for rank in sorted(candidates):
        row = {'question_id': qid, 'paper_id': str(group.iloc[0].paper_id),
               'semantic_rank': rank, 'candidate_passage_index': candidates[rank],
               'compressed_slot_index': slot_map[rank],
               'draft_sha256': hashlib.sha256(draft.encode()).hexdigest(),
               'source_fingerprint': fingerprint, 'attribution_prompt_tokens': tokens,
               'feature_extraction_seconds_query': elapsed,
               **mig[rank], **interaction[rank]}
        for name in ('masking_delta_logp', 'masking_delta_logp_per_token'):
            if not np.isfinite(row[name]):
                row[name] = None
        rows.append(row)
    validate_feature_rows(rows, group, fingerprint)
    return rows


def run_feature_extraction(split, output_dir, *, device='cuda', smoke_limit=None):
    if split not in ('fit', 'internal_val'):
        raise ValueError('W8-D accepts FIT and INTERNAL-VAL only')
    df = load_surface(split)
    records, sfr, _ = load_week8_split(split)
    by_qid = {str(r['question_id']): r for r in records}
    groups = list(df.groupby('question_id', sort=True))
    if smoke_limit is not None:
        if smoke_limit < 1:
            raise ValueError('Smoke limit must be positive')
        groups = groups[:smoke_limit]
    fingerprint, provenance = source_fingerprint(split)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'features_{split}.jsonl'
    existing = {}
    if path.exists():
        with path.open() as f:
            for line_number, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_number}: interrupted cache write; inspect last line') from exc
                qid = str(entry['question_id'])
                if qid in existing:
                    raise ValueError('Duplicate query in W8-D cache')
                existing[qid] = entry['candidates']
    wanted = {str(q) for q, _ in groups}
    if set(existing) - wanted:
        raise ValueError('W8-D cache contains queries outside the requested run')
    for qid, group in groups:
        if str(qid) in existing:
            validate_feature_rows(existing[str(qid)], group, fingerprint)
    pending = [(str(q), g) for q, g in groups if str(q) not in existing]
    if pending:
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable for W8-D SARA extraction')
            free, _ = torch.cuda.mem_get_info(device)
            if free < 18 * 1024**3:
                raise RuntimeError('W8-D needs at least 18 GiB free GPU memory. Release earlier notebook models first.')
        # Use the same canonical SARA loader as W8-A; keep MiniLM on CPU.
        import src
        baseline_src = str(PROJECT_ROOT / 'Baselines/SARA-main/src')
        if baseline_src not in src.__path__:
            src.__path__.append(baseline_src)
        from src.model.loader import load_sara_for_eval
        cfg = load_config()
        model, tokenizer = load_sara_for_eval(str(PROJECT_ROOT / cfg['sara']['checkpoint_path']), device=device)
        scorer = SemanticScorer(device='cpu')
        try:
            with path.open('a') as f:
                for qid, group in tqdm(pending, desc=f'W8-D {split}', unit='query'):
                    rows = compute_query_features(group, by_qid[qid], sfr[qid],
                                                  model, tokenizer, scorer, fingerprint, device)
                    # One complete query per line: no half-complete six-candidate groups.
                    f.write(json.dumps({'question_id': qid, 'candidates': rows}, allow_nan=False) + '\n')
                    f.flush()
                    existing[qid] = rows
        finally:
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    result = pd.DataFrame([r for q, _ in groups for r in existing[str(q)]])
    result.to_csv(output_dir / f'features_{split}.csv', index=False)
    (output_dir / f'provenance_{split}.json').write_text(json.dumps(
        {**provenance, 'source_fingerprint': fingerprint, 'num_queries': len(groups),
         'num_rows': len(result), 'attention_A1_A5': 'not measured; masking fallback only',
         'feature_columns': FEATURE_COLUMNS}, indent=2))
    return result
