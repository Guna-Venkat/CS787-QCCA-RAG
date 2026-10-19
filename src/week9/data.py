"""Real artifacts only, raw paper source order, separated inference/evaluation views."""
from __future__ import annotations
from functools import lru_cache
import csv
import json
import re
from pathlib import Path
from .caching import read_jsonl
from .provenance import ROOT, digest, assert_inference_clean
from .config import check_test


def resolve(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def split_paths(cfg, split):
    key = 'dev' if split == 'dev' else 'train'
    if split == 'test':
        key = 'test'
    data = cfg['data']
    names = ['retrieval', 'reformatted', 'canonical', 'sfr']
    required = [key + '_' + n for n in names] + ['raw_test' if split == 'test' else 'raw_train']
    missing = [name for name in required if not data.get(name)]
    if missing:
        raise FileNotFoundError(f'Explicit real split artifacts must be configured: {missing}')
    paths = {n: resolve(data[key + '_' + n]) for n in names}
    paths['raw'] = resolve(data['raw_test'] if split == 'test' else data['raw_train'])
    if split in ('fit', 'internal_val'):
        paths['manifest'] = resolve(data[split + '_manifest'])
        paths['surface'] = ROOT / f'results/week8/surfaces/promotion_surface_{split}.jsonl'
        paths['diagnostics'] = ROOT / f'results/week8/w8e_gate_inputs/canonical_diagnostics_{split}.jsonl'
    return paths


def load_records(cfg, split, *, allow_test=False, frozen_config=None):
    check_test(split, allow_test, frozen_config)
    paths = split_paths(cfg, split)
    # SFR is checked when generation is requested, not needed for retrieval.
    for name, p in paths.items():
        if name != 'sfr' and not p.is_file():
            raise FileNotFoundError(f'Missing real {name} artifact: {p}')
    allowed = None
    manifest_papers = {}
    if 'manifest' in paths:
        with paths['manifest'].open() as f:
            rows = list(csv.DictReader(f))
        allowed = {str(r['question_id']) for r in rows}
        manifest_papers = {str(r['question_id']): str(r['paper_id']) for r in rows}
    records = read_jsonl(paths['retrieval'])
    records = [r for r in records if allowed is None or str(r['question_id']) in allowed]
    if allowed is not None and {str(r['question_id']) for r in records} != allowed:
        raise ValueError('Retrieval/manifest question coverage differs')
    if len({str(r['question_id']) for r in records}) != len(records):
        raise ValueError('Duplicate query IDs')
    reformatted = {str(r['id']): r for r in read_jsonl(paths['reformatted'])}
    canonical = {}
    for r in read_jsonl(paths['canonical']):
        if int(r['k']) == 4:
            qid = str(r['question_id'])
            if qid in canonical:
                raise ValueError('Duplicate canonical k4 draft')
            canonical[qid] = r
    w8 = {}
    diagnostics = {str(r['question_id']): r for r in read_jsonl(paths['diagnostics'])} if 'diagnostics' in paths else {}
    if 'surface' in paths:
        for row in read_jsonl(paths['surface']):
            w8.setdefault(str(row['question_id']), []).append(row)
    output = []
    for rec in records:
        qid, pid = str(rec['question_id']), str(rec['paper_id'])
        if manifest_papers and manifest_papers[qid] != pid:
            raise ValueError('Manifest paper identity mismatch')
        source, draft = reformatted[qid], canonical[qid]
        if str(source['example_id']) != pid or str(draft['paper_id']) != pid:
            raise ValueError('Canonical/reformatted paper identity mismatch')
        if source['question'].strip() != rec['question'].strip():
            raise ValueError('Question identity differs')
        full = sorted(map(int, draft['selected_full_indices']))
        heavy = sorted(map(int, draft['compressed_indices']))
        if len(full) != 4 or len(heavy) != 6 or set(full) & set(heavy) or set(full + heavy) != set(range(10)):
            raise ValueError('Canonical k4 must be 4 FULL + 6 HEAVY')
        for row in w8.get(qid, []):
            if row['base_prediction'] != draft['prediction'] or int(row['candidate_passage_index']) not in heavy:
                raise ValueError('W8 canonical draft/routing corruption')
        draft = dict(draft)
        if qid in diagnostics:
            diag = diagnostics[qid]
            if diag['draft_answer'] != draft['prediction'] or not diag['greedy_verified'] or str(diag['paper_id']) != pid:
                raise ValueError('Saved generated token IDs disagree with the canonical draft')
            draft['saved_generated_token_ids'] = diag['token_ids']
        if 'surface' in paths and len(w8.get(qid, [])) != 6:
            raise ValueError(f'{qid}: expected six frozen W8 candidate identities')
        if 'surface' in paths and {int(r['candidate_passage_index']) for r in w8[qid]} != set(heavy):
            raise ValueError('W8 candidate identities must exactly cover the six compressed slots')
        inference = dict(question_id=qid, paper_id=pid, question=rec['question'].strip(),
            initial_draft=draft['prediction'], selected_full_indices=full, compressed_indices=heavy,
            passages=[{k: p[k] for k in ('passage_id', 'text', 'rank', 'score')} for p in rec['passages'][:10]])
        assert_inference_clean(inference)
        output.append(dict(inference=inference, canonical=draft, original_passages=source['context'],
            evaluation=dict(answers=rec['gold_answers'], answers_reformatted=rec['gold_answers_reformatted'])))
    return output, paths


@lru_cache(maxsize=3)
def raw_dataset(path):
    from datasets import Dataset
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return Dataset.from_file(str(path))


def source_document(record, raw_path):
    """Match SARA source transformation, retain section and source offsets."""
    pid = record['inference']['paper_id']
    paper = raw_dataset(str(raw_path))[int(pid)]
    sections = [(paper['title'], paper['title'] + '\t' + paper['abstract'])]
    for name, paras in zip(paper['full_text']['section_name'], paper['full_text']['paragraphs']):
        if not paras or paras == ['']:
            continue
        sections.append((str(name), re.sub(r'BIBREF\d+', '', str(name) + '\t' + ' '.join(paras))))
    spans, pieces, cursor = [], [], 0
    for name, text in sections:
        spans.append(dict(section=name, char_start=cursor, char_end=cursor + len(text)))
        pieces.append(text)
        cursor += len(text) + 2
    text = '\n\n'.join(pieces)
    return dict(paper_id=pid, source_document_id=str(paper['id']), text=text, sections=spans,
                source_sha256=digest(text))


def evaluation_evidence(record, raw_path):
    """Exact paper and question lookup; no ambiguous global question-string map."""
    paper = raw_dataset(str(raw_path))[int(record['inference']['paper_id'])]
    question = record['inference']['question']
    indices = [i for i, q in enumerate(paper['qas']['question']) if q.strip() == question]
    if len(indices) != 1:
        raise ValueError('Raw question alignment must be unique within its paper')
    annotations = paper['qas']['answers'][indices[0]]['answer']
    return list(dict.fromkeys(e for a in annotations for e in a.get('evidence', []) if e.strip()))
