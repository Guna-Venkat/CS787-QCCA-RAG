"""Content-addressed identities and strict inference schemas."""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ('gold', 'target_', 'oracle', 'true_delta', 'promoted_f1', 'is_gold', 'delta_f1', 'base_f1')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def assert_inference_clean(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if any(part in key.lower() for part in FORBIDDEN):
                raise ValueError(f'Forbidden inference field: {key}')
            assert_inference_clean(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_inference_clean(child)


def branch_guard():
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'gagan':
        raise RuntimeError(f'BRANCH SAFETY: expected gagan; found {branch}')
    return branch


def source_hashes():
    paths = list((ROOT / 'src/week9').glob('*.py')) + list((ROOT / 'scripts/week9').glob('*.py'))
    paths += [ROOT / p for p in [
        'src/week7/w7e/prompt_builder.py', 'src/week6/evaluation.py', 'src/week8/semantic.py',
        'Baselines/SARA-main/src/model/loader.py', 'Baselines/SARA-main/src/model/xMistral/modeling_xmistral.py']]
    paths += list((ROOT / 'Baselines/SANE-main/sane/agents').glob('sane_*.py'))
    paths += list((ROOT / 'Baselines/SANE-main/config').glob('*.yaml'))
    paths += [ROOT / 'Baselines/papers/SANE.pdf', ROOT / 'Baselines/papers/SemRAG.pdf']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def verify_sources(hashes):
    if not hashes:
        raise ValueError('Missing required source fingerprints')
    for name, expected in hashes.items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f'Source fingerprint changed: {name}; use a new run directory')


def aligned(left, right):
    a = {(str(r['question_id']), str(r['paper_id'])) for r in left}
    b = {(str(r['question_id']), str(r['paper_id'])) for r in right}
    if a != b:
        raise ValueError('Query/paper identities are not aligned')
