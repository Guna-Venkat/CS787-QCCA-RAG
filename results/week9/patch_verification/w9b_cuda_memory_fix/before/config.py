"""Frozen budgets, phase dependencies and explicit TEST authorization."""
import json
from pathlib import Path
import yaml
from .provenance import ROOT, digest, verify_sources, file_hash

PHASES = 'abcdefgh'
PREREQUISITES = {'a': [], 'b': ['a'], 'c': ['a', 'b'], 'd': ['a', 'c'],
                 'e': ['a', 'c', 'd'], 'f': ['a', 'c', 'd', 'e'],
                 'g': ['a', 'b', 'f'], 'h': ['a', 'c', 'd', 'e', 'f', 'g']}


def load_config(path):
    with Path(path).open() as f:
        cfg = yaml.safe_load(f)
    if cfg['initial_full_k'] != 4:
        raise ValueError('Canonical initial_full_k must remain 4')
    if cfg['encoder'] != 'sentence-transformers/all-MiniLM-L6-v2':
        raise ValueError('Primary controlled encoder must remain MiniLM')
    if cfg['retrieval_widths'] != [10, 20, 40] or cfg['evidence_counts'] != [0, 1, 2, 4]:
        raise ValueError('Preserve the preregistered W and K_E grids')
    return cfg


def load_frozen(path, phase=None):
    value = json.loads(Path(path).read_text())
    if value.get('frozen') is not True or value.get('selection_split') != 'fit' or not value.get('complete'):
        raise ValueError(f'Not a complete FIT-frozen artifact: {path}')
    if phase and value['phase'] != phase:
        raise ValueError('Frozen phase mismatch')
    verify_sources(value['source_hashes'])
    if digest(value['settings']) != value['settings_hash']:
        raise ValueError('Frozen settings hash mismatch')
    for name, expected in {**value['input_hashes'], **value.get('model_hashes',{}), **value.get('output_hashes',{})}.items():
        if file_hash(name) != expected:
            raise ValueError(f'Frozen input artifact changed: {name}')
    return value


def check_test(split, allow_test=False, frozen_config=None):
    if split not in ('fit', 'internal_val', 'dev', 'test'):
        raise ValueError('Unknown split')
    if split == 'test':
        if not allow_test or not frozen_config:
            raise PermissionError('TEST requires --allow-test AND --frozen-config')
        frozen = load_frozen(frozen_config, 'h')
        if frozen.get('analysis_role') != 'fit_oof':
            raise PermissionError('TEST requires a final FIT-OOF frozen policy')
        return frozen
