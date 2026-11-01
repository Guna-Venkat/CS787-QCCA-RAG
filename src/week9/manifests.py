"""Run manifests and immutable manual phase freezes."""
from datetime import datetime, timezone
import json
import subprocess
from pathlib import Path
from .provenance import ROOT, branch_guard, digest, file_hash, source_hashes
from .caching import atomic_json


def manifest(cfg, phase, split, seed, inputs, outputs, scope):
    return dict(phase=phase, split=split, seed=seed, git_branch=branch_guard(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        dirty_working_tree=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).splitlines(),
        source_hashes=source_hashes(), config_hash=digest(cfg), config=cfg,
        models=dict(encoder=cfg['encoder'], checkpoint=cfg['checkpoint']),
        input_hashes={str(Path(p).resolve()): file_hash(p) for p in inputs},
        output_paths=[str(Path(p).resolve()) for p in outputs], scope=scope,
        timestamp=datetime.now(timezone.utc).isoformat())


def freeze(path, phase, settings, run_manifest, complete, analysis_role='fit_fixed'):
    if run_manifest['split'] != 'fit' or not complete or run_manifest['scope'] != 'full':
        raise ValueError('Only complete unrestricted FIT runs can be frozen')
    value = dict(frozen=True, complete=True, phase=phase, selection_split='fit',
        analysis_role=analysis_role, settings=settings, settings_hash=digest(settings),
        source_hashes=run_manifest['source_hashes'], input_hashes=run_manifest['input_hashes'],
        model_hashes=run_manifest.get('model_hashes',{}),
        protocol={k:run_manifest['config'][k] for k in (
            'encoder','checkpoint','max_new_tokens','extraction_max_tokens','query_draft_alpha')},
        output_hashes={p: file_hash(p) for p in run_manifest['output_paths']},
        timestamp=datetime.now(timezone.utc).isoformat())
    if phase in 'defgh' and run_manifest['config'].get('textual_model_roles'):
        value['protocol']['textual_model_roles']=run_manifest['config']['textual_model_roles']
    path = Path(path)
    if path.exists() and json.loads(path.read_text())['settings_hash'] != value['settings_hash']:
        raise FileExistsError('Refusing to overwrite a different frozen decision')
    atomic_json(path, value)
    return value
