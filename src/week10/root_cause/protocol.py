"""Freeze configuration and provenance before reading development records."""
import json
import subprocess
from pathlib import Path
from .storage import ROOT, digest, file_hash, atomic

DEVELOPMENT = ('dev', 'fit', 'internal_val')

def guard(split):
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='gagan':
        raise RuntimeError('Week 10 works only on gagan')
    if split not in DEVELOPMENT: raise ValueError('QASPER TEST is closed; development splits only')

def load(path):
    cfg=json.loads(Path(path).read_text()); guard(cfg['split'])
    out=(ROOT/cfg['output_root']).resolve()
    if not out.is_relative_to(ROOT/'results/week10/root_cause'): raise ValueError('Week-10 output required')
    for key in ('dataset','raw_dataset'):
        p=Path(cfg[key])
        if 'test' in p.name.lower() or '/w9h_final/' in str(p): raise ValueError('TEST inputs forbidden')
    return cfg

def source_hashes():
    paths=list((ROOT/'src/week10').rglob('*.py'))
    # Read-only dependencies, including runtime, splitter, metric and checkpoint loader.
    for rel in ('src/week9/backend.py','src/week9/provenance.py','src/week9/data.py',
                'src/week9/senttopk_e3/sentences.py','src/week9/retrieval.py','src/week6/routing.py',
                'src/week6/evaluation.py'):
        paths.append(ROOT/rel)
    paths+=list((ROOT/'Baselines/SARA-main/src/model').glob('*.py'))
    paths+=list((ROOT/'Baselines/SARA-main/src/metrics').glob('*.py'))
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)}

def freeze(cfg):
    guard(cfg['split'])
    inputs={str(Path(cfg[k]).resolve()):file_hash(cfg[k]) for k in ('dataset','raw_dataset')}
    checkpoint=ROOT/cfg['checkpoint']
    weights={str(p.relative_to(ROOT)):file_hash(p) for p in sorted(checkpoint.iterdir())
             if p.suffix in ('.json','.model','.safetensors','.jinja')}
    # Pin and hash actual cached backbone and encoder weights, not just mutable model names.
    from huggingface_hub import snapshot_download
    for spec in cfg['models'].values():
        if not spec.get('required_for_r',False): continue
        if not spec.get('revision') or spec['revision']=='main': raise ValueError('Immutable model revision required')
        directory=Path(snapshot_download(spec['id'],revision=spec['revision'],local_files_only=True))
        for p in sorted(directory.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.safetensors','.bin','.model'):
                weights[str(p)]=file_hash(p)
    import importlib.metadata as md
    versions={name:md.version(name) for name in ('torch','transformers','sentence-transformers','llama-index-core','numpy')}
    identity=dict(config=cfg,config_hash=digest(cfg),datasets=inputs,weights=weights,sources=source_hashes(),packages=versions)
    out=ROOT/cfg['output_root']; p=out/'execution_freeze.json'
    if p.exists() and json.loads(p.read_text())!=identity: raise ValueError('Frozen dependencies changed; new output version required')
    if not p.exists(): atomic(p,identity)
    return identity
