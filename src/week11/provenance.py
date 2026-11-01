import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from .protocol import ROOT, code_identity, guard,scientific_config
from .cache import atomic, digest, file_hash


def model_identity(spec, weights=True):
    from huggingface_hub import snapshot_download
    if not spec.get('revision') or spec['revision'] == 'main':
        raise ValueError('Immutable model revision required')
    path = Path(snapshot_download(spec['id'], revision=spec['revision'], local_files_only=True))
    suffixes = ('.json', '.model', '.jinja', '.safetensors', '.bin') if weights else ('.json', '.model', '.jinja')
    hashes = {str(p.relative_to(path)): file_hash(p) for p in path.rglob('*') if p.is_file() and p.suffix in suffixes}
    return dict(model=spec, files=hashes)


def identity(cfg, phase, sources, models=None, synthetic=False):
    guard()
    return dict(config_hash=digest(scientific_config(cfg,phase)), code_hash=digest(code_identity()), phase=phase,
                dataset_hash=digest(sources), model_hash=digest(models or {}), synthetic=synthetic)


def manifest(path, cfg, run_identity, phase, models):
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['identity'] != run_identity:
            raise ValueError('Resume identity mismatch; new output version required')
        return saved
    versions = {}
    for name in ['torch', 'transformers', 'sentence-transformers', 'numpy', 'pandas', 'llmlingua']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    import torch
    obj = dict(identity=run_identity, run_id=digest(run_identity), timestamp=datetime.now(timezone.utc).isoformat(),
        branch='gagan', git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
        relevant_dirty_hashes=code_identity(), models=models, configuration=cfg, phase=phase,
        environment=versions, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    atomic(path,obj)
    return obj
