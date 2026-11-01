import json
import subprocess
from pathlib import Path
import yaml
from src.week10.root_cause.storage import ROOT, digest, file_hash, atomic

PHASES = ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
DIRECTORIES = dict(zip(PHASES, ['setup', 'retrieval', 'passage_representation', 'granularity_budget',
    'reader_granularity', 'compressor_reader', 'retriever_robustness', 'hierarchical', 'oracle', 'final_report']))


def guard(dataset=None, split=None):
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'gagan':
        raise RuntimeError('Week 11 works ONLY on gagan')
    if dataset == 'QASPER' and split not in ('dev', 'fit', 'internal_val'):
        raise ValueError('QASPER TEST is blocked for normalization, selection, generation and evaluation')


def load(path=None):
    guard()
    path = Path(path or ROOT/'configs/week11/week11_base.yaml')
    cfg = yaml.safe_load(path.read_text())
    for name in cfg.pop('includes', []):
        part = yaml.safe_load((path.parent/name).read_text())
        if set(cfg) & set(part):
            raise ValueError('Duplicate configuration keys: '+name)
        cfg.update(part)
    if cfg['verification_dataset'] != 'SciFact':
        raise ValueError('Verification choice is frozen to SciFact')
    for name, spec in cfg['datasets'].items():
        guard(name, spec['split'])
        for key in ('path', 'raw_path', 'corpus_path'):
            if name == 'QASPER' and key in spec and 'test' in Path(spec[key]).name.lower():
                raise ValueError('QASPER TEST path forbidden')
    if not (ROOT/cfg['output_root']).resolve().is_relative_to(ROOT/'results/week11'):
        raise ValueError('Week 11 outputs only')
    return cfg


def code_identity():
    files = list((ROOT/'src/week11').glob('*.py')) + list((ROOT/'scripts/week11').glob('*.py'))
    # Read-only reused implementations are part of identity as well.
    for rel in ['src/week10/root_cause/encoding.py', 'src/week10/root_cause/runtime.py',
                'src/week10/root_cause/storage.py', 'src/week9/retrieval.py',
                'src/week9/senttopk_e3/sentences.py', 'src/week6/evaluation.py', 'src/week9/evaluation.py']:
        files.append(ROOT/rel)
    from src.week10.root_cause.protocol import source_hashes
    return dict(source_hashes(),**{str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)})


def output(cfg, phase):
    return ROOT/cfg['output_root']/DIRECTORIES[phase]


def scientific_config(cfg,phase=None):
    # Prospective manual finalists and saved-result analyses can be specified
    # after earlier freezes, without silently changing any earlier experiment.
    later={'finalist_run_ids','finalist_configurations','interaction_contrasts','history_trajectory_records','_retrieval_saved_rows'}
    value={k:v for k,v in cfg.items() if k not in later}
    names={'F':['finalist_run_ids','finalist_configurations'],'H':['interaction_contrasts'],'I':['history_trajectory_records']}
    for name in names.get(phase,[]):value[name]=cfg[name]
    return value


def review(cfg, phase, selected, reason):
    if phase not in PHASES or not reason or not isinstance(selected, dict):
        raise ValueError('Explicit phase, reviewed configuration dictionary and rationale required')
    status_path = output(cfg, phase)/'status.json'
    status = json.loads(status_path.read_text())
    if status.get('synthetic') or not status.get('executable_complete'):
        raise ValueError('Cannot freeze smoke or incomplete real work')
    pending = status.get('pending', [])
    if pending and set(selected.get('waive_pending', [])) != set(pending):
        raise ValueError('Explicitly review/waive every pending condition or implement it before freezing')
    from datetime import datetime, timezone
    p = ROOT/cfg['output_root']/f'freeze_{phase}.json'
    identity = dict(phase=phase, reviewed_config=selected, rationale=reason,
        source_result_hash=file_hash(status_path), config_hash=digest(scientific_config(cfg,phase)), code_hash=digest(code_identity()), branch='gagan')
    if p.exists():
        old = json.loads(p.read_text())
        if any(old[k] != v for k, v in identity.items()):
            raise ValueError('Review already frozen differently; use a new output version')
        return old
    obj = dict(identity, timestamp=datetime.now(timezone.utc).isoformat())
    atomic(p, obj)
    return obj


def gate(cfg, phase):
    guard()
    if phase == '0':
        return
    previous = PHASES[PHASES.index(phase)-1]
    path = ROOT/cfg['output_root']/f'freeze_{previous}.json'
    if not path.exists():
        raise RuntimeError(f'Review and freeze {previous} before full phase {phase}')
    frozen = json.loads(path.read_text())
    if frozen['config_hash'] != digest(scientific_config(cfg,previous)) or frozen['code_hash'] != digest(code_identity()):
        raise ValueError('Frozen configuration/code changed; refuse downstream execution')
    status_path = output(cfg, previous)/'status.json'
    if file_hash(status_path) != frozen['source_result_hash']:
        raise ValueError('Reviewed upstream results changed')


def authorization(full=False, explicit=False, smoke=False, limit=None):
    if full and (not explicit or smoke):
        raise ValueError('Full run requires explicit opt-in and smoke=False')
    if smoke and (limit is None or not 1 <= limit <= 3):
        raise ValueError('Smoke is limited to 1–3 queries per dataset')
