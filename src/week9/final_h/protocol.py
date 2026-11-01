"""Immutable choices and separate H0/H1/H2/H3 authorization boundaries."""
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from ..provenance import ROOT, branch_guard, digest, file_hash

PRIMARY = 'W9-PRIMARY'
SAFE = 'W9-SAFE'
VERIFY = 'W9-VERIFY'
STRONG = ['original_SARA_G0', 'semantic_k8']
REQUIRED = ['D0', 'BM25_k8', 'semantic_k2', 'semantic_k4', 'semantic_k5',
            'semantic_k6', 'semantic_k8', 'original_SARA_G0', 'D1', PRIMARY, SAFE, VERIFY]
COMPARISONS = [(PRIMARY, b, 'primary') for b in
               ['D0', 'semantic_k4', 'semantic_k6', 'semantic_k8', 'original_SARA_G0']]
COMPARISONS += [(PRIMARY, 'D1', 'mechanistic'), (SAFE, PRIMARY, 'secondary'),
                (VERIFY, PRIMARY, 'secondary'), (VERIFY, 'D0', 'secondary'),
                (VERIFY, SAFE, 'secondary')]
E3_ROOT = ROOT/'results/week9/w9e3_senttopk/v1'
F_ROOT = ROOT/'results/week9/w9f_utilization/v1'
G_ROOT = ROOT/'results/week9/w9g_action_verification/v1'


def read(path):
    return json.loads(Path(path).read_text())


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT/p


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_once(path, value):
    """Exclusive creation. Even a writable file cannot be silently replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists():
        require(read(path) == value, f'Immutable artifact differs: {path}; preserve v1 and prepare a new pretest version')
        return
    with path.open('x') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def load_config(path):
    c = read(path)
    fixed = dict(schema='W9-H-v1', confirmation_split='test', primary_system=PRIMARY,
                 sentence_k=3, primary_mode='F0', safe_mode='F2-A', verifier_threshold=.70,
                 seed=42, bootstrap_resamples=10000, run_final_confirmation=False, run_robustness=False)
    require(all(c.get(k) == v for k, v in fixed.items()), 'Frozen W9-H method/statistical choices changed')
    return c


def success_criteria():
    return dict(
        primary=PRIMARY, strong_comparators=STRONG,
        quality='Positive lower 95% paired paper-cluster CI against BOTH W8 original_SARA/G0 and semantic k8; no major protocol issue.',
        efficiency='Point F1 loss <= .003 against BOTH strong comparators AND mean cumulative query prompt reduction >= .15 against BOTH.',
        mechanistic='Against D1: exposed evidence reduction >= .90 AND the entire paired 95% F1 difference CI lies within [-.003,.003].',
        quality_tolerance=.003, minimum_cumulative_saving=.15, minimum_evidence_suppression=.90,
        compatibility_is_not_equivalence=True,
        order=['QUALITY_HIT', 'EFFICIENCY_OR_MECHANISTIC_HIT', 'MIXED_OR_NEGATIVE'],
        missing_required_comparator='No final classification; evaluation is incomplete, never drop a comparator',
        note='Both strong comparators designated before confirmation. Conservative interval containment operationalizes quality compatibility; CI crossing zero alone is insufficient.')


def method_definition():
    from ..utilization_f.prompts import definitions
    return dict(primary_system=PRIMARY, primary='E3-k3 + F0', safe='E3-k3 + F2-A',
        verify='E3-k3 + F0 + G2 logistic; KEEP_D0 vs ACCEPT_THIS_F0',
        first_pass='Canonical original SARA: semantic 4 FULL + 6 HEAVY',
        retrieval=dict(representation='FIXED-5', retriever='SEMANTIC', width=20),
        coarse_selector='DraftSemantic', coarse_evidence_k=2, sentence_k=3,
        sentence_ranking='Query-only cosine; frozen E3 rank/tie/source presentation logic',
        sentence_splitter='academic-offset-rules-v1',
        sentence_presentation='Restore original document order, exact source substrings',
        generator_context='Canonical first-pass SARA latent context retained; frozen F messages appended inside original chat envelope',
        prompts={k: definitions()[k] for k in ('F0', 'F2-A')},
        decoding=dict(do_sample=False, max_new_tokens=128),
        verifier=dict(model='logistic', feature_count=32, threshold=.70, post_generation=True),
        no_retroactive_promotion=True, raw_fixed_policy_remains='K_E=0',
        cost_definitions=dict(
            final_prompt='Actual final answer-generation prompt tokens, including retained canonical SARA context',
            cumulative_prompt='Sum of actual query-time LM prompt tokens, including first pass and any revision/selector/extraction',
            generated='Sum of query-time generated tokens',
            verifier='Every query pays D0 + E3 + F0, even KEEP; verifier time additional, no refunds',
            embeddings='Reusable sentence index/preprocessing separated from online query embedding and ranking; cache hits retain conceptual origin cost',
            end_to_end='Only separately measured wall-clock interval; never sum stage estimates or fill missing with zero'))


def verify_hashes(hashes):
    for name, expected in hashes.items():
        p = resolve(name)
        require(p.is_file() and file_hash(p) == expected, f'Frozen dependency changed/missing: {name}')


def implementation_hashes():
    paths = list((ROOT/'src/week9').glob('*.py'))
    for package in ('senttopk_e3', 'utilization_f', 'action_verification_g', 'action_verification_g_audit', 'final_h'):
        paths += list((ROOT/'src/week9'/package).glob('*.py'))
    paths += [ROOT/'src/week6/evaluation.py', ROOT/'src/week7/w7e/prompt_builder.py',
              ROOT/'src/week8/semantic.py']
    paths += list((ROOT/'Baselines/SARA-main/src/model').rglob('*.py'))
    paths += [ROOT/'Baselines/SARA-main/src/const.py']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths))}


def system_contracts():
    """No baseline selection: explicit existing static and adaptive protocols."""
    model = read(F_ROOT/'frozen_w9f_intervention.json')['generator_identity']
    common = dict(generator=model, decoding=dict(do_sample=False, max_new_tokens=128),
                  prompt_builder_hash=file_hash(ROOT/'src/week7/w7e/prompt_builder.py'))
    out = {}
    for k in (2, 4, 5, 6, 8):
        out[f'semantic_k{k}'] = dict(common, retriever='canonical MiniLM semantic top-10 passage routing',
            full_k=k, heavy_k=10-k, revision_mode=None)
    out['BM25_k8'] = dict(common, retriever='canonical BM25 top-10 passage routing', full_k=8, heavy_k=2, revision_mode=None)
    out['D0'] = dict(out['semantic_k4'], role='Canonical shared initial SARA first pass')
    out['D1'] = dict(common, retrieval=dict(representation='FIXED-5', retriever='SEMANTIC', width=20),
                     selector='DraftSemantic', coarse_evidence_k=2, evidence='raw source chunks', revision_mode='F0')
    for name, mode in ((PRIMARY, 'F0'), (SAFE, 'F2-A'), (VERIFY, 'F0+G2')):
        out[name] = dict(method_definition(), system=name, mode=mode, generator=model)
    w8 = ROOT/'results/week8/w8e_controller/frozen_controller.json'
    out['original_SARA_G0'] = dict(common, base_full_k=4, max_promotions=1, revision_mode='G0',
        controller_path=str(w8), controller_sha256=file_hash(w8),
        controller_configuration=read(w8),
        gate_sha256=file_hash(ROOT/'results/week8/w8e_controller/b0_gw_gate.joblib'),
        selector_config_sha256=file_hash(ROOT/'results/week8/w8c_utility/w8c_frozen_selector_config.json'),
        selector_weights_sha256=file_hash(ROOT/'results/week8/w8c_utility/selected_neural.pt'),
        reference='Original W8 final-system protocol, not the W8-E cached replay; actual token uncertainty and frozen gate/selector')
    return out


def validate_freeze(root, need_ready=True):
    branch_guard()
    root = Path(root)
    p = root/'w9h_pretest_freeze_manifest.json'
    require(p.is_file(), 'Pretest freeze must exist before confirmation')
    m = read(p)
    require(m['manifest_digest'] == digest({k: v for k, v in m.items() if k != 'manifest_digest'}), 'Freeze digest differs')
    require(not (p.stat().st_mode & 0o222), 'Pretest freeze must remain read-only')
    require(m['method'] == method_definition(), 'Primary/prompts/settings changed after freeze')
    require(m['success_criteria'] == success_criteria(), 'Success criteria changed after freeze')
    verify_hashes(m['dependency_hashes'])
    if 'runtime' in m:
        import sys
        from importlib.metadata import version
        require(m['runtime'] == dict(python=sys.version, numpy=version('numpy'),
                    sklearn=version('scikit-learn'), torch=version('torch')), 'Frozen numerical runtime differs')
    if need_ready:
        require(m['h0_passed'] and not m['blockers'], 'H0 BLOCKED: ' + '; '.join(m['blockers']))
    return m


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
