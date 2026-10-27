"""H0 never opens TEST data. Missing runtime adapters block, not weaken, the study."""
import csv
import sys
from importlib.metadata import version
from pathlib import Path
from . import protocol as p
from .registry import inventory


def materialization_audit(c):
    blockers = []
    spec = None
    if not c.get('materialization_spec'):
        blockers.append('TEST materialization spec absent: retrieval/first-pass/SFR inputs and same-split baseline producers are not configured')
        return None, blockers
    path = p.resolve(c['materialization_spec'])
    if not path.is_file():
        return None, ['Configured materialization spec is missing']
    spec = p.read(path)  # Protocol metadata only: no answers, predictions or labels.
    contracts = p.system_contracts()
    p.require(spec.get('schema') == 'W9-H-materialization-v1' and spec.get('split') == 'test', 'Invalid materialization protocol')
    allowed = {'schema', 'split', 'systems', 'labels_path', 'membership_path', 'test_inputs', 'provenance'}
    p.require(set(spec) <= allowed, 'Materialization spec must contain metadata only')
    for name in p.REQUIRED:
        if name not in spec.get('systems', {}):
            blockers.append('No frozen TEST producer for ' + name)
    for name, value in spec.get('systems', {}).items():
        p.require({'command', 'producer_source_hashes', 'predictions_path', 'receipt_path', 'method_config'} <= value.keys(), 'Incomplete producer specification: ' + name)
        p.require(value['command'] and value['producer_source_hashes'], 'Producer needs a frozen command and code: ' + name)
        if name in contracts:
            p.require(value['method_config'] == contracts[name], 'Producer method differs from frozen system: ' + name)
        p.verify_hashes(value['producer_source_hashes'])
        # An untouched confirmation cannot already have query predictions.
        if p.resolve(value['predictions_path']).exists() or p.resolve(value['receipt_path']).exists():
            blockers.append('TEST output predates pretest freeze: ' + name)
    if not spec.get('test_inputs'):
        blockers.append('No explicit canonical TEST input paths')
    for name, path in spec.get('test_inputs', {}).items():
        if not p.resolve(path).is_file():
            blockers.append('Missing TEST input: ' + name)
    for name in ('labels_path', 'membership_path'):
        if not spec.get(name):
            blockers.append('Missing final ' + name)
    return spec, blockers


def run(config_path):
    p.branch_guard()
    c = p.load_config(config_path)
    root = p.resolve(c['output_root'])
    freeze_path = root/'w9h_pretest_freeze_manifest.json'
    if freeze_path.exists():
        freeze = p.validate_freeze(root, need_ready=False)
        print('Existing immutable H0:', 'PASS' if freeze['h0_passed'] else 'BLOCKED')
        return freeze
    a_path = p.ROOT/'results/week9/frozen/w9a_frozen_config.json'
    a = p.read(a_path)
    e = p.read(p.E3_ROOT/'frozen_e3_senttopk.json')
    f = p.read(p.F_ROOT/'frozen_w9f_intervention.json')
    safe = p.read(p.F_ROOT/'frozen_w9f_f2.json')
    ep = p.read(p.E3_ROOT/'protocol.json')
    fp = p.read(p.F_ROOT/'protocol.json')
    p.require(a['frozen'] and a['complete'] and a['selection_split'] == 'fit', 'A not FIT-frozen')
    p.require({k: a['settings'][k] for k in ('representation', 'retriever', 'retrieval_width')} ==
              dict(representation='FIXED-5', retriever='SEMANTIC', retrieval_width=20), 'A choice differs')
    p.require(e['frozen'] and e['selected_k'] == 3 and e['selection_split'] == 'fit', 'E3 k changed')
    p.require(f['frozen'] and f['split'] == 'fit' and f['actual_policy'] == f['selected_intervention_mode'] == 'F0', 'F0 not frozen primary')
    p.require(safe['frozen'] and safe['selected_variant'] == 'F2-A', 'F2-A freeze differs')
    p.require(fp['model'] == f['generator_identity'], 'Model identity differs')
    p.require(p.read(p.F_ROOT/'prompts.json') == __import__('src.week9.utilization_f.prompts', fromlist=['definitions']).definitions(), 'Frozen F prompts differ')
    # Verify historical scientific inputs/outputs without opening TEST.
    dep = p.implementation_hashes()
    for source in (a, e, f, ep, fp):
        for key in ('artifact_hashes', 'input_hashes', 'model_hashes'):
            for name, expected in source.get(key, {}).items():
                p.require('test' not in Path(name).name.lower(), 'Unexpected TEST dependency in a FIT freeze')
                dep[name] = expected
    dep.update(ep['encoder']['file_hashes'])
    encoder_sample = Path(next(iter(ep['encoder']['file_hashes'])))
    encoder_snapshot = next(parent for parent in encoder_sample.parents if parent.name == ep['encoder']['revision'])
    encoder_ref = encoder_snapshot.parent.parent/'refs/main'
    if encoder_ref.is_file():
        dep[str(encoder_ref)] = p.file_hash(encoder_ref)
    # The trained checkpoint contains adapters/projector, not the public backbone.
    # Pin the actual local backbone and cache ref as well; no download/model load.
    from huggingface_hub import snapshot_download
    backbone = Path(snapshot_download(f['generator_identity']['model_id'], local_files_only=True))
    backbone_hashes = {str(path): p.file_hash(path) for path in sorted(backbone.rglob('*'))
                       if path.is_file() and path.suffix in ('.safetensors', '.json', '.model')}
    p.require(any(name.endswith('.safetensors') for name in backbone_hashes), 'Canonical public backbone weights unavailable locally')
    ref = backbone.parent.parent/'refs/main'
    if ref.is_file():
        backbone_hashes[str(ref)] = p.file_hash(ref)
    dep.update(backbone_hashes)
    from ..action_verification_g_audit.final_fit import validate_audit
    validate_audit(p.G_ROOT/'audit_pre_final/audit_status.json')
    model_root = p.G_ROOT/'final_fit'
    blockers = []
    if not (model_root/'training_manifest.json').exists():
        blockers.append('Audited G2 final package missing; export audited full-FIT model without refit')
    else:
        gm = p.read(model_root/'training_manifest.json')
        schema = p.read(model_root/'feature_schema.json')
        threshold = p.read(model_root/'threshold.json')
        p.require(gm['complete'] and gm['split'] == 'fit' and gm['n_queries'] == 1746 and not gm['test_touched'], 'G2 lineage invalid')
        p.require(schema['feature_count'] == len(schema['feature_columns']) == 32 and threshold['threshold'] == .70, 'G2 schema/threshold changed')
        for name, expected in gm['artifact_hashes'].items():
            dep[str(model_root/name)] = expected
        dep[str(model_root/'training_manifest.json')] = p.file_hash(model_root/'training_manifest.json')
    p.verify_hashes(dep)
    registry = inventory()
    for entry in registry['entries']:
        if entry.get('artifact_hash'):
            dep[entry['source_artifact']] = entry['artifact_hash']
    import_manifest = p.ROOT/'results/week9/baselines/w8_import_manifest.json'
    dep[str(import_manifest)] = p.file_hash(import_manifest)
    for entry in p.read(import_manifest)['files'].values():
        dep[entry['imported_path']] = entry['sha256']
    spec, missing = materialization_audit(c)
    blockers += missing
    w8_missing = [name for name in ('full_system.py', 'final_dev.py', 'final_reporting.py')
                  if not (p.ROOT/'src/week8'/name).is_file()]
    if w8_missing:
        blockers.append('Historical W8 runtime source absent from current tree: ' + ', '.join(w8_missing) + '; preserve/recover the exact audited implementation for the TEST adapter')
    if not c.get('test_draft_provenance', '').startswith('User confirmed synthetic fixture'):
        blockers.append('Unresolved test_draft provenance; cannot certify untouched TEST')
    split_status = dict(fit='Extensively developed', dev='Inspected; not untouched confirmation',
        internal_val='Inspected; not untouched confirmation', test='Locked according to config/code/history and human provenance; no claim of exhaustive external access logging',
        suspicious_filename=c.get('test_draft_provenance'),
        test_data_opened_by_h0=False, test_labels_opened_by_agent=False,
        evidence=['configs/week8/w8_dafir.yaml: test_disabled=true',
                  'src/week9/config.py: explicit TEST authorization and frozen config guard',
                  'Existing final split paths absent from W9 data configuration',
                  'User explicitly confirmed historical test_draft is synthetic; not inspected'],
        known_test_query_count=c['test_query_count'])
    panel = p.ROOT/'results/week9/preflight/frozen_model_panel.json'
    robustness = dict(executed=False, status='No preexisting frozen model panel found; no new model selection')
    if panel.is_file():
        data = p.read(panel)
        valid = data.get('frozen') and data.get('selection_split') == 'fit' and data.get('panel_hash') == p.digest({k: v for k, v in data.items() if k != 'panel_hash'})
        p.require(valid, 'Model panel exists but has invalid freeze provenance')
        dep[str(panel)] = p.file_hash(panel)
        robustness['status'] = 'Existing FIT-only panel inventoried; robustness not executed in this pass'
    protocol = dict(schema='W9-H-v1', method=p.method_definition(), split_policy=split_status,
        system_contracts=p.system_contracts(),
        success_criteria=p.success_criteria(), comparisons=p.COMPARISONS,
        bootstrap=dict(cluster='paper_id', resamples=10000, seed=42, weighting='queries equally; resample whole papers'),
        multiple_comparisons='Raw paired CIs, predeclared comparison families. No adjusted significance claim.',
        robustness=robustness, required_systems=p.REQUIRED, materialization_spec=spec,
        missing_baseline_policy='Never pair different splits, partial memberships, or incompatible evaluators',
        qualitative_analysis='Deterministic sample after aggregate seal only; never feeds methods')
    p.write_once(root/'protocol.json', protocol)
    p.write_once(root/'baseline_registry.json', registry)
    rows = []
    for entry in registry['entries']:
        rows.append(dict(baseline_id=entry['baseline_id'], saved_split=entry['split'],
            provenance_status=entry['provenance_status'], final_table=entry['final_table'],
            compatible_for_test=False, reason='No saved TEST predictions; earlier splits cannot be paired with TEST',
            source_artifact=entry['source_artifact']))
    matrix = root/'compatibility_matrix.csv'
    with matrix.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    matrix.chmod(0o444)
    for path in (a_path, p.E3_ROOT/'frozen_e3_senttopk.json', p.F_ROOT/'frozen_w9f_intervention.json',
                 p.F_ROOT/'frozen_w9f_f2.json', p.F_ROOT/'prompts.json', p.E3_ROOT/'protocol.json',
                 p.F_ROOT/'protocol.json', root/'protocol.json', root/'baseline_registry.json', matrix, p.resolve(config_path)):
        dep[str(path)] = p.file_hash(path)
    if c.get('materialization_spec'):
        dep[str(p.resolve(c['materialization_spec']))] = p.file_hash(p.resolve(c['materialization_spec']))
        for spec_entry in spec['systems'].values():
            dep.update(spec_entry['producer_source_hashes'])
    for name in ('results/week8/w8e_controller/frozen_controller.json',
                 'results/week8/w8e_controller/b0_gw_gate.joblib',
                 'results/week8/w8c_utility/w8c_frozen_selector_config.json',
                 'results/week8/w8c_utility/selected_neural.pt'):
        dep[name] = p.file_hash(p.ROOT/name)
    freeze = dict(schema='W9-H-pretest-v1', timestamp_utc=p.timestamp(), branch='gagan', git_commit=p.commit(),
        relevant_working_tree_sources=p.implementation_hashes(), dependency_hashes=dep,
        method=p.method_definition(), system_contracts=p.system_contracts(), model=f['generator_identity'], encoder=ep['encoder'],
        public_backbone=dict(model_id=f['generator_identity']['model_id'], revision=backbone.name, file_hashes=backbone_hashes),
        tokenizer_hashes={k: v for k, v in ep['model_hashes'].items() if any(t in k for t in ('token', 'vocab', '.model', 'chat_template'))},
        success_criteria=p.success_criteria(), required_systems=p.REQUIRED,
        split_status=split_status, h0_completed=True, h0_passed=not blockers, blockers=blockers,
        runtime=dict(python=sys.version, numpy=version('numpy'), sklearn=version('scikit-learn'), torch=version('torch')),
        final_confirmation_started=False, no_method_or_threshold_changed=True)
    freeze['manifest_digest'] = p.digest(freeze)
    p.write_once(freeze_path, freeze)
    report = ['# W9-H H0 audit', '', '**' + ('PASS' if not blockers else 'BLOCKED — audit completed; confirmation is not ready') + '**', '',
        'Primary: E3-k3/F0. Safety: E3-k3/F2-A. Verifier: E3-k3/F0 then frozen 32-feature logistic at 0.70.',
        '', '## Frozen baseline registry', '',
        f'{len(registry["entries"])} provenance entries across earlier inspected splits and frozen methods. Historical scores are never TEST scores.',
        'Original Standard RAG is checkpoint-distinct from SARA. W8 original SARA/G0 remains mandatory.',
        '', '## Split audit', '', str(split_status), '', '## Blocking dependencies', '']
    report += ['- ' + x for x in blockers] or ['None.']
    report += ['', '## Success criteria', '', json_pretty(p.success_criteria()), '',
        '## Freeze', '', f'`{freeze_path}`', f'File SHA256: `{p.file_hash(freeze_path)}`', '',
        '## Execution limits', '', 'No TEST files, examples or labels were opened. No confirmation generation, new training or threshold selection ran.',
        'H1 prepares a metadata plan. H2 executes only pre-frozen producer commands and ingests their complete query-level exports.',
        'Readiness gaps are material blockers for a fair confirmation, not newly demonstrated scoring bugs in preserved historical results.',
        'TEST-specific adapters and first-pass/SFR materialization are not supplied by legacy FIT/DEV-only runners. Until specified, H2 fails closed.',
        'Do not simply turn the final flag on with the current BLOCKED audit. Do not edit v1 after freezing; complete missing metadata/adapters in a new version before TEST.',
        '', '## Claims hierarchy', '',
        'Confirmed: pending; no confirmation verdict exists.',
        'Development: prior FIT/DEV diagnostics remain development findings, including E3 compression and G2 nested OOF.',
        'Negative/inconclusive: generative extraction brittleness and inconclusive verifier gains are retained.']
    path = root/'audit_report.md'; path.write_text('\n'.join(report)+'\n'); path.chmod(0o444)
    p.write_once(root/'audit_status.json', dict(complete=True, passed=not blockers, blockers=blockers,
        freeze_sha256=p.file_hash(freeze_path), final_confirmation_started=False, test_touched=False,
        baseline_registry_entries=len(registry['entries'])))
    print('H0 completed:', 'PASS' if not blockers else 'BLOCKED', root)
    return freeze


def json_pretty(value):
    import json
    return '```json\n' + json.dumps(value, indent=2) + '\n```'
