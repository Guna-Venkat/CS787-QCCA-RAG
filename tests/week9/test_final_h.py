"""Outcome-independent W9-H safety, pairing, cost and verdict tests."""
import copy
import json
import subprocess
from pathlib import Path
import pytest
from src.week9.final_h import protocol as p
from src.week9.final_h import analysis as a
from src.week9.final_h import runner


def rows(split='test', ids=('q1', 'q2'), papers=('p1', 'p1')):
    return [dict(question_id=q, paper_id=paper, split=split, evaluation_hash='same', f1=.5,
                 prediction='candidate', initial_draft='draft', total_query_prompt_tokens=200,
                 total_query_generated_tokens=20, num_lm_calls=2, final_prompt_tokens=100)
            for q, paper in zip(ids, papers)]


@pytest.fixture
def frozen(tmp_path):
    m = dict(method=p.method_definition(), success_criteria=p.success_criteria(),
             dependency_hashes={}, h0_passed=True, blockers=[])
    m['manifest_digest'] = p.digest(m)
    p.write_once(tmp_path/'w9h_pretest_freeze_manifest.json', m)
    return tmp_path, m


def test_branch_gagan():
    assert p.branch_guard() == 'gagan'


def test_freeze_required_before_confirmation(tmp_path):
    with pytest.raises(ValueError, match='freeze must exist'):
        runner.confirm(tmp_path, allow_final=True)


def test_primary_exact():
    assert p.method_definition()['primary'] == 'E3-k3 + F0'


def test_sentence_k_exact():
    assert p.method_definition()['sentence_k'] == 3


def test_f2_prompt_frozen():
    old = p.read(p.F_ROOT/'prompts.json')
    assert p.method_definition()['prompts']['F2-A'] == old['F2-A']


def test_g2_32_features():
    schema = p.read(p.G_ROOT/'final_fit/feature_schema.json')
    assert len(schema['feature_columns']) == schema['feature_count'] == 32


def test_g2_threshold_exact():
    assert p.read(p.G_ROOT/'final_fit/threshold.json')['threshold'] == .70


def test_registry_fields():
    from src.week9.final_h.registry import FIELDS
    historical = p.read(p.ROOT/'results/week9/baselines/dev/baseline_manifest.json')
    assert any(e['baseline_id'] == 'original_SARA_G0' and e['status'] == 'AVAILABLE' for e in historical['entries'])
    assert {'query_ids', 'paper_ids', 'artifact_hash', 'provenance_status', 'model_revision'} <= set(FIELDS)


def test_query_mismatch_cannot_pair():
    with pytest.raises(ValueError, match='differ'):
        a.checked_pair(rows(), rows(ids=('q1', 'q3')))


def test_split_mismatch_cannot_pair():
    with pytest.raises(ValueError, match='split/scorer'):
        a.checked_pair(rows(), rows(split='dev'))


def test_paper_mismatch_cannot_pair():
    with pytest.raises(ValueError, match='differ'):
        a.checked_pair(rows(), rows(papers=('p1', 'p2')))


def test_h0_blocked_before_any_test_io(frozen, monkeypatch):
    root, m = frozen
    path = root/'w9h_pretest_freeze_manifest.json'; path.chmod(0o644); path.unlink()
    m.pop('manifest_digest'); m['h0_passed'] = False; m['blockers'] = ['Missing TEST producers']
    m['manifest_digest'] = p.digest(m); p.write_once(path, m)
    monkeypatch.setattr(runner, 'read_jsonl', lambda *x: pytest.fail('TEST opened before H0 passed'))
    with pytest.raises(ValueError, match='H0 BLOCKED'):
        runner.confirm(root, allow_final=True)


def test_notebook_default_never_generates(monkeypatch):
    nb = p.read(p.ROOT/'notebooks/week9/week9h_final_evaluation.ipynb')
    code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    env = {}; exec(code[0], env)
    calls = []; env['stage'] = lambda stage, *args: calls.append(stage)
    for cell in code[1:]:
        exec(cell, env)
    assert calls == ['h0']
    assert not env['RUN_FINAL_CONFIRMATION'] and not env['RUN_ROBUSTNESS']


def test_test_configuration_selection_forbidden(tmp_path):
    c = p.read(p.ROOT/'configs/week9/w9h_final.json'); c['sentence_k'] = 6
    path = tmp_path/'config.json'; path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match='choices changed'):
        p.load_config(path)


def test_success_criteria_saved_before_confirmation(frozen):
    root, m = frozen
    assert p.validate_freeze(root)['success_criteria'] == p.success_criteria()


def test_freeze_cannot_overwrite(frozen):
    root, m = frozen
    with pytest.raises(ValueError, match='Immutable'):
        p.write_once(root/'w9h_pretest_freeze_manifest.json', dict(m, changed=True))


def test_freeze_tamper_detected(frozen):
    root, m = frozen; path = root/'w9h_pretest_freeze_manifest.json'
    path.chmod(0o644); path.write_text(json.dumps(dict(m, h0_passed=False))); path.chmod(0o444)
    with pytest.raises(ValueError, match='digest differs'):
        p.validate_freeze(root)


def test_bootstrap_is_paper_clustered():
    left = rows(ids=('1', '2', '3'), papers=('a', 'a', 'b'))
    right = copy.deepcopy(left)
    left[0]['f1'] = 1; left[1]['f1'] = 1
    result = a.checked_pair(left, right)
    assert result['cluster'] == 'paper_id' and result['resamples'] == 10000 and result['seed'] == 42
    assert result['point_estimate'] == pytest.approx(1/3)


def test_g2_keep_pays_all_generation():
    primary = rows(); gate = copy.deepcopy(primary)
    for r in gate:
        r.update(g2_accept=False, prediction='draft', verifier_latency=.001)
    a.verify_post_generation_costs(primary, gate)
    gate[0]['total_query_prompt_tokens'] = 100
    with pytest.raises(ValueError, match='refund'):
        a.verify_post_generation_costs(primary, gate)


@pytest.mark.parametrize('candidate', [p.SAFE, p.VERIFY])
def test_secondary_cannot_promote(candidate, tmp_path):
    c = p.read(p.ROOT/'configs/week9/w9h_final.json'); c['primary_system'] = candidate
    path = tmp_path/'config.json'; path.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        p.load_config(path)


def test_historical_results_not_confirmation():
    with pytest.raises(ValueError, match='TEST only'):
        a.evaluate([], {}, 'fit')


def verdict_inputs():
    systems = [p.PRIMARY] + p.STRONG + ['D1']
    quality = [dict(system=s, f1=.5) for s in systems]
    cost = [dict(system=s, total_query_prompt_tokens=100) for s in systems]
    comparisons = [dict(left=p.PRIMARY, right=s, status='COMPATIBLE', ci_low=-.01, ci_high=.01, family='primary') for s in p.STRONG + ['D1']]
    return quality, cost, comparisons, dict(evidence_suppression=.95)


def test_verdict_uses_both_strong_comparators():
    q, c, comparisons, compression = verdict_inputs()
    comparisons[0]['ci_low'] = .001
    result = a.verdict(q, c, comparisons, compression, p.success_criteria())
    assert result['final_classification'] == 'MIXED_OR_NEGATIVE'
    comparisons[1]['ci_low'] = .001
    q[0]['f1'] = .51
    assert a.verdict(q, c, comparisons, compression, p.success_criteria())['quality_hit']


def test_crossing_zero_not_equivalence():
    q, c, comparisons, compression = verdict_inputs()
    assert not a.verdict(q, c, comparisons, compression, p.success_criteria())['efficiency_mechanistic_hit']
    comparisons[-1].update(ci_low=-.002, ci_high=.002)
    assert a.verdict(q, c, comparisons, compression, p.success_criteria())['efficiency_mechanistic_hit']


def test_missing_strong_comparator_not_dropped():
    q, c, comparisons, compression = verdict_inputs()
    with pytest.raises(ValueError, match='missing/incompatible'):
        a.verdict(q, c, comparisons[1:], compression, p.success_criteria())


def test_original_sources_unmodified():
    changed = subprocess.check_output(['git', 'diff', '--name-only'], cwd=p.ROOT, text=True).splitlines()
    forbidden = [name for name in changed if name.startswith(('src/week9/', 'results/week9/'))
                 and '/final_h/' not in name and '/w9h_final/' not in name and '/final_fit/' not in name]
    assert forbidden == []


def test_pareto_requires_both_dimensions():
    result = a.frontier([dict(system='a', f1=.5, tokens=100), dict(system='b', f1=.6, tokens=200),
                         dict(system='c', f1=.4, tokens=150)], 'tokens')
    assert [r['frontier'] for r in result] == [True, True, False]


def test_g2_export_zero_new_fits():
    m = p.read(p.G_ROOT/'final_fit/training_manifest.json')
    assert m['new_estimator_fits'] == 0 and not m['heldout_evaluation_performed']
    assert p.file_hash(p.G_ROOT/'final_fit/logistic_pipeline.joblib') == m['source_oof_model_hash']


def test_verifier_never_sees_labels():
    from src.week9.action_verification_g.features import online_features
    with pytest.raises(ValueError):
        online_features(dict(gold_answer='forbidden'))


def test_h2_explicit_flag_required_before_io(tmp_path):
    with pytest.raises(ValueError, match='disabled'):
        runner.confirm(tmp_path)


def test_h3_requires_seal(frozen):
    root, _ = frozen
    with pytest.raises(FileNotFoundError):
        a.report(root)


def test_complete_synthetic_report_is_sealed_and_repeatable(tmp_path, monkeypatch):
    """Fabricated software fixture only; no real TEST input or generation."""
    freeze = dict(manifest_digest='SYNTHETIC', success_criteria=p.success_criteria())
    all_rows = []
    for name in p.REQUIRED:
        for i in range(6):
            f1 = .6 if name in (p.PRIMARY, p.VERIFY, 'D1') else .5
            all_rows.append(dict(system=name, question_id=str(i), paper_id=str(i//2), split='test',
                evaluation_hash='SYNTHETIC', prediction='draft' if name == 'D0' else 'candidate', initial_draft='draft',
                f1=f1, exact_match=0., delta_f1=f1-.5, help=f1-.5 > .01, hurt=False,
                answer_changed=name != 'D0', evidence_tokens=2000 if name == 'D1' else 100,
                final_prompt_tokens=100, total_query_prompt_tokens=200, total_query_generated_tokens=20,
                num_lm_calls=2, g2_accept=True, verifier_latency=.001))
    p.write_once(tmp_path/'sealed_predictions.json', all_rows)
    p.write_once(tmp_path/'w9h_pretest_freeze_manifest.json', dict(synthetic=True))
    p.write_once(tmp_path/'h2_seal.json', dict(complete=True, freeze_digest='SYNTHETIC', artifact_hashes={}))
    monkeypatch.setattr(p, 'validate_freeze', lambda *args, **kwargs: freeze)
    first = a.report(tmp_path); second = a.report(tmp_path)
    assert first == second and first['primary_system'] == p.PRIMARY
    assert first['quality_hit']
    output = tmp_path/'report'
    assert (output/'aggregate_seal.json').stat().st_mtime <= (output/'error_analysis.csv').stat().st_mtime
    assert (output/'quality_cost.png').is_file() and (output/'quality_safety.png').is_file()
    report = (output/'FINAL_W9_REPORT.md').read_text()
    assert 'FIT/DEV development findings' in report and 'TEST confirmation' in report
    p.verify_hashes(p.read(output/'manifest.json')['artifact_hashes'])
