"""Read-only scientific audit plus supplemental C interpretation; no model calls."""
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.week11.protocol import load, gate, output, code_identity, scientific_config
from src.week11.cache import digest, atomic, file_hash
from src.week11.planning import plan, pending_reason
from src.week11.budgets import evidence


def main():
    cfg = load()
    gate(cfg, 'C')
    root = output(cfg, 'C')
    manifest = json.loads((root/'manifest.json').read_text())
    identity = manifest['identity']
    assert identity['code_hash'] == digest(code_identity())
    assert identity['config_hash'] == digest(scientific_config(cfg, 'C'))
    assert manifest['relevant_dirty_hashes'] == code_identity()
    rows = []
    keys = set()
    for path in (root/'rows').glob('*.json'):
        if len(path.stem) != 64:
            continue
        obj = json.loads(path.read_text())
        assert obj['identity'] == digest(identity)
        assert obj['checksum'] == digest(obj['row'])
        assert path.stem == digest(obj['key'])
        key = tuple(obj['key'])
        assert key not in keys
        keys.add(key)
        r = obj['row']
        assert r['code_hash'] == identity['code_hash'] and not r['synthetic']
        assert r['total_measured_latency'] is None
        assert r['source_support'] == 1
        assert 0 <= r['task_score'] <= 1
        assert r['budget'] == 'FULL' or r['actual_evidence_tokens'] <= r['budget']
        rows.append(r)
    cohort = json.loads((root/'cohort.json').read_text())['selected']
    assert {(c['dataset'], str(c['query_id'])) for c in cohort} == {
        (c['dataset'], str(c['query_id'])) for c in json.loads((output(cfg, 'B')/'cohort.json').read_text())['selected']}
    expected = set()
    for c in cohort:
        for a in plan(cfg, 'C', c['dataset']):
            if not pending_reason(a, cfg):
                expected.add((c['dataset'], c['query_id'], a['action_id']))
    assert keys == expected and len(keys) == 5550
    baselines = {(r['dataset'], r['query_id']): r for r in rows if r['evidence_granularity'] == 'NONE'}
    for r in rows:
        base = baselines[r['dataset'], r['query_id']]
        delta = r['task_score'] - base['task_score']
        epsilon = 0 if r['dataset'] == 'SciFact' else cfg['epsilon']
        assert abs(delta - r['delta_task_score']) < 1e-12
        assert r['HELP'] == (delta > epsilon) and r['HURT'] == (delta < -epsilon)
        assert r['answer_changed'] == (r['prediction'].strip() != base['prediction'].strip())
        assert r['reader'] == 'Qwen' and r['reader_variant'] == 'PUBLIC_TEXT_READER'
    # Validate cached selections and their source substrings against frozen documents.
    documents = {}
    for path in (ROOT/cfg['cache_root']/'datasets/documents').glob('*.json'):
        if len(path.stem) == 64:
            obj = json.loads(path.read_text()); d = obj['row']
            assert digest(d) == obj['checksum']
            documents[d['dataset'], d['document_id']] = d['text']
    selections = {}
    for path in (ROOT/cfg['cache_root']/'selections').rglob('*.json'):
        if len(path.stem) != 64:
            continue
        obj = json.loads(path.read_text()); s = obj['row']
        assert digest(s) == obj['checksum']
        dataset = obj['key']['dataset']
        assert s['text'] == evidence(s['units'])
        assert s['selected_unit_ids'] == [u['unit_id'] for u in s['units']]
        for u in s['units']:
            text = documents[dataset, u['document_id']]
            assert text[u['start_char']:u['end_char']] == u['source_text']
        assert s['requested_budget'] == 'FULL' or max(s['token_counts'].values()) <= s['requested_budget']
        selections[digest(s)] = s
    for r in rows:
        s = selections[r['selection_hash']]
        assert r['evidence_hash'] == digest(s['text'])
        assert r['selected_unit_ids'] == s['selected_unit_ids']
        assert r['actual_evidence_tokens'] == s['token_counts']['Qwen']
    df = pd.DataFrame(rows)
    summary = pd.read_csv(root/'c_summary_df.csv')
    boot = pd.read_csv(root/'c_pairwise_bootstrap_df.csv')
    assert len(summary) == 111 and summary.n_queries.eq(50).all()
    for _, s in summary.iterrows():
        g = df[(df.dataset == s.dataset) & (df.action_id == s.action_id)]
        assert len(g) == 50 and abs(g.task_score.mean() - s.task_score) < 1e-12
    flags = []
    for r in rows:
        reasons = []
        if r['evidence_granularity'] != 'NONE' and not r['selected_unit_ids']:
            reasons.append('empty_evidence_strict_prefix')
        if r['dataset'] == 'SciFact' and r['predicted_label'] not in cfg['datasets']['SciFact']['labels']:
            reasons.append('invalid_verification_label')
        if r['generated_tokens'] >= cfg['generation']['max_new_tokens']:
            reasons.append('generation_cap_reached')
        if r['unmapped_gold']:
            reasons.append('unmapped_gold_annotation')
        if reasons:
            flags.append({k:r[k] for k in ['dataset','query_id','action_id','evidence_granularity','selector','budget','task_score','HURT']} | {'flags': ';'.join(reasons)})
    pd.DataFrame(flags).to_csv(root/'flagged_queries.csv', index=False)
    empty = df[df.evidence_granularity != 'NONE'].assign(empty=lambda x:x.selected_unit_ids.map(len).eq(0))
    empty_table = empty.groupby(['dataset','evidence_granularity','budget'], sort=False).agg(empty_rows=('empty','sum'),rows=('empty','size'),empty_rate=('empty','mean')).reset_index()
    empty_table.to_csv(root/'empty_evidence_audit.csv', index=False)
    labelled = boot.merge(summary[['dataset','action_id','evidence_granularity','selector','budget']], left_on=['dataset','action_a'], right_on=['dataset','action_id']).drop(columns='action_id')
    labelled = labelled.merge(summary[['dataset','action_id','evidence_granularity','selector','budget']].rename(columns={k:k+'_reference' for k in ['evidence_granularity','selector','budget']}), left_on=['dataset','action_b'], right_on=['dataset','action_id']).drop(columns='action_id')
    labelled.to_csv(root/'paired_comparisons_labelled.csv', index=False)
    audit = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),checks_passed=True,
        rows=len(rows),per_dataset=dict(Counter(r['dataset'] for r in rows)),queries=len(cohort),
        conditions_per_query=37,checksums_verified=len(rows),exact_source_selections_verified=len(selections),
        same_reader_baseline_pairs_verified=len(rows),physical_generation_calls=int(df.physical_lm_calls.sum()),
        empty_intervention_rows=int(empty['empty'].sum()),invalid_label_rows=sum('invalid_verification_label' in r['flags'] for r in flags),
        cap_hit_rows=sum('generation_cap_reached' in r['flags'] for r in flags),
        pending=['LLMLingua-2'],core_code_hash=identity['code_hash'],config_hash=identity['config_hash'],
        status_sha256=file_hash(root/'status.json'),audit_driver_sha256=file_hash(Path(__file__)))
    atomic(root/'readiness_audit.json', audit)
    text = ['# Week 11 C — granularity and budget pilot',
        'All 5,550 executable conditions finished: 50 queries per dataset, 37 Qwen text conditions per query. The 600 configured token-compression conditions remain pending LLMLingua-2 qualification. This is a pilot, not full-dataset confirmation.',
        '## Observations',
        'QASPER benefits most from the larger context in this surface; HotpotQA peaks at a 1,024-token passage condition; SciFact has competitive small sentence budgets. These are within-pilot point estimates, not a universal winner. QA F1 and verification accuracy are kept separate.']
    for dataset, g in summary.groupby('dataset'):
        base = g[g.evidence_granularity == 'NONE'].iloc[0]
        text += [f'## {dataset}', f'No-evidence {"accuracy" if dataset == "SciFact" else "F1"}: **{base.task_score:.6f}**.',
            'Top point estimates (ties retain saved order):',
            g.sort_values('task_score',ascending=False).head(6)[['evidence_granularity','selector','budget','task_score','delta_task_score','HELP','HURT','actual_evidence_tokens']].to_markdown(index=False,floatfmt='.6f')]
        best = g.sort_values('task_score',ascending=False).iloc[0]
        ci = boot[(boot.dataset == dataset) & (boot.action_a == best.action_id) & (boot.action_b == base.action_id)]
        if len(ci):
            c = ci.iloc[0]
            text += [f'Best displayed action versus no evidence: delta {c.point:.6f}, nominal paired 95% CI [{c.ci_low:.6f}, {c.ci_high:.6f}]. Bootstrap: 10,000 resamples, seed 42; paper clusters for QASPER, query/claim clusters otherwise. The displayed best was found on these same 50 queries; its interval does not correct for winner selection or multiple comparisons.']
    text += ['## Flags and limitations',
        f"- {audit['empty_intervention_rows']} intervention rows expose no evidence under the preregistered strict-prefix budget rule. These are preserved and itemized in `empty_evidence_audit.csv`; an oversized first passage can yield an empty block. This is a real limitation of the allocation policy.",
        f"- {audit['invalid_label_rows']} invalid SciFact label rows; {audit['cap_hit_rows']} rows reached the 128-token generation cap. These flags are observational; saved answers and scoring were preserved.",
        '- Previously identified QASPER annotation-mapping gaps remain flagged. SciFact no-evidence claims have null retrieval recall; they still contribute to classification accuracy.',
        '- FULL selectors can expose identical complete evidence and legitimately produce ties. Identical prompt reuse reduces physical calls; it does not establish independent replications.',
        f"- {audit['physical_generation_calls']} physical generation calls served 5,550 logical rows. Recorded selection/generation timings can be cached original measurements. End-to-end latency was not measured and no serving-latency claim is made.",
        '- Exact source support validates exposure text, not answer correctness or semantic faithfulness. Gold annotations enter evaluation, not evidence ranking.',
        '## Full result surface',
        summary[['dataset','evidence_granularity','selector','budget','task_score','delta_task_score','HELP','HURT','actual_evidence_tokens','prompt_tokens']].sort_values(['dataset','evidence_granularity','selector','budget'],key=lambda x:x.astype(str)).to_markdown(index=False,floatfmt='.6f'),
        '## Continuation',
        'Proceed to D with SFR/P0 and all reviewed C textual selections fixed; vary only the reader. No single winning C condition is promoted, and token conditions are explicitly waived as pending. D must reuse C selection identities and same-reader baselines. Existing results and scientific configuration remain unchanged.']
    (root/'INTERPRETATION.md').write_text('\n\n'.join(text)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
