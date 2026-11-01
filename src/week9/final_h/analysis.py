"""Common scoring, aligned paper-cluster comparisons and preregistered verdict."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import protocol as p
from ..baseline_registry import aligned_rows
from ..evaluation import paired_paper_bootstrap, annotate_outcome

COSTS = ['final_prompt_tokens', 'total_query_prompt_tokens', 'total_query_generated_tokens',
         'num_lm_calls', 'evidence_tokens', 'retrieval_query_time', 'selection_latency',
         'extraction_latency', 'revision_latency', 'verifier_latency',
         'sentence_index_conceptual_seconds', 'query_embedding_conceptual_seconds',
         'embedding_lookup_compute_latency', 'sentence_ranking_latency', 'measured_end_to_end_latency']


def average(rows, key):
    values = [r.get(key) for r in rows]
    return float(np.mean(values)) if values and all(v is not None and np.isfinite(v) for v in values) else None


def checked_pair(a, b):
    aligned_rows(a, b)
    return paired_paper_bootstrap(a, b, n=10000, seed=42)


def verify_post_generation_costs(primary, verify):
    aligned_rows(primary, verify)
    byid = {str(r['question_id']): r for r in primary}
    for r in verify:
        f = byid[str(r['question_id'])]
        for name in ('total_query_prompt_tokens', 'total_query_generated_tokens', 'num_lm_calls', 'final_prompt_tokens'):
            p.require(f.get(name) is not None and r.get(name) == f[name], 'G2 cannot refund or invent F0 generation costs: ' + name)
        p.require(r.get('verifier_latency') is not None and r['verifier_latency'] >= 0, 'Verifier latency must be measured')
        expected = f['prediction'] if r['g2_accept'] else f['initial_draft']
        p.require(r['prediction'] == expected, 'G2 must choose only KEEP_D0 or ACCEPT_THIS_F0')


def evaluate(raw, labels, split):
    """Ignore producer F1 values; recompute every system with one frozen evaluator."""
    p.require(split == 'test', 'Confirmation is TEST only; FIT results are development')
    out = []
    ids = [str(r['question_id']) for r in raw]
    p.require(len(ids) == len(set(ids)) and set(ids) == set(labels), 'Incomplete/duplicate confirmation membership')
    for row in raw:
        q = str(row['question_id']); label = labels[q]
        p.require(str(row['paper_id']) == str(label['paper_id']) and row.get('split') == split, 'Paper/split mismatch')
        p.require(isinstance(row.get('prediction'), str) and isinstance(row.get('initial_draft'), str), 'Missing real prediction/draft')
        scored = annotate_outcome(row, dict(initial_draft=row['initial_draft']), label['evaluation'])
        scored.update(question_id=q, paper_id=str(row['paper_id']), split=split,
                      evaluation_hash=p.file_hash(p.ROOT/'src/week6/evaluation.py'))
        out.append(scored)
    return sorted(out, key=lambda r: r['question_id'])


def savings(a, b):
    return 1-a/b if a is not None and b is not None and b > 0 else None


def verdict(quality, cost, comparisons, compression, criteria):
    p.require(criteria == p.success_criteria(), 'Verdict must use the pretest criteria exactly')
    q = {r['system']: r for r in quality}; c = {r['system']: r for r in cost}
    effects = {(r['left'], r['right']): r for r in comparisons if r.get('status') == 'COMPATIBLE'}
    needed = [(p.PRIMARY, s) for s in p.STRONG + ['D1']]
    p.require(all(key in effects for key in needed), 'Cannot classify with missing/incompatible strong comparators')
    quality_hit = all(effects[(p.PRIMARY, s)]['ci_low'] > 0 and q[p.PRIMARY]['f1'] > q[s]['f1'] for s in p.STRONG)
    losses = {s: q[p.PRIMARY]['f1']-q[s]['f1'] for s in p.STRONG}
    cost_savings = {s: savings(c[p.PRIMARY]['total_query_prompt_tokens'], c[s]['total_query_prompt_tokens']) for s in p.STRONG}
    efficiency = all(losses[s] >= -.003 and cost_savings[s] is not None and cost_savings[s] >= .15 for s in p.STRONG)
    d = effects[(p.PRIMARY, 'D1')]
    mechanistic = (compression['evidence_suppression'] is not None and compression['evidence_suppression'] >= .90
                   and d['ci_low'] >= -.003 and d['ci_high'] <= .003)
    category = 'QUALITY_HIT' if quality_hit else 'EFFICIENCY_OR_MECHANISTIC_HIT' if efficiency or mechanistic else 'MIXED_OR_NEGATIVE'
    return dict(primary_system=p.PRIMARY, confirmation_split='test', baseline_comparators=p.STRONG,
        primary_quality_effects=[r for r in comparisons if r['family'] == 'primary'],
        efficiency_effects=dict(f1_deltas=losses, cumulative_prompt_reductions=cost_savings,
                               raw_evidence_compression=compression),
        quality_hit=quality_hit, efficiency_mechanistic_hit=efficiency or mechanistic,
        final_classification=category,
        rationale='Applied the frozen conjunction across both strong comparators and the separate conservative D1 compression criterion.',
        limitations=['FIT/DEV findings are development evidence', 'Evidence exposure reduction is not an equal reduction in end-to-end compute',
                     'One frozen canonical generator; no new robustness panel', 'Raw bootstrap CIs; no multiplicity-adjusted significance claim'],
        test_used_once_under_frozen_configuration=True)


def frontier(rows, x):
    result = []
    for row in rows:
        valid = row.get(x) is not None and np.isfinite(row[x])
        dominated = valid and any(other.get(x) is not None and other[x] <= row[x] and other['f1'] >= row['f1']
            and (other[x] < row[x] or other['f1'] > row['f1']) for other in rows)
        result.append(dict(row, frontier=not dominated if valid else None, cost_dimension=x))
    return result


def report(root):
    root = Path(root); freeze = p.validate_freeze(root)
    sealed = p.read(root/'h2_seal.json')
    p.require(sealed['complete'] and sealed['freeze_digest'] == freeze['manifest_digest'], 'H3 needs sealed complete H2 predictions')
    p.verify_hashes(sealed['artifact_hashes'])
    dest = root/'report'
    if (dest/'manifest.json').exists():
        p.verify_hashes(p.read(dest/'manifest.json')['artifact_hashes'])
        return p.read(dest/'final_verdict.json')
    p.require(not dest.exists(), 'Preserve incomplete H3 output; do not silently overwrite')
    dest.mkdir()
    rows = p.read(root/'sealed_predictions.json')
    groups = {s: [r for r in rows if r['system'] == s] for s in sorted({r['system'] for r in rows})}
    p.require(set(p.REQUIRED) <= groups.keys(), 'Required frozen comparison systems missing')
    d0 = {r['question_id']: r for r in groups['D0']}
    for name, group in groups.items():
        aligned_rows(group, groups['D0'])
        p.require(all(r['initial_draft'] == d0[r['question_id']]['prediction'] for r in group), 'Systems do not share the canonical D0')
    verify_post_generation_costs(groups[p.PRIMARY], groups[p.VERIFY])
    quality = []; cost = []; safety = []
    base = groups[p.PRIMARY]
    for name, group in groups.items():
        quality.append(dict(system=name, split='test', n_queries=len(group), f1=average(group, 'f1'), exact_match=average(group, 'exact_match'),
            table='primary' if name in ['D0', 'BM25_k8', 'semantic_k4', 'semantic_k6', 'semantic_k8', 'original_SARA_G0', p.PRIMARY, p.SAFE, p.VERIFY] else 'extended'))
        cost.append(dict(system=name, **{k: average(group, k) for k in COSTS},
                         cost_note='Null means unmeasured/unestablished. Cached work retains conceptual attribution; no summed end-to-end latency.'))
        delta = np.array([r['delta_f1'] for r in group])
        entry = dict(system=name, help=float(np.mean(delta > .01)), hurt=float(np.mean(delta < -.01)),
            answer_changed=average(group, 'answer_changed'), preservation=1-average(group, 'answer_changed'),
            mean_positive_delta=float(delta[delta > .01].mean()) if any(delta > .01) else None,
            mean_negative_delta=float(delta[delta < -.01].mean()) if any(delta < -.01) else None)
        if name in (p.SAFE, p.VERIFY):
            other = {r['question_id']: r for r in group}
            help_ids = [r['question_id'] for r in base if r['help']]
            hurt_ids = [r['question_id'] for r in base if r['hurt']]
            entry.update(f0_help_retention=float(np.mean([other[q]['help'] for q in help_ids])) if help_ids else None,
                f0_hurt_rescue=float(np.mean([not other[q]['hurt'] for q in hurt_ids])) if hurt_ids else None,
                retention_definition='Still HELP vs D0; rescue means no longer HURT vs D0')
            if name == p.VERIFY:
                entry['f0_hurt_rejection'] = float(np.mean([not other[q]['g2_accept'] for q in hurt_ids])) if hurt_ids else None
        safety.append(entry)
    comparisons = []
    for left, right, family in p.COMPARISONS:
        try:
            result = checked_pair(groups[left], groups[right])
            comparisons.append(dict(left=left, right=right, family=family, status='COMPATIBLE',
                delta_f1=result['point_estimate'], ci_low=result['ci95'][0], ci_high=result['ci95'][1],
                cluster='paper_id', resamples=10000, seed=42))
        except (KeyError, ValueError) as exc:
            comparisons.append(dict(left=left, right=right, family=family, status='INCOMPATIBLE', reason=str(exc)))
    c = {r['system']: r for r in cost}
    raw = average(groups['D1'], 'evidence_tokens'); compact = average(base, 'evidence_tokens')
    compression = dict(raw_selected_chunk_tokens=raw, e3_evidence_tokens=compact,
        retained_fraction=compact/raw if compact is not None and raw is not None and raw > 0 else None,
        evidence_suppression=savings(compact, raw),
        final_prompt_reduction=savings(c[p.PRIMARY]['final_prompt_tokens'], c['D1']['final_prompt_tokens']),
        cumulative_prompt_reduction=savings(c[p.PRIMARY]['total_query_prompt_tokens'], c['D1']['total_query_prompt_tokens']),
        raw_lm_calls=c['D1']['num_lm_calls'], compact_lm_calls=c[p.PRIMARY]['num_lm_calls'],
        f1_delta_vs_d1=average(base, 'f1')-average(groups['D1'], 'f1'),
        help_rate_delta_vs_d1=average(base, 'help')-average(groups['D1'], 'help'),
        hurt_rate_delta_vs_d1=average(base, 'hurt')-average(groups['D1'], 'hurt'),
        interpretation='Evidence exposure, final prompt, cumulative prompt and LM calls are different quantities.')
    final = verdict(quality, cost, comparisons, compression, freeze['success_criteria'])
    final['safety_effects'] = [r for r in safety if r['system'] in (p.PRIMARY, p.SAFE, p.VERIFY)]
    final['pretest_manifest_sha256'] = p.file_hash(root/'w9h_pretest_freeze_manifest.json')
    joined = [dict(q, **{k: v for k, v in c[q['system']].items() if k != 'system'}) for q in quality]
    safety_map = {r['system']: r for r in safety}
    pf = frontier(joined, 'total_query_prompt_tokens')
    sf = frontier([dict(q, hurt=safety_map[q['system']]['hurt']) for q in quality], 'hurt')
    for filename, data in [('summary_quality.csv', quality), ('summary_cost.csv', cost), ('summary_safety.csv', safety),
                           ('paired_bootstrap.csv', comparisons), ('pareto_quality_cost.csv', pf), ('pareto_quality_safety.csv', sf)]:
        pd.DataFrame(data).to_csv(dest/filename, index=False)
    qframe = pd.DataFrame(quality)
    qframe[qframe.table == 'primary'].to_csv(dest/'table1_primary.csv', index=False)
    qframe[qframe.table == 'extended'].to_csv(dest/'table2_extended.csv', index=False)
    pd.DataFrame(rows).to_csv(dest/'per_query_predictions.csv', index=False)
    p.write_once(dest/'evidence_compression.json', compression)
    p.write_once(dest/'final_verdict.json', final)
    # Seal aggregate metrics before selecting any qualitative examples.
    p.write_once(dest/'aggregate_seal.json', dict(freeze_digest=freeze['manifest_digest'],
        artifact_hashes={str(path): p.file_hash(path) for path in dest.iterdir() if path.is_file()}))
    examples = qualitative(groups)
    pd.DataFrame(examples, columns=['category', 'question_id', 'paper_id', 'question', 'y0', 'evidence',
                                  'f0_answer', 'f2_answer', 'g2_accept', 'evaluation']).to_csv(dest/'error_analysis.csv', index=False)
    plots(dest, pf, sf)
    write_report(dest, final, quality, cost, safety, comparisons, compression, freeze)
    p.write_once(dest/'manifest.json', dict(complete=True, split='test', freeze_digest=freeze['manifest_digest'],
        no_method_changes=True, robustness_executed=False, strata='Skipped: no frozen answer-type/difficulty schema supplied',
        artifact_hashes={str(path): p.file_hash(path) for path in dest.iterdir() if path.is_file()}))
    return final


def qualitative(groups):
    maps = {name: {r['question_id']: r for r in rows} for name, rows in groups.items()}
    categories = {k: [] for k in ('W9 helps D0', 'W9 hurts D0', 'F2 rescues F0 harm', 'G2 rejects F0 harm', 'G2 misses F0 help')}
    for q in sorted(maps[p.PRIMARY], key=lambda q: p.digest([q, maps[p.PRIMARY][q]['paper_id']])):
        f = maps[p.PRIMARY][q]; safe = maps[p.SAFE][q]; gate = maps[p.VERIFY][q]
        tests = [f['help'], f['hurt'], f['hurt'] and not safe['hurt'], f['hurt'] and not gate['g2_accept'], f['help'] and not gate['g2_accept']]
        for category, yes in zip(categories, tests):
            if yes and len(categories[category]) < 5:
                categories[category].append(dict(category=category, question_id=q, paper_id=f['paper_id'],
                    question=f.get('question'), y0=f['initial_draft'], evidence=f.get('evidence_text'),
                    f0_answer=f['prediction'], f2_answer=safe['prediction'], g2_accept=gate['g2_accept'],
                    evaluation=json.dumps(f.get('evaluation', {}))))
    return [item for values in categories.values() for item in values]


def plots(dest, quality_cost, quality_safety):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, data, x, label in [('quality_cost', quality_cost, 'total_query_prompt_tokens', 'Mean cumulative query prompt tokens'),
                                 ('quality_safety', quality_safety, 'hurt', 'HURT rate vs D0')]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for row in data:
            if row.get(x) is None:
                continue
            ax.scatter(row[x], row['f1'])
            ax.annotate(row['system'], (row[x], row['f1']), xytext=(4, 4), textcoords='offset points', fontsize=8)
        ax.set(xlabel=label, ylabel='Answer F1', title='Frozen TEST confirmation')
        fig.tight_layout(); fig.savefig(dest/(name+'.png'), dpi=180); plt.close(fig)


def write_report(dest, final, quality, cost, safety, comparisons, compression, freeze):
    table = lambda data: pd.DataFrame(data).to_markdown(index=False)
    sections = [
        ('Executive conclusion', final['final_classification'] + '. ' + final['rationale']),
        ('Frozen protocol', 'TEST confirmation under immutable pretest manifest. Paper-cluster bootstrap: 10,000 resamples, seed 42. No tuning or retrospective promotion.'),
        ('Baselines', 'Table 1 is the compact primary comparison; Table 2 contains separately labelled extended ablations. Earlier FIT/DEV artifacts are never confirmation rows.'),
        ('W9 primary system', 'Canonical SARA first pass; FIXED-5/SEMANTIC@20; DraftSemantic coarse K=2; exact query-only MiniLM top-3 sentences in source order; F0 fresh regeneration inside unchanged SARA context.'),
        ('Final quality results', table(quality)), ('Statistical comparisons', table(comparisons)),
        ('Cost results', table(cost)), ('Safety / HELP / HURT results', table(safety)),
        ('Evidence compression analysis', json.dumps(compression, indent=2)),
        ('Controller result', 'G2 is secondary and post-generation. KEEP pays the full F0 cost. Its deployment threshold remains 0.70.'),
        ('Mechanistic findings', 'FIT/DEV development findings: sentence suppression produced compact exact-source evidence; sentence-count surfaces and oracle ceilings are retrospective, not deployable policies. Confirmation comparisons above determine transfer.'),
        ('Negative findings', 'Earlier generative extraction was brittle and did not establish a deployable gain. Nested FIT verifier estimates did not convincingly beat F0/F2-A; pooled deployment FIT scores are not honest OOF estimates.'),
        ('Error analysis', 'Deterministic category samples saved only after aggregate seal; explanatory only. No method changes permitted.'),
        ('Final hit/fail classification', final['final_classification']),
        ('Limitations', '\n'.join('- '+x for x in final['limitations'])),
        ('What W9 establishes', 'Only the measured frozen-system effects, costs and intervals on this confirmation split; development findings remain separately labelled.'),
        ('What W9 does NOT establish', 'No general RAG superiority, causal evidence identification, dynamic-k success, or proportional end-to-end speedup from evidence suppression.'),
        ('Recommended next step after Week 9', 'Close Week 9 with the frozen result. Carry limitations into a separately preregistered Week 10 study; do not retune on this TEST split.')]
    (dest/'FINAL_W9_REPORT.md').write_text('# Final Week 9 report\n\n'+'\n\n'.join('## '+name+'\n\n'+body for name, body in sections)+'\n')
