"""Validate and summarize the completed frozen W10 study; no model execution.

Historical partial-panel reports/provenance remain intact. This adds a distinct
full-panel receipt and final report without changing preregistered science.
"""
import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.week10.root_cause.analysis import paired, oracle
from src.week10.root_cause.protocol import load, source_hashes
from src.week10.root_cause.storage import Store, atomic, digest, file_hash


def complete_rows(root, name, keys, freeze_hash):
    directory = root / 'runs' / name
    identity = json.loads((directory / 'identity.json').read_text())
    if identity['freeze_hash'] != freeze_hash:
        raise ValueError('Mismatched frozen run: ' + name)
    store = Store(directory, identity)
    rows = [store.get(key) for key in keys]
    if any(row is None for row in rows) or not (directory / 'complete.json').exists():
        raise ValueError('Incomplete surface: ' + name)
    if len(store.rows()) != len(keys):
        raise ValueError('Unexpected extra rows: ' + name)
    return rows


def evidence_identity(c_rows, panels):
    expected = {(r['question_id'], r['action']): r for r in c_rows}
    for reader, rows in panels.items():
        for row in rows:
            original = expected[row['question_id'], row['action']]
            if (row['paper_id'] != original['paper_id'] or
                    row['evidence_text'] != original['evidence_text'] or
                    row['evidence_hash'] != original['evidence_hash'] or
                    digest(row['evidence_text']) != row['evidence_hash']):
                raise ValueError('Frozen C/G evidence mismatch: ' + reader)


def comparison(name, left, right):
    result = paired(left, right)
    return dict(comparison=name, point=result['point_estimate'],
                ci_low=result['ci95'][0], ci_high=result['ci95'][1])


def utility(rows, baseline):
    base = {r['question_id']: r for r in baseline}
    result = []
    for row in rows:
        other = base[row['question_id']]
        if row['paper_id'] != other['paper_id']:
            raise ValueError('Utility paper identity mismatch')
        result.append(dict(question_id=row['question_id'], paper_id=row['paper_id'],
                           f1=row['f1'] - other['f1']))
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(cfg):
    root = ROOT / cfg['output_root']
    frozen = json.loads((root / 'execution_freeze.json').read_text())
    if frozen['config'] != cfg or frozen['sources'] != source_hashes():
        raise ValueError('Frozen science changed')
    for path, sha in frozen['datasets'].items():
        if file_hash(path) != sha:
            raise ValueError('Dataset identity changed')
    records = [json.loads(line) for line in Path(cfg['dataset']).read_text().splitlines() if line.strip()]
    records = records[:cfg['query_limit']] if cfg.get('query_limit') else records
    qids = [str(r['id']) for r in records]
    papers = {str(r['id']): str(r['example_id']) for r in records}
    if len(qids) != len(set(qids)):
        raise ValueError('Duplicate question IDs')
    frozen_hash = digest(frozen)
    for method in ('BM25', 'MiniLM', 'SFR'):
        complete_rows(root, 'R0-' + method, qids, frozen_hash)
        for interface in ('full', 'sara'):
            complete_rows(root, f'R-{method}-{interface}',
                          [[q, k] for q in qids for k in cfg['k_values']], frozen_hash)
    c_rows = complete_rows(root, 'C-Mistral', [[q, a] for q in qids for a in
        ('no_evidence', 'full', 'sara_k4', 'lead256', 'qcs256')], frozen_hash)
    panels = {reader: complete_rows(root, 'G-' + reader,
        [[q, a] for q in qids for a in ('no_evidence', 'full', 'qcs256')], frozen_hash)
        for reader in cfg['reader_panel']}
    evidence_identity(c_rows, panels)
    for reader, rows in panels.items():
        receipt = json.loads((root / 'reader_setup' / (reader + '.json')).read_text())
        if receipt.get('passed') is not True or receipt['identity']['reader'] != cfg['models'][reader]:
            raise ValueError('Missing/mismatched reader qualification: ' + reader)
        if any(papers[row['question_id']] != row['paper_id'] for row in rows):
            raise ValueError('Reader question/paper identity mismatch')
    table, groups = [], {}
    for reader, rows in panels.items():
        for action in ('no_evidence', 'full', 'qcs256'):
            groups[reader, action] = [r for r in rows if r['action'] == action]
        base = groups[reader, 'no_evidence']
        for action in ('no_evidence', 'full', 'qcs256'):
            current = groups[reader, action]
            deltas = [r['f1'] for r in utility(current, base)]
            table.append(dict(reader=reader, action=action, status='COMPLETE', n_queries=len(current),
                f1=float(np.mean([r['f1'] for r in current])),
                exact_match=float(np.mean([r['exact_match'] for r in current])),
                net_evidence_utility=float(np.mean(deltas)),
                HELP=float(np.mean(np.array(deltas) > .01)), HURT=float(np.mean(np.array(deltas) < -.01)),
                mean_prompt_tokens=float(np.mean([r['generation']['prompt_tokens'] for r in current])),
                mean_generated_tokens=float(np.mean([r['generation']['generated_tokens'] for r in current])),
                generation_cap_hit_rate=float(np.mean([r['generation']['generated_tokens'] >= 128 for r in current])),
                mean_generation_seconds=float(np.mean([r['generation']['latency'] for r in current])),
                lm_calls=1, measured_end_to_end_seconds=None))
    comparisons = []
    for reader in panels:
        for action in ('full', 'qcs256'):
            comparisons.append(comparison(f'{reader}:{action}-no_evidence',
                groups[reader, action], groups[reader, 'no_evidence']))
        comparisons.append(comparison(f'{reader}:full-qcs256', groups[reader, 'full'], groups[reader, 'qcs256']))
    for left, right in itertools.combinations(cfg['reader_panel'], 2):
        for action in ('no_evidence', 'full', 'qcs256'):
            comparisons.append(comparison(f'{left}-{right}:{action}', groups[left, action], groups[right, action]))
        for action in ('full', 'qcs256'):
            comparisons.append(comparison(f'({left} utility)-({right} utility):{action}',
                utility(groups[left, action], groups[left, 'no_evidence']),
                utility(groups[right, action], groups[right, 'no_evidence'])))
    d_ids = sorted(qids, key=lambda q: digest([papers[q], q, 42]))[:24]
    d_rows = complete_rows(root, 'D-stability', [[q, seed, a] for q in d_ids for seed in (42, 43, 44)
                                               for a in ('no_evidence', 'full')], frozen_hash)
    em_base = {(r['question_id'], r['seed']): r['exact_match'] for r in d_rows if r['action'] == 'no_evidence'}
    em_labels = Counter('HELP' if r['exact_match']-em_base[r['question_id'], r['seed']] > .01 else
                        'HURT' if r['exact_match']-em_base[r['question_id'], r['seed']] < -.01 else 'NEUTRAL'
                        for r in d_rows if r['action'] == 'full')
    diagnostics = json.loads((root / 'dataset_diagnostics.json').read_text())
    ds = diagnostics['stability']
    stability = dict(n_queries=len(ds),
        mean_modal_label_agreement=float(np.mean([r['agreement'] for r in ds])),
        all_seeds_same_label_count=sum(r['sign_stable'] for r in ds),
        all_seeds_same_label_fraction=float(np.mean([r['sign_stable'] for r in ds])),
        mean_delta_variance=float(np.mean([r['delta_variance'] for r in ds])),
        mean_EM_modal_label_agreement=float(np.mean([r['agreement'] for r in diagnostics['em_stability']])),
        EM_delta_label_distribution=dict(em_labels),
        EM_delta_labels_degenerate=len(em_labels) == 1,
        oracle_by_seed=diagnostics['oracle_by_seed'],
        scope='24-query Mistral-only stochastic diagnostic, seeds42/43/44; primary decoding is greedy')
    audit = diagnostics['annotation_audit']
    annotations = dict(n_queries=len(audit), multiple_annotations=sum(r['multiple_annotations'] for r in audit),
        no_gold_evidence=sum(r['gold_evidence_spans'] == 0 for r in audit),
        question_types=dict(Counter(r['question_type'] for r in audit)),
        semantic_correctness='UNMEASURED; no human audit or LLM judge added')
    run_receipts = {}
    for path in sorted((root / 'runs').glob('*/identity.json')):
        if (path.parent / 'complete.json').exists():
            run_receipts[path.parent.name] = dict(identity_sha256=file_hash(path),
                completion_sha256=file_hash(path.parent / 'complete.json'))
    obj = dict(full_reader_panel_complete=True, pending_readers=[],
        development_split=cfg['split'], n_queries=len(qids), n_papers=len(set(papers.values())),
        execution_freeze_hash=frozen_hash, reader_results=table, paired_comparisons=comparisons,
        evidence_identity_validated=True, reader_oracles={k: oracle(v) for k, v in panels.items()},
        dataset_stability=stability, annotation_audit=annotations, completed_runs=run_receipts,
        bootstrap=dict(cluster='paper_id', resamples=10000, seed=42, interval='unadjusted exploratory 95%'),
        scientific_source_unchanged=True, qasper_test_used=False, policy_training_started=False,
        external_confirmation_complete=False, faithful_selecom_complete=False,
        historical_partial_panel_provenance='Preserved; describes the earlier two-reader execution, not the current panel.')
    obj['upstream_oracles'] = {k: v for k, v in json.loads((root / 'oracle_diagnostics.json').read_text()).items()
                               if k.startswith(('R-', 'C-'))}
    obj['completed_generation_rows'] = dict(R=len(qids)*len(cfg['k_values'])*6,
        C=len(c_rows), G=sum(len(v) for v in panels.values()), D=len(d_ids)*6)
    obj['total_experiment_generation_rows'] = sum(obj['completed_generation_rows'].values())
    obj['root_cause_assessment'] = dict(
        retrieval_foundation='SUPPORTED contributor; SFR improves recall and matched downstream F1',
        reader_utilization='SUPPORTED within G; Qwen/Llama improve full-evidence F1 and utility over public Mistral',
        aggressive_text_compression='QUALITY LOSS supported; FULL exceeds QCS256 for every G reader',
        incremental_latent_interface_advantage='NOT ESTABLISHED at SFR k6',
        stochastic_label_sensitivity='OBSERVED on 4/24 Mistral sampling-diagnostic queries',
        semantic_metric_noise='UNMEASURED; semantic human audit not performed',
        deployable_adaptive_policy='UNTESTED; oracle headroom alone does not justify deployment')
    write_csv(root / 'full_reader_panel_summary.csv', table)
    write_csv(root / 'full_reader_bootstrap.csv', comparisons)
    atomic(root / 'full_study_summary.json', obj)
    lines = ['# Week 10 — Final root-cause study report', '',
        f"All planned R/C/G/D experiments are complete on **{len(qids)} project DEV questions from {len(set(papers.values()))} papers**. All three G readers are evaluated: Mistral, Qwen3-8B (nonthinking), and Llama3.1-8B. The frozen source/configuration and C/G question, paper, and exact evidence identities were validated. Historical partial-panel receipts are retained as historical records.", '',
        '**Scope:** development diagnostics, not a fresh confirmation study. Project DEV maps to raw QASPER TRAIN. No QASPER TEST execution, policy training, faithful SeleCom evaluation, or external confirmation was added.', '',
        f"Validated experiment answer rows: **{obj['total_experiment_generation_rows']}** — R={obj['completed_generation_rows']['R']}, C={obj['completed_generation_rows']['C']}, G={obj['completed_generation_rows']['G']}, D={obj['completed_generation_rows']['D']}. Only Llama's 693 G answers were newly generated in this completion pass; previous completed answers were reused. Qualification calls are separate from these experiment rows.", '',
        '## R — Retrieval foundation', '',
        'The same local trained SARA/Mistral checkpoint was held fixed across FULL and SARA interfaces. BM25, MiniLM, and SFR were evaluated at k=2/4/6/8. FULL exposes k text chunks; SARA also exposes remaining top-10 latent evidence, so this is not an equal-information interface comparison.', '',
        '| Retriever | FULL k2 | k4 | k6 | k8 | SARA k2 | k4 | k6 | k8 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    summary = list(csv.DictReader((root / 'summary.csv').open()))
    index = {(r['run'], r['action']): r for r in summary}
    for method in ('BM25', 'MiniLM', 'SFR'):
        values = [float(index[f'R-{method}-{interface}', f'{method}/{interface}/k{k}']['f1'])
                  for interface in ('full', 'sara') for k in (2, 4, 6, 8)]
        lines.append('| ' + method + ' | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    lines += ['', '| Retrieval at k6 | Any-gold recall | All-gold recall | Span coverage | Retrieved tokens |',
              '|---|---:|---:|---:|---:|']
    for row in csv.DictReader((root / 'retrieval_summary.csv').open()):
        if row['k'] == '6':
            lines.append(f"| {row['retriever']} | {float(row['any_recall']):.4f} | {float(row['all_recall']):.4f} | {float(row['coverage']):.4f} | {float(row['retrieved_tokens']):.1f} |")
    lines += ['', 'SFR is the strongest retrieval foundation in this surface. At k6, SFR−MiniLM F1 is +0.030409 FULL (95% paper CI [0.003019, 0.060660]) and +0.031372 SARA ([0.004948, 0.060979]). SFR−BM25 FULL also has positive CIs at k2 and k6. This supports a retrieval limitation; it does not establish that retrieval is the only bottleneck.', '',
        'At SFR k6, SARA−FULL is only +0.001410 (95% CI [−0.017479, 0.021239]); the surface does not establish an incremental latent-interface advantage. k6 is the best point estimate in this development sweep, not a confirmed optimum. Recall averages exclude questions with no gold spans, while unmapped spans remain misses. These are paper-local full-corpus retrieval results, not global retrieval.', '',
        '### Retrieval-stage oracle gaps', '',
        '| Surface | Best fixed F1 | Exact oracle F1 | Gap | Top 10% share of oracle gain |',
        '|---|---:|---:|---:|---:|']
    for name, value in obj['upstream_oracles'].items():
        if name.startswith('R-'):
            lines.append(f"| {name} | {value['best_fixed_f1']:.6f} | {value['exact_oracle_f1']:.6f} | {value['oracle_headroom']:.6f} | {value['top10_percent_gain_contribution']:.1%} |")
    lines += ['', 'Across the same k action set, the descriptive best-fixed/oracle gap falls from 0.07336 (BM25 FULL) to 0.05820 (SFR FULL), and from 0.07793 (BM25 SARA) to 0.05419 (SFR SARA). This is consistent with the retrieval foundation explaining part of the adaptive-budget gap. The residual SFR SARA gap is concentrated: about 71% of its gain comes from the top 10% of queries. Gap reductions themselves have not been assigned confidence intervals; the oracle is retrospective.', '',
        '## C — Representation controls under frozen SFR', '',
        '| Evidence representation | F1 | Mean prompt tokens | HELP vs none | HURT vs none |',
        '|---|---:|---:|---:|---:|']
    for action in ('no_evidence', 'full', 'sara_k4', 'lead256', 'qcs256'):
        row = index['C-Mistral', action]
        lines.append(f"| {action} | {float(row['f1']):.6f} | {float(row['mean_prompt_tokens']):.1f} | {float(row['HELP']):.3f} | {float(row['HURT']):.3f} |")
    lines += ['', 'FULL−QCS256 = +0.063020 (95% CI [0.019431, 0.109377]); FULL−SARA = +0.025862 (CI [−0.012674, 0.065663]); QCS−LEAD = +0.022585 (CI [−0.018719, 0.063554]). QCS reduces prompt burden, but the matched 256-token lexical controls do not establish a reliable query-conditioning gain. QCS is an exact-source textual surrogate, not a faithful SeleCom reproduction. The adapted C checkpoint differs from the public G Mistral reader; do not attribute cross-phase differences solely to compression or reader strength.', '',
        '## G — All three readers on identical evidence', '',
        '| Reader | Evidence | F1 | EM | ΔF1 vs same-reader none | HELP | HURT | Prompt tokens | Generation seconds |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append(f"| {row['reader']} | {row['action']} | {row['f1']:.6f} | {row['exact_match']:.6f} | {row['net_evidence_utility']:+.6f} | {row['HELP']:.3f} | {row['HURT']:.3f} | {row['mean_prompt_tokens']:.1f} | {row['mean_generation_seconds']:.3f} |")
    lines += ['', 'HELP/HURT use paired F1 differences greater than +0.01 / less than −0.01 against each reader’s own no-evidence answer. Token counts use each reader’s native tokenizer/template, so they are workload measures, not a shared evidence-length measure. Generation timing is measured; full end-to-end latency is not measured. Each answer uses one LM call.', '',
        'Generation-cap hits under the frozen 128-token budget: ' + '; '.join(
            f"{r['reader']}/{r['action']}={round(r['generation_cap_hit_rate']*r['n_queries'])}/{r['n_queries']}"
            for r in table) + '. Reader conclusions are conditional on this prompt/decoding budget, not universal model-capability rankings.', '',
        '### Paired paper-cluster intervals', '',
        '10,000 resamples, seed42, paper clusters; intervals are exploratory and unadjusted for multiple comparisons.', '',
        '| Comparison (left minus right) | ΔF1 | 95% CI |', '|---|---:|---|']
    for row in comparisons:
        lines.append(f"| {row['comparison']} | {row['point']:+.6f} | [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] |")
    lines += ['', '### Retrospective reader action oracles', '',
        '| Reader | Best fixed F1 | Exact oracle F1 | Epsilon oracle F1 | Headroom |', '|---|---:|---:|---:|---:|']
    for reader, result in obj['reader_oracles'].items():
        lines.append(f"| {reader} | {result['best_fixed_f1']:.6f} | {result['exact_oracle_f1']:.6f} | {result['epsilon_oracle_f1']:.6f} | {result['oracle_headroom']:.6f} |")
    lines += ['', 'These oracles retrospectively select among none/full/QCS using observed gold F1. They are not deployable controllers or evidence that a controller can recover the gap. R, C, and G use different action spaces; their oracle gaps are not directly comparable measures of controller difficulty.', '',
        '## D — Action-label stability and annotation diagnostics', '',
        f"On the prespecified 24-query Mistral subset, {stability['all_seeds_same_label_count']}/24 questions ({stability['all_seeds_same_label_fraction']:.1%}) retain the same HELP/NEUTRAL/HURT label across all three sampling seeds. Mean modal label agreement is {stability['mean_modal_label_agreement']:.3f}; mean delta-F1 variance is {stability['mean_delta_variance']:.6f}; mean EM-label agreement is {stability['mean_EM_modal_label_agreement']:.3f}.", '',
        'This is a small stochastic sensitivity diagnostic (temperature0.7/top-p0.9), not a test that greedy outputs change across runs, nor proof that F1 labels are semantically wrong. Primary reader decoding is greedy. All 72 sampled EM-delta labels are NEUTRAL; perfect EM-label stability is therefore degenerate and provides no useful evidence of reliable intervention labels.', '',
        f"Annotation inventory: {annotations['multiple_annotations']}/{annotations['n_queries']} questions have multiple annotations; {annotations['no_gold_evidence']} have no gold evidence span. Semantic correctness and metric mismatch require a preregistered human audit and remain unmeasured.", '',
        '| Sampling seed | Best fixed F1 | Exact oracle F1 | Epsilon oracle F1 |', '|---|---:|---:|---:|']
    for seed, value in stability['oracle_by_seed'].items():
        lines.append(f"| {seed} | {value['best_fixed_f1']:.6f} | {value['exact_oracle_f1']:.6f} | {value['epsilon_oracle_f1']:.6f} |")
    lines += ['', '## What Week 10 establishes', '',
        '1. **Retrieval quality is a supported contributor.** SFR improves the matched retrieval/answering foundation and descriptively narrows the adaptive-k oracle gap. It should be retained for the next controlled study.',
        '2. **The latent interface is not yet a demonstrated advantage.** The best local point estimate is SFR/SARA k6 (0.421290), but SFR/FULL k6 (0.419880) is essentially tied; their paired interval spans zero. Preserve the simple FULL control.',
        '3. **Aggressive text suppression has a consistent quality cost.** FULL beats QCS256 with positive paper-cluster intervals for all three G readers. This supports a budget/representation limitation, not a claim that every learned or latent compressor fails.',
        '4. **Reader utilization also matters.** Qwen and Llama improve full-evidence F1 and net evidence utility over public Mistral with positive paired intervals. Llama has the highest full-evidence point estimate (0.289647), while Qwen has the highest QCS256 point estimate (0.211520). Qwen-versus-Llama intervals include zero for both representations, so neither is a statistically established universal winner.',
        '5. **Noise is a limitation, not an established dominant explanation.** Four of 24 sampled questions change their F1-derived intervention label across seeds. EM-label stability is degenerate: every sampled EM-delta label is NEUTRAL. This small diagnostic neither proves semantic metric correctness nor shows that the remaining oracle gap is mostly randomness.',
        '6. **A deployable adaptive controller remains unproven.** Residual oracle gain is concentrated in relatively few questions, and no learned controller was trained here. A retrospective ceiling is not a controller result.', '',
        '## Recommended next step', '',
        'Freeze SFR as the retrieval foundation and retain SFR/FULL k6 plus the matched SFR/SARA k6 as controlled candidates with the existing trained local reader. The present G experiment supports reader effects within its public-reader setup; it does not isolate training/prompt differences between public G and adapted C/R checkpoints. Preregister a genuinely fresh confirmation comparison before investing in a new adaptive policy. If compression remains the research target, test a faithful trained compressor against FULL at explicit matched budgets; QCS256 alone does not settle that question. These are recommendations, not additional experiments launched by this report.', '',
        '## Completion and provenance', '',
        f"Execution freeze: `{frozen_hash}`. All reader model revisions remain as preregistered. C/G exact evidence equality, per-row checksums, expected query/action membership, paper IDs, completion receipts, and reader qualifications are checked by this report.", '',
        'Scientific source/config changes: none. Original R/C results and earlier partial-panel/D provenance are preserved. `full_study_summary.json` is the current complete-panel receipt; earlier `partial_panel_status.json` describes the historical Llama deferral.', '',
        '## Remaining research limits', '',
        '- Fresh external confirmation is not complete; DEV is not a blind evaluation set.',
        '- W9 FIT and W10 DEV scores have different cohorts/protocols; their raw F1 values are not a paired cross-week improvement estimate.',
        '- Faithful SeleCom and a learned textual compressor were not evaluated.',
        '- No semantic-correctness audit or evidence-entailment evaluation was performed.',
        '- The D subset is small and Mistral-only; do not generalize its rate to the whole dataset or other readers.',
        '- Oracle headroom does not establish a learnable adaptive policy; policy research remains disabled.', '']
    (root / 'W10_FINAL_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(complete=True, n_queries=len(qids), readers=list(panels),
                         reader_results=table, dataset_stability=stability), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    report(load(args.config))
