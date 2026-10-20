"""Real-output evaluation and paired paper-cluster bootstrap."""
from __future__ import annotations
import numpy as np
import pandas as pd


def answer_f1(prediction, evaluation):
    from src.week6.evaluation import compute_qa_answer_metrics
    return float(compute_qa_answer_metrics(prediction, evaluation['answers'],
        evaluation['answers_reformatted'])['token_f1'])


def annotate_outcome(row, inference, evaluation):
    base = answer_f1(inference['initial_draft'], evaluation)
    score = answer_f1(row['prediction'], evaluation)
    changed = row['prediction'].strip() != inference['initial_draft'].strip()
    evidence = row.get('evidence_text', '')
    answer_words = set(row['prediction'].lower().split()) - set(inference['initial_draft'].lower().split())
    from src.week6.evaluation import compute_qa_answer_metrics
    em=compute_qa_answer_metrics(row['prediction'],evaluation['answers'],evaluation['answers_reformatted'])['exact_match']
    return dict(row, exact_match=float(em), f1=score, initial_f1=base, delta_f1=score-base,
        answer_changed=changed, help=score-base > .01, hurt=score-base < -.01,
        unchanged_and_correct=not changed and score >= 1-1e-8,
        unchanged_but_incomplete=not changed and score < 1-1e-8,
        beneficial_revision=changed and score-base > .01, harmful_revision=changed and score-base < -.01,
        evidence_utilization_proxy=bool(answer_words & set(evidence.lower().split())),
        utilization_note='Lexical newly introduced evidence words; diagnostic, not causal attribution.')


def policy_summary(rows):
    if not rows:
        raise ValueError('Cannot summarize missing real outcomes')
    df = pd.DataFrame(rows)
    result = dict(n_queries=len(df), f1=float(df.f1.mean()))
    if 'delta_f1' in df:
        result['mean_selected_delta_f1'] = float(df.delta_f1.mean())
    selected = df[df.evidence_actual_count > 0] if 'evidence_actual_count' in df else df
    result['help_selected_rate'] = float(selected.help.mean()) if len(selected) and 'help' in selected else None
    result['hurt_selected_rate'] = float(selected.hurt.mean()) if len(selected) and 'hurt' in selected else None
    for name in ['exact_match','help', 'hurt', 'answer_changed', 'unchanged_and_correct', 'unchanged_but_incomplete',
        'beneficial_revision', 'harmful_revision', 'evidence_utilization_proxy',
        'final_prompt_tokens', 'total_query_prompt_tokens', 'total_query_generated_tokens', 'num_lm_calls',
        'retrieval_query_time', 'selection_latency', 'extraction_latency', 'revision_latency',
        'measured_end_to_end_latency', 'extraction_rate', 'promotion_rate', 'faithfulness']:
        result[name] = float(df[name].mean()) if name in df and df[name].notna().any() else None
    return result


def choose_global(records, tolerance=.003, primary='f1', cost='total_query_prompt_tokens'):
    if not records or not all(np.isfinite(r[primary]) for r in records):
        raise ValueError('Missing/nonfinite selection metrics')
    best = max(r[primary] for r in records)
    eligible = [r for r in records if r[primary] >= best-tolerance-1e-12]
    return min(eligible, key=lambda r: (r[cost], r.get('measured_latency') if r.get('measured_latency') is not None else float('inf'),
                                      r.get('complexity', 0), r['policy']))


def paired_paper_bootstrap(a, b, n=10000, seed=42):
    a, b = pd.DataFrame(a), pd.DataFrame(b)
    keys = ['question_id', 'paper_id']
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError('Bootstrap requires one outcome per query/policy')
    x = a.merge(b, on=keys, suffixes=('_a', '_b'), validate='one_to_one')
    if len(x) != len(a) or len(x) != len(b) or x.empty:
        raise ValueError('Unpaired/missing queries')
    x['difference'] = x.f1_a-x.f1_b
    groups = x.groupby('paper_id').difference.agg(['sum', 'count'])
    rng = np.random.default_rng(seed)
    samples = []
    # Bounded memory for 10,000 resamples on large paper counts.
    for start in range(0, n, 500):
        ix = rng.integers(0, len(groups), size=(min(500, n-start), len(groups)))
        samples.extend(groups['sum'].to_numpy()[ix].sum(1)/groups['count'].to_numpy()[ix].sum(1))
    return dict(point_estimate=float(x.difference.mean()), ci95=np.quantile(samples, [.025,.975]).tolist(),
                resamples=n, cluster='paper_id', seed=seed)


def selector_diagnostics(selected_rows, reference_rows):
    chosen, surface = pd.DataFrame(selected_rows), pd.DataFrame(reference_rows)
    best = surface.groupby('question_id').f1.max()
    base = chosen.set_index('question_id').initial_f1
    scores = chosen.set_index('question_id').f1
    if not set(scores.index) <= set(best.index):
        raise ValueError('Missing response surface identities')
    best = best.loc[scores.index]
    useful = (best-base) > .01
    regret = best-scores
    denominator = float((best-base).clip(lower=0).mean())
    return dict(mean_regret=float(regret.mean()), near_oracle_rate=float((regret<=.01).mean()),
        selector_headroom_recovery=float((scores-base).mean())/denominator if denominator else None,
        useful_queries=int(useful.sum()), conditional_help_rate=float(((scores-base)[useful]>.01).mean()) if useful.any() else None,
        conditional_hurt_rate=float(((scores-base)[useful]<-.01).mean()) if useful.any() else None,
        conditional_mean_delta=float((scores-base)[useful].mean()) if useful.any() else None,
        headroom_note='Oracle over measured selector/K actions only, not unmeasured combinations.')
