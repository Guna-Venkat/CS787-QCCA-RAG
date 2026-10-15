"""Stratified response analysis (4->6 gain groups) and paper-clustered bootstrap for W7-E.

Implements:
- 3-Way Stratified Gain Analysis:
    - Group A (Gain > +0.01): FULL ranks 5-6 help.
    - Group B (|Gain| <= 0.01): No meaningful change (~70% of queries).
    - Group C (Loss < -0.01): FULL ranks 5-6 hurt (distraction).
- Tests whether MEDIUM:
    - Recovers benefit in Group A (F_E1 > F_4)
    - Remains neutral in Group B (F_E1 approx F_4)
    - Reduces degradation in Group C (F_E1 > F_6)
- 2,000-sample Paper-Clustered Bootstrap testing statistical significance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_stratified_4to6_analysis(
    df_e1_eval: Optional[pd.DataFrame] = None,
    df_response_matrix: Optional[pd.DataFrame] = None,
    df_eval_winner: Optional[pd.DataFrame] = None,
    epsilon: float = 0.01,
) -> pd.DataFrame:
    """Compute 3-group stratified analysis of multi-fidelity performance against 4->6 marginal gain behavior.

    Groups defined on canonical response surface:
    - Group A: F6 - F4 > epsilon
    - Group B: |F6 - F4| <= epsilon
    - Group C: F6 - F4 < -epsilon
    """
    eval_df = df_e1_eval if df_e1_eval is not None else df_eval_winner
    if eval_df is None or df_response_matrix is None:
        raise ValueError("Both eval DataFrame (df_e1_eval or df_eval_winner) and df_response_matrix must be provided.")

    df_e1 = eval_df.copy()
    df_e1["question_id"] = df_e1["question_id"].astype(str).str.strip()

    df_resp = df_response_matrix.copy()
    df_resp["question_id"] = df_resp["question_id"].astype(str).str.strip()

    # Join E1 predictions with response surface
    merged = pd.merge(df_e1[["question_id", "token_f1", "context_tokens"]], df_resp, on="question_id", how="inner")
    total_q = len(merged)
    if total_q == 0:
        return pd.DataFrame()

    gain_4to6 = merged["f1_k6"].values.astype(float) - merged["f1_k4"].values.astype(float)
    f1_k4 = merged["f1_k4"].values.astype(float)
    f1_k6 = merged["f1_k6"].values.astype(float)
    f1_e1 = merged["token_f1"].values.astype(float)

    group_a_mask = gain_4to6 > epsilon
    group_c_mask = gain_4to6 < -epsilon
    group_b_mask = ~group_a_mask & ~group_c_mask

    group_definitions = [
        ("Group A: Ranks 5-6 Help (Gain > +0.01)", group_a_mask),
        ("Group B: Neutral (|Gain| <= 0.01)", group_b_mask),
        ("Group C: Ranks 5-6 Hurt (Loss < -0.01)", group_c_mask),
    ]

    summary_rows: List[Dict[str, Any]] = []

    for grp_name, mask in group_definitions:
        n_grp = int(np.sum(mask))
        if n_grp == 0:
            summary_rows.append({
                "group": grp_name,
                "Group": grp_name,
                "n_queries": 0,
                "pct_queries": 0.0,
                "mean_f1_k4": 0.0,
                "mean_f1_k6": 0.0,
                "mean_f1_e1": 0.0,
                "delta_f1_vs_k4": 0.0,
                "delta_e1_vs_k4": 0.0,
                "delta_f1_vs_k6": 0.0,
                "delta_e1_vs_k6": 0.0,
                "benefit_recovery_pct": 0.0,
            })
            continue

        pct = float(n_grp / total_q) * 100.0
        m_k4 = float(np.mean(f1_k4[mask]))
        m_k6 = float(np.mean(f1_k6[mask]))
        m_e1 = float(np.mean(f1_e1[mask]))

        d_vs_k4 = m_e1 - m_k4
        d_vs_k6 = m_e1 - m_k6
        denom = m_k6 - m_k4
        recovery_pct = float((d_vs_k4 / denom) * 100.0) if abs(denom) > 1e-5 else 0.0

        summary_rows.append({
            "group": grp_name,
            "Group": grp_name,
            "n_queries": n_grp,
            "pct_queries": pct,
            "mean_f1_k4": m_k4,
            "mean_f1_k6": m_k6,
            "mean_f1_e1": m_e1,
            "delta_f1_vs_k4": d_vs_k4,
            "delta_e1_vs_k4": d_vs_k4,
            "delta_f1_vs_k6": d_vs_k6,
            "delta_e1_vs_k6": d_vs_k6,
            "benefit_recovery_pct": recovery_pct,
        })

    return pd.DataFrame(summary_rows)


def _pairwise_paper_clustered_bootstrap(
    df_eval: pd.DataFrame,
    f1_col_a: str,
    f1_col_b: str,
    tok_col_a: str,
    tok_col_b: str,
    paper_col: str = "paper_id",
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Internal helper computing pairwise bootstrap metrics on one dataset."""
    rng = np.random.RandomState(random_seed)
    unique_papers = np.array(df_eval[paper_col].unique())
    n_papers = len(unique_papers)

    paper_to_indices = {p: df_eval.index[df_eval[paper_col] == p].to_numpy() for p in unique_papers}

    delta_f1s = np.zeros(n_resamples, dtype=float)
    delta_toks = np.zeros(n_resamples, dtype=float)

    for b in range(n_resamples):
        sampled_papers = rng.choice(unique_papers, size=n_papers, replace=True)
        sampled_indices = np.concatenate([paper_to_indices[p] for p in sampled_papers])
        sub = df_eval.iloc[sampled_indices]

        delta_f1s[b] = np.mean(sub[f1_col_a].values) - np.mean(sub[f1_col_b].values)
        delta_toks[b] = np.mean(sub[tok_col_a].values) - np.mean(sub[tok_col_b].values)

    alpha = 1.0 - confidence_level
    f1_ci_lower = float(np.percentile(delta_f1s, 100.0 * (alpha / 2.0)))
    f1_ci_upper = float(np.percentile(delta_f1s, 100.0 * (1.0 - alpha / 2.0)))

    tok_ci_lower = float(np.percentile(delta_toks, 100.0 * (alpha / 2.0)))
    tok_ci_upper = float(np.percentile(delta_toks, 100.0 * (1.0 - alpha / 2.0)))

    mean_delta_f1 = float(np.mean(delta_f1s))
    if mean_delta_f1 >= 0:
        p_val = float(2.0 * np.mean(delta_f1s <= 0))
    else:
        p_val = float(2.0 * np.mean(delta_f1s >= 0))
    p_val = min(1.0, max(0.0, p_val))

    sig = bool(f1_ci_lower > 0 or f1_ci_upper < 0)

    return {
        "mean_delta_f1": mean_delta_f1,
        "f1_ci_lower": f1_ci_lower,
        "f1_ci_upper": f1_ci_upper,
        "p_value_f1": p_val,
        "mean_delta_tokens": float(np.mean(delta_toks)),
        "token_ci_lower": tok_ci_lower,
        "token_ci_upper": tok_ci_upper,
        "significant": sig,
    }


def run_multifidelity_paper_clustered_bootstrap(
    df_eval: Optional[pd.DataFrame] = None,
    f1_col_a: Optional[str] = None,
    f1_col_b: Optional[str] = None,
    tok_col_a: Optional[str] = None,
    tok_col_b: Optional[str] = None,
    paper_col: str = "paper_id",
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    # Multi-comparison interface for Section 23:
    df_eval_e1: Optional[pd.DataFrame] = None,
    df_eval_e1x: Optional[pd.DataFrame] = None,
    df_response_matrix: Optional[pd.DataFrame] = None,
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Run paper-clustered bootstrap resamples comparing policies.

    Supports two interfaces:
    1. Multi-comparison (Section 23): pass df_eval_e1, df_response_matrix, and optionally df_eval_e1x.
       Returns pd.DataFrame with all Section 35 required comparisons.
    2. Pairwise: pass df_eval, f1_col_a, f1_col_b, tok_col_a, tok_col_b.
       Returns Dict[str, Any].
    """
    if df_eval is not None and f1_col_a is not None and f1_col_b is not None:
        return _pairwise_paper_clustered_bootstrap(
            df_eval=df_eval,
            f1_col_a=f1_col_a,
            f1_col_b=f1_col_b,
            tok_col_a=tok_col_a or "context_tokens",
            tok_col_b=tok_col_b or "context_tokens",
            paper_col=paper_col,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )

    if df_eval_e1 is None or df_response_matrix is None:
        raise ValueError("Must provide either (df_eval, f1_col_a, f1_col_b) or (df_eval_e1, df_response_matrix).")

    # Build unified evaluation dataframe
    e1_sub = df_eval_e1[["question_id", "paper_id", "token_f1", "context_tokens"]].copy()
    e1_sub.rename(columns={"token_f1": "f1_e1", "context_tokens": "tokens_e1"}, inplace=True)
    e1_sub["question_id"] = e1_sub["question_id"].astype(str).str.strip()

    resp_sub = df_response_matrix.copy()
    resp_sub["question_id"] = resp_sub["question_id"].astype(str).str.strip()

    merged = pd.merge(e1_sub, resp_sub, on="question_id", how="inner")
    if "paper_id_x" in merged.columns:
        merged["paper_id"] = merged["paper_id_x"]

    if df_eval_e1x is not None:
        e1x_sub = df_eval_e1x[["question_id", "token_f1", "context_tokens"]].copy()
        e1x_sub.rename(columns={"token_f1": "f1_e1x", "context_tokens": "tokens_e1x"}, inplace=True)
        e1x_sub["question_id"] = e1x_sub["question_id"].astype(str).str.strip()
        merged = pd.merge(merged, e1x_sub, on="question_id", how="inner")

    # Comparisons from Section 35:
    comparisons = [
        ("E1 (4F+2M+4H) vs Semantic k=4", "f1_e1", "f1_k4", "tokens_e1", "tokens_k4"),
        ("E1 (4F+2M+4H) vs Semantic k=6", "f1_e1", "f1_k6", "tokens_e1", "tokens_k6"),
    ]

    if df_eval_e1x is not None and "f1_e1x" in merged.columns:
        comparisons.extend([
            ("E1-X (CE-Informed) vs E1 Primary", "f1_e1x", "f1_e1", "tokens_e1x", "tokens_e1"),
            ("E1-X (CE-Informed) vs Semantic k=4", "f1_e1x", "f1_k4", "tokens_e1x", "tokens_k4"),
            ("E1-X (CE-Informed) vs Semantic k=6", "f1_e1x", "f1_k6", "tokens_e1x", "tokens_k6"),
        ])

    comp_rows: List[Dict[str, Any]] = []
    for comp_name, col_fa, col_fb, col_ta, col_tb in comparisons:
        res = _pairwise_paper_clustered_bootstrap(
            df_eval=merged,
            f1_col_a=col_fa,
            f1_col_b=col_fb,
            tok_col_a=col_ta,
            tok_col_b=col_tb,
            paper_col=paper_col,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )
        comp_rows.append({
            "Comparison": comp_name,
            "mean_delta_f1": res["mean_delta_f1"],
            "f1_ci_lower": res["f1_ci_lower"],
            "f1_ci_upper": res["f1_ci_upper"],
            "p_value_f1": res["p_value_f1"],
            "mean_delta_tokens": res["mean_delta_tokens"],
            "token_ci_lower": res["token_ci_lower"],
            "token_ci_upper": res["token_ci_upper"],
            "significant": res["significant"],
        })

    return pd.DataFrame(comp_rows)
