"""Paper-clustered bootstrap hypothesis testing for Week 7-B.

Implements:
- 2,000 paper-clustered resamples ensuring questions from the same paper are sampled together.
- Paired statistical comparisons:
  1. NEW semantic-aware allocator vs old W4 + semantic (W7-A)
  2. NEW semantic-aware allocator vs static semantic k=4
  3. NEW semantic-aware allocator vs static semantic k=6
  4. NEW semantic-aware allocator vs static semantic k=8
  5. Semantic epsilon-oracle vs static semantic k=8 (headroom diagnostic)
- Reports Delta F1, 95% bootstrap confidence intervals, empirical p-values, Delta context tokens, and token CIs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_paper_clustered_bootstrap(
    paper_ids: Sequence[str],
    vec_treatment_f1: Sequence[float],
    vec_control_f1: Sequence[float],
    vec_treatment_tokens: Optional[Sequence[int]] = None,
    vec_control_tokens: Optional[Sequence[int]] = None,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute paper-clustered bootstrap comparison between treatment and control policies.

    Args:
        paper_ids: Length N list of string paper IDs.
        vec_treatment_f1: Length N float list of per-query F1 for treatment policy.
        vec_control_f1: Length N float list of per-query F1 for control policy.
        vec_treatment_tokens: Length N int list of context tokens for treatment policy.
        vec_control_tokens: Length N int list of context tokens for control policy.
        n_resamples: Number of cluster bootstrap iterations (default 2000).
        confidence_level: Desired coverage (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        Dict with mean_delta_f1, ci_f1_lower, ci_f1_upper, p_value, mean_delta_tokens, ci_tok_lower, ci_tok_upper.
    """
    n = len(paper_ids)
    assert len(vec_treatment_f1) == n and len(vec_control_f1) == n

    df_sample = pd.DataFrame({
        "paper_id": list(paper_ids),
        "f1_treat": list(vec_treatment_f1),
        "f1_ctrl": list(vec_control_f1),
    })

    has_tokens = vec_treatment_tokens is not None and vec_control_tokens is not None
    if has_tokens:
        df_sample["tok_treat"] = list(vec_treatment_tokens)
        df_sample["tok_ctrl"] = list(vec_control_tokens)

    unique_papers = np.array(sorted(df_sample["paper_id"].unique()))
    n_papers = len(unique_papers)

    # Pre-aggregate indices by paper for fast cluster sampling
    paper_to_indices = {p: df_sample.index[df_sample["paper_id"] == p].values for p in unique_papers}

    obs_delta_f1 = float(np.mean(vec_treatment_f1) - np.mean(vec_control_f1))
    obs_delta_tok = float(np.mean(vec_treatment_tokens) - np.mean(vec_control_tokens)) if has_tokens else 0.0

    rng = np.random.default_rng(seed)
    boot_deltas_f1 = np.empty(n_resamples, dtype=np.float64)
    boot_deltas_tok = np.empty(n_resamples, dtype=np.float64) if has_tokens else None

    treat_f1_arr = df_sample["f1_treat"].values
    ctrl_f1_arr = df_sample["f1_ctrl"].values
    if has_tokens:
        treat_tok_arr = df_sample["tok_treat"].values
        ctrl_tok_arr = df_sample["tok_ctrl"].values

    for b in range(n_resamples):
        sampled_papers = rng.choice(unique_papers, size=n_papers, replace=True)
        sampled_idx_list = [paper_to_indices[p] for p in sampled_papers]
        idx_concat = np.concatenate(sampled_idx_list)

        d_f1 = np.mean(treat_f1_arr[idx_concat]) - np.mean(ctrl_f1_arr[idx_concat])
        boot_deltas_f1[b] = d_f1

        if has_tokens:
            d_tok = np.mean(treat_tok_arr[idx_concat]) - np.mean(ctrl_tok_arr[idx_concat])
            boot_deltas_tok[b] = d_tok

    alpha = 1.0 - confidence_level
    ci_f1_low = float(np.percentile(boot_deltas_f1, 100 * (alpha / 2.0)))
    ci_f1_high = float(np.percentile(boot_deltas_f1, 100 * (1.0 - alpha / 2.0)))

    # Empirical two-sided p-value against null of delta <= 0
    if obs_delta_f1 > 0:
        p_val = float(2.0 * np.mean(boot_deltas_f1 <= 0.0))
    else:
        p_val = float(2.0 * np.mean(boot_deltas_f1 >= 0.0))
    p_val = min(1.0, max(1.0 / n_resamples, p_val))

    res: Dict[str, Any] = {
        "obs_delta_f1": obs_delta_f1,
        "ci_f1_lower": ci_f1_low,
        "ci_f1_upper": ci_f1_high,
        "p_value": p_val,
        "significant": bool(ci_f1_low > 0 or ci_f1_high < 0),
    }

    if has_tokens:
        ci_tok_low = float(np.percentile(boot_deltas_tok, 100 * (alpha / 2.0)))
        ci_tok_high = float(np.percentile(boot_deltas_tok, 100 * (1.0 - alpha / 2.0)))
        res.update({
            "obs_delta_tokens": obs_delta_tok,
            "ci_tokens_lower": ci_tok_low,
            "ci_tokens_upper": ci_tok_high,
        })

    return res


def run_all_w7b_bootstrap_comparisons(
    paper_ids: Sequence[str],
    f1_w7b_adaptive: Sequence[float],
    tokens_w7b_adaptive: Sequence[int],
    f1_w7a_composite: Sequence[float],
    tokens_w7a_composite: Sequence[int],
    df_dev_matrix: pd.DataFrame,
    df_sem_oracle: Optional[pd.DataFrame] = None,
    output_dir: Optional[Union[str, Path]] = None,
    n_resamples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all primary Week 7-B bootstrap comparisons and compile results table."""
    comparisons = [
        ("W7B_Adaptive_vs_W7A_Composite", f1_w7b_adaptive, tokens_w7b_adaptive, f1_w7a_composite, tokens_w7a_composite),
        ("W7B_Adaptive_vs_Static_Semantic_k4", f1_w7b_adaptive, tokens_w7b_adaptive, df_dev_matrix["f1_k4"].values, df_dev_matrix["tokens_k4"].values),
        ("W7B_Adaptive_vs_Static_Semantic_k6", f1_w7b_adaptive, tokens_w7b_adaptive, df_dev_matrix["f1_k6"].values, df_dev_matrix["tokens_k6"].values),
        ("W7B_Adaptive_vs_Static_Semantic_k8", f1_w7b_adaptive, tokens_w7b_adaptive, df_dev_matrix["f1_k8"].values, df_dev_matrix["tokens_k8"].values),
    ]

    if df_sem_oracle is not None:
        comparisons.append((
            "Semantic_Oracle_vs_Static_Semantic_k8",
            df_sem_oracle["oracle_f1"].values,
            df_sem_oracle["oracle_tokens"].values,
            df_dev_matrix["f1_k8"].values,
            df_dev_matrix["tokens_k8"].values,
        ))

    rows = []
    for comp_name, t_f1, t_tok, c_f1, c_tok in comparisons:
        res = run_paper_clustered_bootstrap(
            paper_ids=paper_ids,
            vec_treatment_f1=t_f1,
            vec_control_f1=c_f1,
            vec_treatment_tokens=t_tok,
            vec_control_tokens=c_tok,
            n_resamples=n_resamples,
            seed=seed,
        )
        rows.append({
            "comparison": comp_name,
            "delta_f1": round(res["obs_delta_f1"], 4),
            "ci_95_f1": f"[{res['ci_f1_lower']:+.4f}, {res['ci_f1_upper']:+.4f}]",
            "p_value": round(res["p_value"], 4),
            "significant_at_05": res["significant"],
            "delta_tokens": round(res.get("obs_delta_tokens", 0.0), 1),
            "ci_95_tokens": f"[{res.get('ci_tokens_lower', 0.0):+.1f}, {res.get('ci_tokens_upper', 0.0):+.1f}]",
        })

    df_boot = pd.DataFrame(rows)

    if output_dir:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        df_boot.to_csv(p / "w7b_bootstrap_comparisons.csv", index=False)
        logger.info("Saved bootstrap comparisons table to %s", p / "w7b_bootstrap_comparisons.csv")

    return df_boot
