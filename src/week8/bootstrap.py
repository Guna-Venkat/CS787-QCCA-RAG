"""Paper-Clustered Bootstrap Significance Engine for Week 8.

Executes 2,000 cluster-resamples (clustered by paper_id to account for inter-document dependence)
to compute:
- Delta F1
- 95% Confidence Intervals [2.5%, 97.5%]
- Two-sided empirical p-value
- Delta final context tokens
- Delta cumulative prompt tokens
- Latency differences
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_paper_clustered_bootstrap(
    df_treatment: pd.DataFrame,
    df_control: pd.DataFrame,
    treatment_name: str = "treatment",
    control_name: str = "control",
    n_resamples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute paper-clustered paired bootstrap test between treatment and control."""
    rng = np.random.RandomState(seed)

    # Ensure alignment on question_id
    t_df = df_treatment.set_index("question_id").sort_index()
    c_df = df_control.set_index("question_id").sort_index()

    common_qids = t_df.index.intersection(c_df.index)
    if len(common_qids) == 0:
        raise ValueError("No common question_ids found between treatment and control.")

    t_sub = t_df.loc[common_qids]
    c_sub = c_df.loc[common_qids]

    # Cluster by paper_id
    paper_ids = t_sub["paper_id"].values
    unique_papers = np.unique(paper_ids)
    num_papers = len(unique_papers)

    # Group question indices by paper
    paper_to_indices = {}
    for idx, pid in enumerate(paper_ids):
        paper_to_indices.setdefault(pid, []).append(idx)

    # Pre-extract arrays
    t_f1 = t_sub["token_f1"].values
    c_f1 = c_sub["token_f1"].values

    if "final_context_tokens" in t_sub.columns:
        t_final_tokens = t_sub["final_context_tokens"].values
    elif "context_tokens" in t_sub.columns:
        t_final_tokens = t_sub["context_tokens"].values
    else:
        t_final_tokens = np.zeros(len(common_qids))

    if "final_context_tokens" in c_sub.columns:
        c_final_tokens = c_sub["final_context_tokens"].values
    elif "context_tokens" in c_sub.columns:
        c_final_tokens = c_sub["context_tokens"].values
    else:
        c_final_tokens = np.zeros(len(common_qids))

    t_cum_tokens = t_sub["cumulative_prompt_tokens"].values if "cumulative_prompt_tokens" in t_sub.columns else t_final_tokens
    c_cum_tokens = c_sub["cumulative_prompt_tokens"].values if "cumulative_prompt_tokens" in c_sub.columns else c_final_tokens

    t_latency = t_sub["e2e_latency_sec"].values if "e2e_latency_sec" in t_sub.columns else np.zeros(len(common_qids))
    c_latency = c_sub["e2e_latency_sec"].values if "e2e_latency_sec" in c_sub.columns else np.zeros(len(common_qids))

    delta_f1_observed = float(np.mean(t_f1 - c_f1))
    delta_final_tokens_observed = float(np.mean(t_final_tokens - c_final_tokens))
    delta_cum_tokens_observed = float(np.mean(t_cum_tokens - c_cum_tokens))
    delta_latency_observed = float(np.mean(t_latency - c_latency))

    boot_deltas_f1 = np.empty(n_resamples, dtype=np.float64)
    boot_deltas_final_tokens = np.empty(n_resamples, dtype=np.float64)
    boot_deltas_cum_tokens = np.empty(n_resamples, dtype=np.float64)
    boot_deltas_latency = np.empty(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        sampled_papers = rng.choice(unique_papers, size=num_papers, replace=True)
        sampled_indices = []
        for p in sampled_papers:
            sampled_indices.extend(paper_to_indices[p])

        s_idx = np.array(sampled_indices, dtype=np.int64)
        boot_deltas_f1[b] = np.mean(t_f1[s_idx] - c_f1[s_idx])
        boot_deltas_final_tokens[b] = np.mean(t_final_tokens[s_idx] - c_final_tokens[s_idx])
        boot_deltas_cum_tokens[b] = np.mean(t_cum_tokens[s_idx] - c_cum_tokens[s_idx])
        boot_deltas_latency[b] = np.mean(t_latency[s_idx] - c_latency[s_idx])

    ci_f1_lower = float(np.percentile(boot_deltas_f1, 2.5))
    ci_f1_upper = float(np.percentile(boot_deltas_f1, 97.5))

    ci_tokens_lower = float(np.percentile(boot_deltas_final_tokens, 2.5))
    ci_tokens_upper = float(np.percentile(boot_deltas_final_tokens, 97.5))

    # Empirical two-sided p-value
    # H0: delta_f1 == 0
    p_val_f1 = float(2 * min(np.mean(boot_deltas_f1 <= 0), np.mean(boot_deltas_f1 >= 0)))
    p_val_f1 = min(p_val_f1, 1.0)

    return {
        "comparison": f"{treatment_name}_vs_{control_name}",
        "n_resamples": n_resamples,
        "n_queries": len(common_qids),
        "n_papers": num_papers,
        "mean_treatment_f1": float(np.mean(t_f1)),
        "mean_control_f1": float(np.mean(c_f1)),
        "delta_f1": delta_f1_observed,
        "ci_f1_95": [ci_f1_lower, ci_f1_upper],
        "p_value_f1": p_val_f1,
        "delta_final_tokens": delta_final_tokens_observed,
        "ci_final_tokens_95": [ci_tokens_lower, ci_tokens_upper],
        "delta_cum_tokens": delta_cum_tokens_observed,
        "delta_latency_sec": delta_latency_observed,
    }


def run_all_week8_bootstrap_comparisons(
    results_by_system: Dict[str, pd.DataFrame],
    n_resamples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all required bootstrap comparisons specified in Section 50."""
    comparison_pairs = [
        ("DAFiR_Full", "semantic_k4"),
        ("DAFiR_Full", "semantic_k6"),
        ("DAFiR_Full", "semantic_k8"),
        ("DAFiR_1", "semantic_k4"),
        ("DAFiR_2", "semantic_k6"),
        ("U2_model", "U1_model"),
        ("U4_model", "U2_model"),
        ("QCLA_system", "SFR_system"),
        ("HC_LoRA", "base_generator"),
        ("OW_LoRA", "HC_LoRA"),
    ]

    records = []
    for treat, ctrl in comparison_pairs:
        if treat in results_by_system and ctrl in results_by_system:
            res = run_paper_clustered_bootstrap(
                df_treatment=results_by_system[treat],
                df_control=results_by_system[ctrl],
                treatment_name=treat,
                control_name=ctrl,
                n_resamples=n_resamples,
                seed=seed,
            )
            records.append({
                "comparison": res["comparison"],
                "delta_f1": res["delta_f1"],
                "ci_95_f1": f"[{res['ci_f1_95'][0]:.4f}, {res['ci_f1_95'][1]:.4f}]",
                "p_value": res["p_value_f1"],
                "delta_final_tokens": res["delta_final_tokens"],
                "delta_cum_tokens": res["delta_cum_tokens"],
                "delta_latency_sec": res["delta_latency_sec"],
            })

    return pd.DataFrame(records)
