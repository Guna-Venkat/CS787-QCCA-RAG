"""Evaluation routines and latency analysis for W7-E multi-fidelity representations.

Computes:
- Downstream QA answer quality metrics (token F1, Exact Match, ROUGE-L).
- End-to-end latency breakdowns (preprocessing, generation, total).
- Summary comparison tables across policies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_policy_metrics_summary(
    df_eval: pd.DataFrame,
    policy_name: str,
) -> Dict[str, Any]:
    """Compute aggregate evaluation metrics for a single policy."""
    if df_eval.empty:
        return {"policy_name": policy_name, "n_queries": 0}

    f1_vals = df_eval["token_f1"].values.astype(float)
    em_vals = df_eval["exact_match"].values.astype(float)
    rouge_vals = df_eval["rouge_l"].values.astype(float)
    tok_vals = df_eval["context_tokens"].values.astype(float)

    pre_lat = df_eval["preprocessing_latency"].values.astype(float) if "preprocessing_latency" in df_eval else np.zeros_like(f1_vals)
    gen_lat = df_eval["generation_latency"].values.astype(float) if "generation_latency" in df_eval else np.zeros_like(f1_vals)
    tot_lat = df_eval["total_latency"].values.astype(float) if "total_latency" in df_eval else (pre_lat + gen_lat)

    return {
        "policy_name": policy_name,
        "n_queries": len(df_eval),
        "mean_f1": float(np.mean(f1_vals)),
        "median_f1": float(np.median(f1_vals)),
        "std_f1": float(np.std(f1_vals)),
        "mean_em": float(np.mean(em_vals)),
        "mean_rouge_l": float(np.mean(rouge_vals)),
        "mean_tokens": float(np.mean(tok_vals)),
        "median_tokens": float(np.median(tok_vals)),
        "std_tokens": float(np.std(tok_vals)),
        "preprocessing_latency_s": float(np.mean(pre_lat)),
        "generation_latency_s": float(np.mean(gen_lat)),
        "total_latency_s": float(np.mean(tot_lat)),
    }


def compute_latency_comparison_table(
    policy_dataframes: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build comparative latency table across policies."""
    records: List[Dict[str, Any]] = []

    for pol_name, df_pol in policy_dataframes.items():
        summary = compute_policy_metrics_summary(df_pol, policy_name=pol_name)
        records.append({
            "Policy": pol_name,
            "Mean F1": summary["mean_f1"],
            "Context Tokens": summary["mean_tokens"],
            "Preprocessing (s)": summary["preprocessing_latency_s"],
            "Generation (s)": summary["generation_latency_s"],
            "Total Latency (s)": summary["total_latency_s"],
        })

    return pd.DataFrame(records)
