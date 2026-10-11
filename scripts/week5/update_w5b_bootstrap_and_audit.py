"""Script to produce the complete, transparent multi-seed and across-seed bootstrap audit table.

Addresses Audit 1:
- Computes bootstrap CIs for Seed 42, Seed 123, Seed 2026 independently (clearly labeled)
- Computes bootstrap CIs for the Across-Seed Mean Policy (matching Table 1 3-seed mean)
- Overwrites results/week5/content_allocator/w5b/bootstrap/w5b_bootstrap_comparisons.csv
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from src.week5.evaluation import compute_paper_clustered_bootstrap, evaluate_allocation_policy

# Load predictions and supervision
dev_sup = pd.read_csv("results/week5/supervision/dev_allocation_supervision.csv")
dev_preds = pd.read_csv("results/week5/content_allocator/w5b/predictions/dev_predictions_t3.csv")
paper_ids = dev_preds["paper_id"].tolist()

# Baselines
k8_f1s = dev_sup["f1_k8"].values
k8_toks = np.array([1750.5] * len(dev_sup))

w4_oof = pd.read_csv("results/week4/model_benchmark/oof_predictions.csv")
w4_oof["question_id"] = w4_oof["question_id"].astype(str)
w4_col = "pred_t05_Balanced Logistic Regression (5 feats, Baseline)"
dev_qids_df = pd.DataFrame({"question_id": dev_sup["question_id"].astype(str)})
w4_merged = pd.merge(dev_qids_df, w4_oof[["question_id", w4_col]], on="question_id", how="left")
w4_preds_k = np.where(w4_merged[w4_col].values == 1, 6, 4)
w4_eval = evaluate_allocation_policy(dev_sup, w4_preds_k)
w4_f1s = w4_eval["per_query_f1s"]
w4_toks = w4_eval["per_query_tokens"]

rows = []

# 1. Per-seed evaluations for I3
evals_i3 = {}
for seed in [42, 123, 2026]:
    e = evaluate_allocation_policy(dev_sup, dev_preds[f"pred_k_I3_seed{seed}"].values)
    evals_i3[seed] = e

# 2. Per-seed evaluations for I4
evals_i4 = {}
for seed in [42, 123, 2026]:
    e = evaluate_allocation_policy(dev_sup, dev_preds[f"pred_k_I4_seed{seed}"].values)
    evals_i4[seed] = e

# Per-seed bootstrap comparisons
for seed in [42, 123, 2026]:
    e3 = evals_i3[seed]
    e4 = evals_i4[seed]

    # I3 vs W4
    b_i3_w4 = compute_paper_clustered_bootstrap(paper_ids, e3["per_query_f1s"], w4_f1s, e3["per_query_tokens"], w4_toks, seed=42)
    rows.append({
        "aggregation_level": f"Single Seed (Seed {seed})",
        "comparison": f"I3-T3 (Seed {seed}) vs W4",
        "observed_delta_f1": b_i3_w4["observed_delta_f1"],
        "ci_lower_95_f1": b_i3_w4["ci_lower_95"],
        "ci_upper_95_f1": b_i3_w4["ci_upper_95"],
        "p_value_f1": b_i3_w4["p_value"],
        "observed_delta_tokens": b_i3_w4["observed_delta_tokens"],
        "ci_lower_95_tokens": b_i3_w4["tokens_ci_lower_95"],
        "ci_upper_95_tokens": b_i3_w4["tokens_ci_upper_95"],
        "significant_f1": b_i3_w4["ci_lower_95"] > 0 or b_i3_w4["ci_upper_95"] < 0,
    })

    # I3 vs Static k=8
    b_i3_k8 = compute_paper_clustered_bootstrap(paper_ids, e3["per_query_f1s"], k8_f1s, e3["per_query_tokens"], k8_toks, seed=42)
    rows.append({
        "aggregation_level": f"Single Seed (Seed {seed})",
        "comparison": f"I3-T3 (Seed {seed}) vs Static k=8",
        "observed_delta_f1": b_i3_k8["observed_delta_f1"],
        "ci_lower_95_f1": b_i3_k8["ci_lower_95"],
        "ci_upper_95_f1": b_i3_k8["ci_upper_95"],
        "p_value_f1": b_i3_k8["p_value"],
        "observed_delta_tokens": b_i3_k8["observed_delta_tokens"],
        "ci_lower_95_tokens": b_i3_k8["tokens_ci_lower_95"],
        "ci_upper_95_tokens": b_i3_k8["tokens_ci_upper_95"],
        "significant_f1": b_i3_k8["ci_lower_95"] > 0 or b_i3_k8["ci_upper_95"] < 0,
    })

    # I4 vs I3
    b_i4_i3 = compute_paper_clustered_bootstrap(paper_ids, e4["per_query_f1s"], e3["per_query_f1s"], e4["per_query_tokens"], e3["per_query_tokens"], seed=42)
    rows.append({
        "aggregation_level": f"Single Seed (Seed {seed})",
        "comparison": f"I4-T3 (Seed {seed}) vs I3-T3 (Seed {seed})",
        "observed_delta_f1": b_i4_i3["observed_delta_f1"],
        "ci_lower_95_f1": b_i4_i3["ci_lower_95"],
        "ci_upper_95_f1": b_i4_i3["ci_upper_95"],
        "p_value_f1": b_i4_i3["p_value"],
        "observed_delta_tokens": b_i4_i3["observed_delta_tokens"],
        "ci_lower_95_tokens": b_i4_i3["tokens_ci_lower_95"],
        "ci_upper_95_tokens": b_i4_i3["tokens_ci_upper_95"],
        "significant_f1": b_i4_i3["ci_lower_95"] > 0 or b_i4_i3["ci_upper_95"] < 0,
    })

# 3. Across-seed mean policy bootstrap
i3_mean_f1 = (evals_i3[42]["per_query_f1s"] + evals_i3[123]["per_query_f1s"] + evals_i3[2026]["per_query_f1s"]) / 3.0
i3_mean_tok = (evals_i3[42]["per_query_tokens"] + evals_i3[123]["per_query_tokens"] + evals_i3[2026]["per_query_tokens"]) / 3.0

i4_mean_f1 = (evals_i4[42]["per_query_f1s"] + evals_i4[123]["per_query_f1s"] + evals_i4[2026]["per_query_f1s"]) / 3.0
i4_mean_tok = (evals_i4[42]["per_query_tokens"] + evals_i4[123]["per_query_tokens"] + evals_i4[2026]["per_query_tokens"]) / 3.0

b_mean_i3_w4 = compute_paper_clustered_bootstrap(paper_ids, i3_mean_f1, w4_f1s, i3_mean_tok, w4_toks, seed=42)
rows.append({
    "aggregation_level": "Across-Seed Mean Policy (3 Seeds)",
    "comparison": "I3-T3 (Across-Seed Mean) vs W4",
    "observed_delta_f1": b_mean_i3_w4["observed_delta_f1"],
    "ci_lower_95_f1": b_mean_i3_w4["ci_lower_95"],
    "ci_upper_95_f1": b_mean_i3_w4["ci_upper_95"],
    "p_value_f1": b_mean_i3_w4["p_value"],
    "observed_delta_tokens": b_mean_i3_w4["observed_delta_tokens"],
    "ci_lower_95_tokens": b_mean_i3_w4["tokens_ci_lower_95"],
    "ci_upper_95_tokens": b_mean_i3_w4["tokens_ci_upper_95"],
    "significant_f1": b_mean_i3_w4["ci_lower_95"] > 0 or b_mean_i3_w4["ci_upper_95"] < 0,
})

b_mean_i3_k8 = compute_paper_clustered_bootstrap(paper_ids, i3_mean_f1, k8_f1s, i3_mean_tok, k8_toks, seed=42)
rows.append({
    "aggregation_level": "Across-Seed Mean Policy (3 Seeds)",
    "comparison": "I3-T3 (Across-Seed Mean) vs Static k=8",
    "observed_delta_f1": b_mean_i3_k8["observed_delta_f1"],
    "ci_lower_95_f1": b_mean_i3_k8["ci_lower_95"],
    "ci_upper_95_f1": b_mean_i3_k8["ci_upper_95"],
    "p_value_f1": b_mean_i3_k8["p_value"],
    "observed_delta_tokens": b_mean_i3_k8["observed_delta_tokens"],
    "ci_lower_95_tokens": b_mean_i3_k8["tokens_ci_lower_95"],
    "ci_upper_95_tokens": b_mean_i3_k8["tokens_ci_upper_95"],
    "significant_f1": b_mean_i3_k8["ci_lower_95"] > 0 or b_mean_i3_k8["ci_upper_95"] < 0,
})

b_mean_i4_i3 = compute_paper_clustered_bootstrap(paper_ids, i4_mean_f1, i3_mean_f1, i4_mean_tok, i3_mean_tok, seed=42)
rows.append({
    "aggregation_level": "Across-Seed Mean Policy (3 Seeds)",
    "comparison": "I4-T3 (Across-Seed Mean) vs I3-T3 (Across-Seed Mean)",
    "observed_delta_f1": b_mean_i4_i3["observed_delta_f1"],
    "ci_lower_95_f1": b_mean_i4_i3["ci_lower_95"],
    "ci_upper_95_f1": b_mean_i4_i3["ci_upper_95"],
    "p_value_f1": b_mean_i4_i3["p_value"],
    "observed_delta_tokens": b_mean_i4_i3["observed_delta_tokens"],
    "ci_lower_95_tokens": b_mean_i4_i3["tokens_ci_lower_95"],
    "ci_upper_95_tokens": b_mean_i4_i3["tokens_ci_upper_95"],
    "significant_f1": b_mean_i4_i3["ci_lower_95"] > 0 or b_mean_i4_i3["ci_upper_95"] < 0,
})

df_out = pd.DataFrame(rows)
df_out.to_csv("results/week5/content_allocator/w5b/bootstrap/w5b_bootstrap_comparisons.csv", index=False)
print("Updated results/week5/content_allocator/w5b/bootstrap/w5b_bootstrap_comparisons.csv with 12 rows.")
