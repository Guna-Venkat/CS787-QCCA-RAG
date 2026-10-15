"""Smoke test script for Week 7-C2: Utility-Aware Budget Ranking and Continuation Policy.

Runs on 15 TRAIN-FIT and 5 INTERNAL-VAL queries.
Verifies end-to-end W7-C2 pipeline:
1. Candidate feature loading and group alignment (5 rows per query)
2. Lexicographic relevance grade generation (0 to 4)
3. Non-monotonicity and oracle stability diagnostics
4. M1 Direct F1 Regressor training & cost-aware near-tie selection
5. M2 Query-Grouped Ranker training & predicted ranking
6. Programmatic model selection on INTERNAL-VAL (enforcing 0.003 F1 tolerance rule)
7. Corrected 4-way Error Taxonomy (E1, E2, E3, E4)
8. Saves diagnostic smoke JSON marked SMOKE_ONLY_NOT_FOR_REPORTING
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.week7.w7c2.data import (
    ALLOWED_BUDGETS,
    assert_test_disabled,
    build_w7c2_ranking_dataset,
    load_w7c2_splits,
)
from src.week7.w7c2.diagnostics import (
    compute_adjacent_marginal_gains,
    compute_admissible_set_distribution,
    compute_best_vs_second_margin,
    compute_monotonicity_analysis,
)
from src.week7.w7c2.evaluation import (
    compute_corrected_error_taxonomy,
    compute_recovered_oracle_headroom,
)
from src.week7.w7c2.policy import (
    apply_m1_regression_decision_rule,
    apply_m2_ranking_decision_rule,
    evaluate_downstream_policy,
    select_best_model_internal_val,
)
from src.week7.w7c2.ranking import QueryGroupedBudgetRanker
from src.week7.w7c2.regression import DirectF1Regressor
from src.week7.w7c2.targets import (
    compute_continuation_target,
    compute_lexicographic_relevance_grades,
)
from src.week7.w7c.leakage import assert_zero_feature_leakage, sanitize_feature_matrix
from src.week7.w7c.training import PreprocessingPipeline, get_ablation_feature_names

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_w7c2")

assert_test_disabled("train")

SMOKE_DIR = Path("results/week7/w7c2_utility_ranking/diagnostics/smoke")
SMOKE_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Loading manifests and response matrices...")
df_fit_manifest, df_val_manifest = load_w7c2_splits()
df_train_matrix = pd.read_csv("results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_matrix.csv")
df_train_matrix["question_id"] = df_train_matrix["question_id"].astype(str).str.strip()
df_train_matrix["paper_id"] = df_train_matrix["paper_id"].astype(str).str.strip()

# Subsets for smoke
fit_qids_smoke = df_fit_manifest["question_id"].head(15).tolist()
val_qids_smoke = df_val_manifest["question_id"].head(5).tolist()

df_fit_matrix_smoke = df_train_matrix[df_train_matrix["question_id"].isin(fit_qids_smoke)].copy().reset_index(drop=True)
df_val_matrix_smoke = df_train_matrix[df_train_matrix["question_id"].isin(val_qids_smoke)].copy().reset_index(drop=True)

# Run Diagnostics D1 - D4 on smoke fit
logger.info("Running Oracle Stability and Non-Monotonicity Diagnostics...")
margins_arr, margin_summary = compute_best_vs_second_margin(df_fit_matrix_smoke)
sizes_arr, df_adm_dist = compute_admissible_set_distribution(df_fit_matrix_smoke)
mono_summary = compute_monotonicity_analysis(df_fit_matrix_smoke)
df_gains = compute_adjacent_marginal_gains(df_fit_matrix_smoke)

logger.info("Non-monotonic queries in smoke subset: %d / %d (%.1f%%)",
            mono_summary["non_monotonic_queries"], mono_summary["total_queries"], mono_summary["non_monotonic_pct"])

# Load cached W7-C1 candidate features
features_path = Path("results/week7/w7c_retrieval_aware_allocator/features/train_candidate_budget_features.parquet")
if features_path.exists():
    df_train_feats = pd.read_parquet(features_path)
    df_train_feats["question_id"] = df_train_feats["question_id"].astype(str).str.strip()
else:
    raise FileNotFoundError(f"W7-C1 feature cache not found at {features_path}!")

# Build ranking datasets
df_fit_rank, fit_groups = build_w7c2_ranking_dataset(
    candidate_features_df=df_train_feats[df_train_feats["question_id"].isin(fit_qids_smoke)],
    response_matrix_df=df_fit_matrix_smoke,
    split_name="train_fit",
)

df_val_rank, val_groups = build_w7c2_ranking_dataset(
    candidate_features_df=df_train_feats[df_train_feats["question_id"].isin(val_qids_smoke)],
    response_matrix_df=df_val_matrix_smoke,
    split_name="internal_val",
)

logger.info("Built smoke ranking datasets: FIT=%d rows (%s groups), VAL=%d rows (%s groups)",
            len(df_fit_rank), len(fit_groups), len(df_val_rank), len(val_groups))

# Select features (Family F4)
f4_features = get_ablation_feature_names("F4", df_fit_rank.columns)
assert_zero_feature_leakage(f4_features)

pipe = PreprocessingPipeline()
X_fit_clean = sanitize_feature_matrix(df_fit_rank, f4_features)
X_val_clean = sanitize_feature_matrix(df_val_rank, f4_features)

X_fit_scaled = pipe.fit_transform(X_fit_clean)
X_val_scaled = pipe.transform(X_val_clean)

y_fit_f1 = df_fit_rank["target_f1"].values
y_fit_grades = df_fit_rank["target_relevance_grade"].values

# Model Family M1: Direct Response-Surface Regressor
logger.info("Fitting M1 DirectF1Regressor...")
m1_reg = DirectF1Regressor(max_depth=3, n_estimators=20, random_state=42)
m1_reg.fit(X_fit_scaled, y_fit_f1)
val_pred_f1 = m1_reg.predict(X_val_scaled)

# Model Family M2: Query-Grouped Learning-to-Rank
logger.info("Fitting M2 QueryGroupedBudgetRanker...")
m2_ranker = QueryGroupedBudgetRanker(max_depth=3, n_estimators=20, random_state=42)
m2_ranker.fit(X_fit_scaled, y_fit_grades, fit_groups)
val_pred_scores = m2_ranker.predict(X_val_scaled)

# Evaluate M1 across delta grid
val_f1_by_qid = {}
val_scores_by_qid = {}
for idx, r in df_val_rank.iterrows():
    qid = str(r["question_id"])
    k = int(r["k"])
    if qid not in val_f1_by_qid:
        val_f1_by_qid[qid] = {}
        val_scores_by_qid[qid] = {}
    val_f1_by_qid[qid][k] = float(val_pred_f1[idx])
    val_scores_by_qid[qid][k] = float(val_pred_scores[idx])

model_candidates = []

# M1 variants
for delta in [0.00, 0.01, 0.02]:
    pred_ks_m1 = {
        qid: apply_m1_regression_decision_rule(k_map, delta=delta)
        for qid, k_map in val_f1_by_qid.items()
    }
    m1_metrics = evaluate_downstream_policy(pred_ks_m1, df_val_matrix_smoke)
    model_candidates.append({
        "model_name": f"M1_delta_{delta}",
        "mean_f1": m1_metrics["mean_f1"],
        "mean_tokens": m1_metrics["mean_tokens"],
        "mean_k": m1_metrics["mean_k"],
        "complexity_rank": 1,
        "pred_ks": pred_ks_m1,
    })

# M2 variants
for margin in [0.00, 0.02, 0.05]:
    pred_ks_m2 = {
        qid: apply_m2_ranking_decision_rule(k_map, score_tie_margin=margin)
        for qid, k_map in val_scores_by_qid.items()
    }
    m2_metrics = evaluate_downstream_policy(pred_ks_m2, df_val_matrix_smoke)
    model_candidates.append({
        "model_name": f"M2_margin_{margin}",
        "mean_f1": m2_metrics["mean_f1"],
        "mean_tokens": m2_metrics["mean_tokens"],
        "mean_k": m2_metrics["mean_k"],
        "complexity_rank": 2,
        "pred_ks": pred_ks_m2,
    })

# Select winner via programmatic selection rule
winning_record = select_best_model_internal_val(model_candidates, tolerance_f1=0.003)
logger.info("Winning Model on Smoke VAL: %s (F1=%.4f, tokens=%.1f, mean_k=%.2f)",
            winning_record["model_name"], winning_record["mean_f1"], winning_record["mean_tokens"], winning_record["mean_k"])

# Compute corrected 4-way error taxonomy on smoke val
df_val_eval = df_val_matrix_smoke.copy()
df_val_eval["pred_k"] = [winning_record["pred_ks"][qid] for qid in df_val_eval["question_id"]]
df_val_detailed, df_error_summary = compute_corrected_error_taxonomy(df_val_eval)

logger.info("Error Taxonomy Distribution (Smoke VAL):")
for _, r in df_error_summary.iterrows():
    logger.info("  %s: %d (%.1f%%)", r["category"], r["count"], r["pct_queries"])

# Save smoke summary artifact
smoke_out = {
    "status": "SMOKE_ONLY_NOT_FOR_REPORTING",
    "notes": "Verified end-to-end W7-C2 pipeline on 15 TRAIN-FIT and 5 INTERNAL-VAL queries.",
    "n_train_fit_queries": len(fit_qids_smoke),
    "n_val_queries": len(val_qids_smoke),
    "non_monotonic_pct": mono_summary["non_monotonic_pct"],
    "margin_summary": margin_summary,
    "winning_model": winning_record["model_name"],
    "smoke_val_f1": winning_record["mean_f1"],
    "smoke_val_tokens": winning_record["mean_tokens"],
    "smoke_val_k": winning_record["mean_k"],
    "error_taxonomy": df_error_summary.to_dict(orient="records"),
}

with open(SMOKE_DIR / "smoke_w7c2_summary.json", "w") as f:
    json.dump(smoke_out, f, indent=2)

logger.info("Smoke test complete. Artifact saved to %s", SMOKE_DIR / "smoke_w7c2_summary.json")
