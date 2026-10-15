"""Small smoke test for Week 7-C: Retrieval-Aware Semantic Sufficiency Allocation.

Runs on 10 TRAIN-FIT and 5 INTERNAL-VAL queries.
Verifies end-to-end pipeline:
1. Feature extraction for Families A-H
2. Leakage verification
3. Preprocessing fit on TRAIN-FIT
4. Small LightGBM fit & prediction
5. Small Multitask MLP fit & prediction (2 epochs)
6. Tau calibration on INTERNAL-VAL
7. Downstream policy evaluation
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
import torch

from src.week7.w7c.data import (
    ALLOWED_BUDGETS,
    assert_test_disabled,
    build_candidate_budget_dataframe,
    load_w7c_dataset_bundle,
    load_w7c_fit_val_splits,
)
from src.week7.w7c.leakage import assert_zero_feature_leakage, sanitize_feature_matrix
from src.week7.w7c.models_mlp import MultitaskMLPWrapper
from src.week7.w7c.models_tree import RegretGBDTRegressor, SufficiencyGBDTClassifier
from src.week7.w7c.policy import (
    apply_sufficiency_decision_rule,
    calibrate_tau_on_internal_val,
    evaluate_policy_on_response_surface,
)
from src.week7.w7c.training import (
    PreprocessingPipeline,
    build_candidate_feature_bank,
    get_ablation_feature_names,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_w7c")

assert_test_disabled("train")

# Paths
SMOKE_OUT_DIR = Path("results/week7/w7c_retrieval_aware_allocator/diagnostics/smoke")
SMOKE_OUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Loading dataset bundle and split manifests...")
train_bundle = load_w7c_dataset_bundle("results/week5/train_response_surface/train_retrieval.jsonl", split="train")
df_fit, df_val = load_w7c_fit_val_splits()

fit_qids_subset = df_fit["question_id"].head(10).tolist()
val_qids_subset = df_val["question_id"].head(5).tolist()

train_indices = [train_bundle["question_ids"].index(qid) for qid in fit_qids_subset]
val_indices = [train_bundle["question_ids"].index(qid) for qid in val_qids_subset]

# MiniLM semantic scorer for real scores
from src.week6.models import PretrainedSemanticScorer

device = "cuda" if torch.cuda.is_available() else "cpu"
scorer = PretrainedSemanticScorer(device=device)

logger.info("Computing semantic scores for smoke subsets...")
fit_sem_scores = []
for idx in train_indices:
    q = train_bundle["questions"][idx]
    p_texts = [p.get("text", "") for p in train_bundle["passages"][idx]]
    s = scorer.score_query_passages(q, p_texts, device=device).cpu().numpy().tolist()
    fit_sem_scores.append(s)

val_sem_scores = []
for idx in val_indices:
    q = train_bundle["questions"][idx]
    p_texts = [p.get("text", "") for p in train_bundle["passages"][idx]]
    s = scorer.score_query_passages(q, p_texts, device=device).cpu().numpy().tolist()
    val_sem_scores.append(s)

# Load canonical W7-B train response matrix
df_train_matrix = pd.read_csv("results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_matrix.csv")
df_train_matrix["question_id"] = df_train_matrix["question_id"].astype(str).str.strip()
df_train_matrix["paper_id"] = df_train_matrix["paper_id"].astype(str).str.strip()

df_fit_matrix = df_train_matrix[df_train_matrix["question_id"].isin(fit_qids_subset)].copy().reset_index(drop=True)
df_val_matrix = df_train_matrix[df_train_matrix["question_id"].isin(val_qids_subset)].copy().reset_index(drop=True)

# Build candidate dataframes
df_fit_cand = build_candidate_budget_dataframe(df_fit_matrix, split_name="train_fit")
df_val_cand = build_candidate_budget_dataframe(df_val_matrix, split_name="internal_val")

logger.info("Extracted %d fit candidate rows, %d val candidate rows.", len(df_fit_cand), len(df_val_cand))

# Build candidate feature bank
df_fit_feats = build_candidate_feature_bank(
    question_ids=fit_qids_subset,
    paper_ids=[train_bundle["paper_ids"][i] for i in train_indices],
    questions=[train_bundle["questions"][i] for i in train_indices],
    passages_list=[train_bundle["passages"][i] for i in train_indices],
    bm25_scores_list=[train_bundle["bm25_scores"][i] for i in train_indices],
    semantic_scores_list=fit_sem_scores,
    split_name="train_fit",
)

df_val_feats = build_candidate_feature_bank(
    question_ids=val_qids_subset,
    paper_ids=[train_bundle["paper_ids"][i] for i in val_indices],
    questions=[train_bundle["questions"][i] for i in val_indices],
    passages_list=[train_bundle["passages"][i] for i in val_indices],
    bm25_scores_list=[train_bundle["bm25_scores"][i] for i in val_indices],
    semantic_scores_list=val_sem_scores,
    split_name="internal_val",
)

# Merge features with targets
df_fit_full = pd.merge(df_fit_cand, df_fit_feats, on=["split", "question_id", "paper_id", "k"])
df_val_full = pd.merge(df_val_cand, df_val_feats, on=["split", "question_id", "paper_id", "k"])

# Select F4 feature ablation
f4_feature_names = get_ablation_feature_names("F4", df_fit_full.columns)
logger.info("Ablation F4 features count: %d", len(f4_feature_names))

# Leakage audit
assert_zero_feature_leakage(f4_feature_names)

# Preprocessing
pipe = PreprocessingPipeline()
X_fit = sanitize_feature_matrix(df_fit_full, f4_feature_names)
X_val = sanitize_feature_matrix(df_val_full, f4_feature_names)

X_fit_scaled = pipe.fit_transform(X_fit)
X_val_scaled = pipe.transform(X_val)

y_fit_cls = df_fit_full["target_sufficient"].values
y_fit_reg = df_fit_full["target_regret"].values

# Train GBDT models
logger.info("Fitting SufficiencyGBDTClassifier...")
gbdt_clf = SufficiencyGBDTClassifier(max_depth=3, n_estimators=30, random_state=42)
gbdt_clf.fit(X_fit_scaled, y_fit_cls)
val_pred_probs = gbdt_clf.predict_proba(X_val_scaled)

logger.info("Fitting RegretGBDTRegressor...")
gbdt_reg = RegretGBDTRegressor(max_depth=3, n_estimators=30, random_state=42)
gbdt_reg.fit(X_fit_scaled, y_fit_reg)
val_pred_regret = gbdt_reg.predict(X_val_scaled)

# Train small Multitask MLP (2 epochs for smoke)
logger.info("Fitting MultitaskMLPWrapper (2 epochs)...")
mlp = MultitaskMLPWrapper(hidden_dims=(32, 16), epochs=2, batch_size=16)
mlp.fit(X_fit_scaled, y_fit_cls, y_fit_reg)
val_mlp_probs = mlp.predict_proba(X_val_scaled)

# Calibrate tau on INTERNAL-VAL
logger.info("Calibrating tau on INTERNAL-VAL...")
best_tau, df_tau_summary = calibrate_tau_on_internal_val(
    val_candidate_df=df_val_full,
    val_matrix_df=df_val_matrix,
    pred_probs=val_pred_probs,
    tau_grid=[0.50, 0.70, 0.90],
)

# Apply policy on val
probs_by_qid = {}
for idx, r in df_val_full.iterrows():
    qid = str(r["question_id"])
    k = int(r["k"])
    if qid not in probs_by_qid:
        probs_by_qid[qid] = {}
    probs_by_qid[qid][k] = float(val_pred_probs[idx])

val_pred_ks = {
    qid: apply_sufficiency_decision_rule(k_map, tau=best_tau, budgets=[2, 4, 5, 6, 8])
    for qid, k_map in probs_by_qid.items()
}

val_policy_metrics = evaluate_policy_on_response_surface(val_pred_ks, df_val_matrix)
logger.info("Smoke policy metrics: F1=%.4f, tokens=%.1f, mean_k=%.2f",
            val_policy_metrics["mean_f1"], val_policy_metrics["mean_tokens"], val_policy_metrics["mean_k"])

# Save smoke artifact marked SMOKE_ONLY_NOT_FOR_REPORTING
smoke_artifact = {
    "status": "SMOKE_ONLY_NOT_FOR_REPORTING",
    "notes": "Verified end-to-end W7-C pipeline on 10 TRAIN-FIT and 5 INTERNAL-VAL queries.",
    "n_train_fit_queries": len(fit_qids_subset),
    "n_val_queries": len(val_qids_subset),
    "n_fit_candidate_rows": len(df_fit_cand),
    "n_val_candidate_rows": len(df_val_cand),
    "f4_features_count": len(f4_feature_names),
    "sample_features": f4_feature_names[:15],
    "best_tau_val": best_tau,
    "smoke_val_f1": val_policy_metrics["mean_f1"],
    "smoke_val_tokens": val_policy_metrics["mean_tokens"],
    "smoke_val_k": val_policy_metrics["mean_k"],
}

out_path = SMOKE_OUT_DIR / "smoke_w7c_summary.json"
with open(out_path, "w") as f:
    json.dump(smoke_artifact, f, indent=2)

logger.info("Smoke verification complete. Saved to %s", out_path)
