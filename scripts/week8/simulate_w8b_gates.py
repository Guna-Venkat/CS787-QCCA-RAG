"""W8-B: Gate Simulation, Threshold Sweeps, and Self-Consistency Incremental Value Analysis.

Implements:
1. Feature extraction for repaired U1..U12 features.
2. Self-consistency feature extraction (Jaccard, Disagreement, Unique count).
3. 5-Fold Stratified Cross-Validation for B0 (Logistic), B1 (GBDT), B2 (MLP), and B3 (MLP + SC).
4. P(needs_promotion | high disagreement) vs P(needs_promotion | low disagreement) and AUC_sc.
5. Threshold sweep (tau in [0.1..0.9]) reporting:
   - promotion recall
   - promotion precision
   - safe-stop accuracy
   - missed-help rate (FN / NeedsPromotion)
   - unnecessary-promotion rate (FP / SafeStop)
   - downstream oracle-assisted F1
   - promotion rate
   - cumulative / mean tokens
   - Oracle Headroom Retention = (F1_gate - F1_k4) / (F1_oracle - F1_k4)
6. Comparison of GATE-0, GATE-1, GATE-2 (B2), and GATE-3 (B3).
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
import torch
import torch.nn as nn
import torch.optim as optim

from src.week8.uncertainty import (
    compute_uncertainty_features_for_row,
    compute_expected_calibration_error,
    SmallMLPClassifier,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dataset() -> pd.DataFrame:
    # 1. Draft diagnostics (U1..U12)
    draft_path = REPO_ROOT / "results/week8/drafts/draft_diagnostics_internal_val.jsonl"
    u_rows = []
    with open(draft_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                u = compute_uncertainty_features_for_row(r)
                u["question_id"] = str(r["question_id"])
                u_rows.append(u)
    df_u = pd.DataFrame(u_rows)

    # 2. Self-consistency (N=3)
    sc_path = REPO_ROOT / "results/week8/drafts/self_consistency_internal_val.jsonl"
    sc_rows = []
    with open(sc_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                sc_rows.append({
                    "question_id": str(r["question_id"]),
                    "sc_jaccard": float(r.get("sc_mean_pairwise_jaccard", r.get("lexical_agreement", 1.0))),
                    "sc_disagreement": float(r.get("has_disagreement", 0.0)),
                    "sc_num_unique": float(r.get("num_unique_answers", 1.0)),
                })
    df_sc = pd.DataFrame(sc_rows)

    # 3. Promotion Surface Oracle
    prom_path = REPO_ROOT / "results/week8/surfaces/promotion_surface_internal_val.jsonl"
    proms_by_qid = defaultdict(list)
    with open(prom_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                proms_by_qid[str(r["question_id"])].append(r)

    oracle_rows = []
    for qid, cands in proms_by_qid.items():
        b_f1 = float(cands[0]["base_f1"])
        m_delta = max(float(c["delta_f1"]) for c in cands)
        o_f1 = b_f1 + max(0.0, m_delta)
        b_tokens = int(cands[0]["base_context_tokens"])
        best_c = max(cands, key=lambda c: float(c["delta_f1"]))
        p_tokens = int(best_c["promoted_context_tokens"])
        oracle_rows.append({
            "question_id": qid,
            "needs_promotion": int(m_delta > 0.01),
            "max_delta": m_delta,
            "base_f1": b_f1,
            "oracle_f1": o_f1,
            "base_tokens": b_tokens,
            "promoted_tokens": p_tokens,
        })
    df_oracle = pd.DataFrame(oracle_rows)

    # Merge
    merged = pd.merge(df_u, df_sc, on="question_id")
    merged = pd.merge(merged, df_oracle, on="question_id")
    logger.info("Loaded complete dataset with %d queries.", len(merged))
    return merged


def train_mlp_cv(X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> np.ndarray:
    oof_probs = np.zeros(len(y), dtype=np.float32)
    device = "cpu"
    in_dim = X.shape[1]

    for train_idx, val_idx in cv.split(X, y):
        X_tr, y_tr = torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(y[train_idx], dtype=torch.float32)
        X_va = torch.tensor(X[val_idx], dtype=torch.float32)

        mlp = SmallMLPClassifier(in_dim, hidden_dim=32).to(device)
        optimizer = optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        mlp.train()
        for _ in range(60):
            optimizer.zero_grad()
            out = mlp(X_tr)
            loss = criterion(out, y_tr)
            loss.backward()
            optimizer.step()

        mlp.eval()
        with torch.no_grad():
            logits = mlp(X_va)
            probs = torch.sigmoid(logits).cpu().numpy()
            oof_probs[val_idx] = probs

    return oof_probs


def evaluate_models_oof(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    u_cols = [
        "u1_entropy_mean", "u2_entropy_max", "u3_margin_mean", "u4_margin_min",
        "u5_logprob_mean", "u6_seq_logprob", "u7_answer_len", "u8_eos_conf",
        "u9_empty_answer", "u10_repetition_ratio", "u11_lexical_overlap", "u12_semantic_sim"
    ]
    sc_cols = ["sc_jaccard", "sc_disagreement", "sc_num_unique"]

    X_u = df[u_cols].fillna(0.0).values
    X_u_sc = df[u_cols + sc_cols].fillna(0.0).values
    y = df["needs_promotion"].values
    weights_gw = 1.0 + 5.0 * np.maximum(0.0, df["max_delta"].values)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds: Dict[str, np.ndarray] = {}

    # B0: Logistic Regression (Standard, OOF)
    oof_b0 = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X_u, y):
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X_u[train_idx], y[train_idx])
        oof_b0[val_idx] = clf.predict_proba(X_u[val_idx])[:, 1]
    oof_preds["B0_Logistic"] = oof_b0

    # B0-GW: Gain-Weighted Logistic Regression (OOF)
    # w_q = 1 + alpha * max(0, Delta_q^*)
    oof_b0_gw = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X_u, y):
        clf_gw = LogisticRegression(max_iter=500, random_state=42)
        clf_gw.fit(X_u[train_idx], y[train_idx], sample_weight=weights_gw[train_idx])
        oof_b0_gw[val_idx] = clf_gw.predict_proba(X_u[val_idx])[:, 1]
    oof_preds["B0_GW_Logistic"] = oof_b0_gw

    # B1: GBDT (OOF)
    oof_b1 = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X_u, y):
        clf = HistGradientBoostingClassifier(random_state=42, max_iter=100)
        clf.fit(X_u[train_idx], y[train_idx])
        oof_b1[val_idx] = clf.predict_proba(X_u[val_idx])[:, 1]
    oof_preds["B1_GBDT"] = oof_b1

    # B2: MLP on Uncertainty Features (OOF)
    oof_b2 = train_mlp_cv(X_u, y, cv)
    oof_preds["B2_MLP"] = oof_b2

    # B3: MLP on Uncertainty + Self-Consistency Features (OOF)
    oof_b3 = train_mlp_cv(X_u_sc, y, cv)
    oof_preds["B3_MLP_plus_SC"] = oof_b3

    # Predictive Metrics
    metrics: Dict[str, Dict[str, float]] = {}
    for name, p in oof_preds.items():
        roc_auc = float(roc_auc_score(y, p))
        prec, rec, _ = precision_recall_curve(y, p)
        pr_auc = float(auc(rec, prec))
        brier = float(brier_score_loss(y, p))
        ece = float(compute_expected_calibration_error(p, y))
        acc = float(np.mean((p >= 0.5).astype(int) == y))

        metrics[name] = {
            "Accuracy": round(acc, 4),
            "ROC_AUC": round(roc_auc, 4),
            "PR_AUC": round(pr_auc, 4),
            "Brier": round(brier, 4),
            "ECE": round(ece, 4),
        }

    return metrics, oof_preds


def simulate_gate_threshold_sweep(df: pd.DataFrame, probs: np.ndarray, thresholds: List[float]) -> pd.DataFrame:
    base_f1_mean = df["base_f1"].mean()
    oracle_f1_mean = df["oracle_f1"].mean()
    oracle_headroom = oracle_f1_mean - base_f1_mean

    y_true = df["needs_promotion"].values
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    pos_deltas = np.maximum(0.0, df["max_delta"].values)
    total_pos_delta = np.sum(pos_deltas)

    rows = []
    for tau in thresholds:
        preds = (probs >= tau).astype(int)

        # TP, FP, TN, FN
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tn = np.sum((preds == 0) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))

        # Standard rates
        recall = tp / n_pos if n_pos > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        safe_stop_acc = tn / n_neg if n_neg > 0 else 0.0
        missed_help_rate = fn / n_pos if n_pos > 0 else 0.0
        unnecessary_prom_rate = fp / n_neg if n_neg > 0 else 0.0
        prom_rate = np.mean(preds)

        # Weighted Headroom Recall (WHR)
        # Sum of achievable gains caught by the gate / total achievable gain
        caught_delta = np.sum(np.where(preds == 1, pos_deltas, 0.0))
        whr = caught_delta / total_pos_delta if total_pos_delta > 0 else 0.0

        # Downstream QA F1 & Tokens
        realized_f1 = np.where(preds == 1, df["oracle_f1"], df["base_f1"]).mean()
        realized_tokens = np.where(preds == 1, df["promoted_tokens"], df["base_tokens"]).mean()
        headroom_retention = (realized_f1 - base_f1_mean) / oracle_headroom if oracle_headroom > 0 else 0.0

        rows.append({
            "tau": tau,
            "promotion_rate": round(prom_rate * 100, 1),
            "promotion_recall": round(recall * 100, 1),
            "whr": round(whr * 100, 1),
            "promotion_precision": round(prec * 100, 1),
            "safe_stop_accuracy": round(safe_stop_acc * 100, 1),
            "missed_help_rate": round(missed_help_rate * 100, 1),
            "unnecessary_prom_rate": round(unnecessary_prom_rate * 100, 1),
            "realized_f1": round(realized_f1, 4),
            "headroom_retention": round(headroom_retention * 100, 1),
            "mean_context_tokens": int(round(realized_tokens)),
        })

    return pd.DataFrame(rows)


def main():
    df = load_dataset()

    print("=" * 80)
    print("SECTION 1: SELF-CONSISTENCY EMPIRICAL BEHAVIOR")
    print("=" * 80)
    p_needs_high_dis = df[df["sc_disagreement"] == 1.0]["needs_promotion"].mean()
    p_needs_low_dis = df[df["sc_disagreement"] == 0.0]["needs_promotion"].mean()
    auc_sc_jaccard = roc_auc_score(df["needs_promotion"], 1.0 - df["sc_jaccard"])
    auc_sc_disagree = roc_auc_score(df["needs_promotion"], df["sc_disagreement"])

    print(f"P(needs_promotion | high disagreement): {p_needs_high_dis:.4f} ({p_needs_high_dis*100:.1f}%)")
    print(f"P(needs_promotion | low disagreement):  {p_needs_low_dis:.4f} ({p_needs_low_dis*100:.1f}%)")
    print(f"Self-Consistency Disagreement Lift:     {(p_needs_high_dis - p_needs_low_dis):.4f}")
    print(f"AUC (1 - Jaccard Agreement alone):      {auc_sc_jaccard:.4f}")
    print(f"AUC (Disagreement Indicator alone):     {auc_sc_disagree:.4f}")

    print("\n" + "=" * 80)
    print("SECTION 2: SUFFICIENCY CLASSIFIER EVALUATION (5-FOLD CV OOF)")
    print("=" * 80)
    model_metrics, oof_preds = evaluate_models_oof(df)
    metrics_df = pd.DataFrame(model_metrics).T
    print(metrics_df.to_string())

    delta_auc_b3_b2 = model_metrics["B3_MLP_plus_SC"]["ROC_AUC"] - model_metrics["B2_MLP"]["ROC_AUC"]
    print(f"\nIncremental AUC of Self-Consistency (B3 vs B2): Delta AUC = {delta_auc_b3_b2:+.4f}")

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]

    print("\n" + "=" * 80)
    print("SECTION 3: THRESHOLD SWEEPS FOR B0, B0-GW, B2, B3")
    print("=" * 80)
    b0_sweep = simulate_gate_threshold_sweep(df, oof_preds["B0_Logistic"], thresholds)
    b0_gw_sweep = simulate_gate_threshold_sweep(df, oof_preds["B0_GW_Logistic"], thresholds)
    b2_sweep = simulate_gate_threshold_sweep(df, oof_preds["B2_MLP"], thresholds)
    b3_sweep = simulate_gate_threshold_sweep(df, oof_preds["B3_MLP_plus_SC"], thresholds)

    print("\n--- B0 (Standard Logistic) Sweep ---")
    print(b0_sweep[["tau", "promotion_rate", "promotion_recall", "whr", "promotion_precision", "safe_stop_accuracy", "realized_f1", "headroom_retention", "mean_context_tokens"]].to_string(index=False))

    print("\n--- B0-GW (Gain-Weighted Logistic) Sweep ---")
    print(b0_gw_sweep[["tau", "promotion_rate", "promotion_recall", "whr", "promotion_precision", "safe_stop_accuracy", "realized_f1", "headroom_retention", "mean_context_tokens"]].to_string(index=False))

    print("\n--- B2 (MLP) Sweep ---")
    print(b2_sweep[["tau", "promotion_rate", "promotion_recall", "whr", "promotion_precision", "safe_stop_accuracy", "realized_f1", "headroom_retention", "mean_context_tokens"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("SECTION 4: CANONICAL GATE BASELINES (USING FROZEN PROJECT SELECTION RULES)")
    print("=" * 80)

    base_f1 = df["base_f1"].mean()
    oracle_f1 = df["oracle_f1"].mean()
    base_tokens = df["base_tokens"].mean()
    prom_tokens = df["promoted_tokens"].mean()

    # Find cost-sensitive sweet spots (targeting recall >= 90% and high WHR)
    b0_opt = b0_sweep[b0_sweep["tau"] == 0.20].iloc[0]
    b0_gw_opt = b0_gw_sweep[b0_gw_sweep["tau"] == 0.25].iloc[0]
    b2_opt = b2_sweep[b2_sweep["tau"] == 0.25].iloc[0]
    b3_opt = b3_sweep[b3_sweep["tau"] == 0.25].iloc[0]

    gate_table = [
        {
            "Gate Policy": "GATE-0: Always Stop (Base k=4)",
            "Prom Rate": "0.0%",
            "Recall": "0.0%",
            "WHR": "0.0%",
            "Realized F1": f"{base_f1:.4f}",
            "Headroom Ret.": "0.0%",
            "Tokens": f"{int(round(base_tokens))}",
            "Latency/Q": "0.0s",
        },
        {
            "Gate Policy": "GATE-1: Always Escalate + Oracle Selector",
            "Prom Rate": "100.0%",
            "Recall": "100.0%",
            "WHR": "100.0%",
            "Realized F1": f"{oracle_f1:.4f}",
            "Headroom Ret.": "100.0%",
            "Tokens": f"{int(round(prom_tokens))}",
            "Latency/Q": "+2.8s",
        },
        {
            "Gate Policy": "GATE-2A: B0 Logistic (tau=0.20)",
            "Prom Rate": f"{b0_opt['promotion_rate']}%",
            "Recall": f"{b0_opt['promotion_recall']}%",
            "WHR": f"{b0_opt['whr']}%",
            "Realized F1": f"{b0_opt['realized_f1']:.4f}",
            "Headroom Ret.": f"{b0_opt['headroom_retention']}%",
            "Tokens": f"{b0_opt['mean_context_tokens']}",
            "Latency/Q": "+0.5s",
        },
        {
            "Gate Policy": "GATE-2B: B0-GW Logistic (tau=0.25)",
            "Prom Rate": f"{b0_gw_opt['promotion_rate']}%",
            "Recall": f"{b0_gw_opt['promotion_recall']}%",
            "WHR": f"{b0_gw_opt['whr']}%",
            "Realized F1": f"{b0_gw_opt['realized_f1']:.4f}",
            "Headroom Ret.": f"{b0_gw_opt['headroom_retention']}%",
            "Tokens": f"{b0_gw_opt['mean_context_tokens']}",
            "Latency/Q": "+0.5s",
        },
        {
            "Gate Policy": "GATE-2C: B2 MLP (tau=0.25)",
            "Prom Rate": f"{b2_opt['promotion_rate']}%",
            "Recall": f"{b2_opt['promotion_recall']}%",
            "WHR": f"{b2_opt['whr']}%",
            "Realized F1": f"{b2_opt['realized_f1']:.4f}",
            "Headroom Ret.": f"{b2_opt['headroom_retention']}%",
            "Tokens": f"{b2_opt['mean_context_tokens']}",
            "Latency/Q": "+0.8s",
        },
        {
            "Gate Policy": "GATE-3: B3 B2+SC (tau=0.25, Research Ablation)",
            "Prom Rate": f"{b3_opt['promotion_rate']}%",
            "Recall": f"{b3_opt['promotion_recall']}%",
            "WHR": f"{b3_opt['whr']}%",
            "Realized F1": f"{b3_opt['realized_f1']:.4f}",
            "Headroom Ret.": f"{b3_opt['headroom_retention']}%",
            "Tokens": f"{b3_opt['mean_context_tokens']}",
            "Latency/Q": "+8.6s (SC cost)",
        },
    ]

    df_gates = pd.DataFrame(gate_table)
    print(df_gates.to_string(index=False))

    out_dir = REPO_ROOT / "results/week8/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_gates.to_csv(out_dir / "w8b_gate_baselines.csv", index=False)
    b0_sweep.to_csv(out_dir / "w8b_b0_threshold_sweep.csv", index=False)
    b0_gw_sweep.to_csv(out_dir / "w8b_b0_gw_threshold_sweep.csv", index=False)
    b2_sweep.to_csv(out_dir / "w8b_b2_threshold_sweep.csv", index=False)
    b3_sweep.to_csv(out_dir / "w8b_b3_threshold_sweep.csv", index=False)
    metrics_df.to_csv(out_dir / "w8b_sufficiency_classifiers_cv.csv")
    print(f"\nAll W8-B tables successfully saved to {out_dir}/.")


if __name__ == "__main__":
    main()
