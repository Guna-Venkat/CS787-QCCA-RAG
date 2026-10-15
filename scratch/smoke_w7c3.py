"""Smoke test script for Week 7-C3 Anchor-and-Deviate Semantic Budget Policy.

Strictly marked: SMOKE_ONLY_NOT_FOR_REPORTING.
Validates end-to-end functionality on a tiny subset of 20 queries without scientific reporting.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.week7.w7c3.data import build_w7c3_query_dataset, get_feature_subsets
from src.week7.w7c3.diagnostics import compute_4to6_gain_analysis, compute_anchor_oracle_diagnostics
from src.week7.w7c3.downshift import SafeDownshiftClassifier
from src.week7.w7c3.evaluation import (
    compute_downshift_safety_metrics,
    compute_upshift_precision_metrics,
    compute_w7c3_error_taxonomy,
)
from src.week7.w7c3.oof import evaluate_oof_policy_grid, run_group_kfold_oof
from src.week7.w7c3.policy import apply_anchor_deviate_decision_rule, select_best_anchor_policy

def main():
    print("Starting Week 7-C3 Smoke Validation...")
    output_dir = Path("results/week7/w7c3_anchor_deviate/diagnostics/smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load canonical data
    df_cand_all = pd.read_parquet("results/week7/w7c_retrieval_aware_allocator/features/train_candidate_budget_features.parquet")
    df_mat_all = pd.read_csv("results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_matrix.csv")

    df_cand_all["question_id"] = df_cand_all["question_id"].astype(str).str.strip()
    df_mat_all["question_id"] = df_mat_all["question_id"].astype(str).str.strip()
    df_cand_all["paper_id"] = df_cand_all["paper_id"].astype(str).str.strip()
    df_mat_all["paper_id"] = df_mat_all["paper_id"].astype(str).str.strip()

    # Select 20 queries across at least 4 papers for GroupKFold
    sample_papers = df_mat_all["paper_id"].unique()[:4]
    df_mat_sub = df_mat_all[df_mat_all["paper_id"].isin(sample_papers)].copy()
    if len(df_mat_sub) < 15:
        # Take first 6 papers to get ~20 queries
        sample_papers = df_mat_all["paper_id"].unique()[:6]
        df_mat_sub = df_mat_all[df_mat_all["paper_id"].isin(sample_papers)].copy()
    df_mat_sub = df_mat_sub.head(20).copy()
    valid_qids = set(df_mat_sub["question_id"])
    df_cand_sub = df_cand_all[df_cand_all["question_id"].isin(valid_qids)].copy()

    print(f"Smoke subset: {len(df_mat_sub)} queries across {df_mat_sub['paper_id'].nunique()} papers.")

    # 2. Build unified dataset
    df_unified = build_w7c3_query_dataset(
        candidate_features_df=df_cand_sub,
        response_matrix_df=df_mat_sub,
        budgets=[2, 4, 5, 6, 8],
        anchor_k=4,
        epsilon=0.01,
        split_name="train",
    )
    down_f, up_f, del_f = get_feature_subsets(df_unified)
    print(f"Features: downshift={len(down_f)}, upshift={len(up_f)}, delta={len(del_f)}")

    # 3. Diagnostics
    oracle_stats, _ = compute_anchor_oracle_diagnostics(df_mat_sub, anchor_k=4)
    gain_stats, _ = compute_4to6_gain_analysis(df_mat_sub, anchor_k=4)

    # 4. Tiny 2-fold OOF
    model_params = {
        "downshift_classifier": {"n_estimators": 10, "max_depth": 3, "min_child_samples": 2},
        "upshift_classifier": {"n_estimators": 10, "max_depth": 3, "min_child_samples": 2},
        "delta_regressors": {"n_estimators": 10, "max_depth": 3, "min_child_samples": 2},
    }
    df_oof = run_group_kfold_oof(
        df_train_query=df_unified,
        downshift_features=down_f,
        upshift_features=up_f,
        delta_features=del_f,
        n_splits=2,
        group_col="paper_id",
        model_params=model_params,
    )

    # 5. Policy grid evaluation
    df_grid = evaluate_oof_policy_grid(
        df_oof=df_oof,
        df_response_matrix=df_mat_sub,
        tau_down_grid=[0.75, 0.85],
        tau_up_grid=[0.60, 0.70],
        delta_up_grid=[0.0, 0.01],
        anchor_k=4,
    )

    f1_k4_ref = float(df_mat_sub["f1_k4"].mean())
    winner, _ = select_best_anchor_policy(df_grid, f1_k4_reference=f1_k4_ref, eta=0.005)

    # 6. Apply winning policy to generate decisions
    chosen_ks = {}
    sample_decisions = []
    for _, r in df_oof.iterrows():
        qid = r["question_id"]
        deltas = {5: r["pred_delta_5"], 6: r["pred_delta_6"], 8: r["pred_delta_8"]}
        k_pred = apply_anchor_deviate_decision_rule(
            p_down=r["p_down"],
            p_up=r["p_up"],
            pred_deltas=deltas,
            tau_down=winner["tau_down"],
            tau_up=winner["tau_up"],
            delta_up=winner["delta_up"],
            anchor_k=4,
        )
        chosen_ks[qid] = k_pred
        sample_decisions.append({
            "question_id": qid,
            "p_down": round(float(r["p_down"]), 4),
            "p_up": round(float(r["p_up"]), 4),
            "pred_deltas": {k: round(float(v), 4) for k, v in deltas.items()},
            "chosen_k": int(k_pred),
        })

    df_eval_queries = df_mat_sub.copy()
    df_eval_queries["pred_k"] = [chosen_ks[q] for q in df_eval_queries["question_id"]]

    df_detailed, df_summary_tax = compute_w7c3_error_taxonomy(df_eval_queries, anchor_k=4)
    down_safety = compute_downshift_safety_metrics(df_eval_queries, anchor_k=4)
    up_prec = compute_upshift_precision_metrics(df_eval_queries, anchor_k=4)

    summary = {
        "status": "SMOKE_ONLY_NOT_FOR_REPORTING",
        "n_queries": len(df_mat_sub),
        "f1_k4_reference": f1_k4_ref,
        "winner_policy": {
            "tau_down": winner["tau_down"],
            "tau_up": winner["tau_up"],
            "delta_up": winner["delta_up"],
            "mean_f1": winner["mean_f1"],
            "mean_tokens": winner["mean_tokens"],
            "mean_k": winner["mean_k"],
        },
        "oracle_diagnostics": oracle_stats,
        "gain_4to6_diagnostics": gain_stats,
        "downshift_safety": down_safety,
        "upshift_precision": up_prec,
        "sample_decisions": sample_decisions[:5],
        "error_taxonomy": df_summary_tax.to_dict(orient="records"),
    }

    out_file = output_dir / "smoke_w7c3_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Smoke test successful! Saved results to {out_file}")

if __name__ == "__main__":
    main()
