"""Feature Signal and Allocator Feasibility Analysis for Week 3.

Explores associations between pre-generation retrieval features and
observed allocation targets (k*_qual, k*_eps, oracle_gain, f1_range).
Adheres strictly to scientific reporting guardrails: treats associations as
exploratory signal without claiming causal prediction or final model accuracy.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "rho_1",
    "delta_12",
    "entropy",
    "entropy_normalized",
    "n_high",
    "query_token_length"
]


def load_and_align_features(
    feature_path: str,
    oracle_df: pd.DataFrame,
    epsilon_df: pd.DataFrame,
    het_df: pd.DataFrame,
    selected_eps: float = 0.01
) -> pd.DataFrame:
    """Load retrieval features and strictly join with oracle targets by question_id."""
    feat_records = []
    with open(feature_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                feat_records.append(json.loads(line))
                
    df_feat = pd.DataFrame(feat_records)
    df_feat["question_id"] = df_feat["question_id"].astype(str)
    
    # Check duplicates and alignment
    assert not df_feat["question_id"].duplicated().any(), "Duplicate question_id found in features!"
    
    # Filter epsilon df to selected_eps
    df_eps_sub = epsilon_df[epsilon_df["epsilon"] == selected_eps].copy()
    df_eps_sub["question_id"] = df_eps_sub["question_id"].astype(str)
    
    # Ensure question_id is str across all dataframes
    oracle_df = oracle_df.copy()
    oracle_df["question_id"] = oracle_df["question_id"].astype(str)
    het_df = het_df.copy()
    het_df["question_id"] = het_df["question_id"].astype(str)
    
    # Merge datasets strictly by question_id
    df_merged = df_feat.merge(
        oracle_df[["question_id", "best_k_qual", "best_f1_qual"]],
        on="question_id",
        how="inner"
    )
    df_merged = df_merged.merge(
        df_eps_sub[["question_id", "selected_k_epsilon", "f1_regret"]],
        on="question_id",
        how="inner"
    )
    df_merged = df_merged.merge(
        het_df[["question_id", "f1_range", "f1_std", "oracle_gain", "gain_gt_00"]],
        on="question_id",
        how="inner"
    )
    
    assert len(df_merged) == len(df_feat), (
        f"Alignment mismatch: features={len(df_feat)}, merged={len(df_merged)}"
    )
    logger.info("Successfully aligned %d questions across features and targets.", len(df_merged))
    return df_merged


def compute_feature_associations(df_aligned: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute Pearson, Spearman, and Kruskal-Wallis associations for each feature."""
    continuous_targets = ["oracle_gain", "f1_range"]
    ordinal_targets = ["best_k_qual", "selected_k_epsilon"]
    
    records = []
    feature_group_stats = {}
    
    for feat in FEATURE_NAMES:
        if feat not in df_aligned.columns:
            logger.warning("Feature %s not in aligned dataframe, skipping.", feat)
            continue
            
        x = df_aligned[feat].astype(float).to_numpy()
        feature_group_stats[feat] = {}
        
        # 1. Group statistics by target budget (best_k_qual)
        for target in ordinal_targets:
            groups = [
                df_aligned[df_aligned[target] == k][feat].dropna().to_numpy()
                for k in sorted(df_aligned[target].unique())
            ]
            # Kruskal-Wallis test across groups
            if len(groups) > 1 and all(len(g) > 0 for g in groups):
                kw_stat, kw_p = stats.kruskal(*groups)
            else:
                kw_stat, kw_p = 0.0, 1.0
                
            # Spearman rank correlation with ordinal target
            y_ord = df_aligned[target].astype(float).to_numpy()
            sp_r, sp_p = stats.spearmanr(x, y_ord)
            
            # Interpret effect
            abs_sp = abs(sp_r)
            if abs_sp < 0.10:
                effect = "Negligible"
            elif abs_sp < 0.30:
                effect = "Weak"
            elif abs_sp < 0.50:
                effect = "Moderate"
            else:
                effect = "Strong"
                
            interp = (
                f"Spearman rho={sp_r:.3f} (p={sp_p:.3e}), KW H={kw_stat:.2f} (p={kw_p:.3e}). "
                f"{effect} association with {target}."
            )
            
            records.append({
                "feature": feat,
                "target": target,
                "target_type": "ordinal_budget",
                "spearman_rho": round(float(sp_r), 4),
                "spearman_p": round(float(sp_p), 4),
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "kruskal_wallis_h": round(float(kw_stat), 4),
                "kruskal_wallis_p": round(float(kw_p), 4),
                "effect_size": effect,
                "interpretation": interp
            })
            
            # Group means
            if target == "best_k_qual":
                for k_val in sorted(df_aligned[target].unique()):
                    sub_vals = df_aligned[df_aligned[target] == k_val][feat].dropna()
                    feature_group_stats[feat][f"k={k_val}_mean"] = round(float(sub_vals.mean()), 4)
                    feature_group_stats[feat][f"k={k_val}_std"] = round(float(sub_vals.std()), 4)
                    
        # 2. Associations with continuous targets
        for c_target in continuous_targets:
            y_cont = df_aligned[c_target].astype(float).to_numpy()
            p_r, p_p = stats.pearsonr(x, y_cont)
            s_r, s_p = stats.spearmanr(x, y_cont)
            
            abs_s = abs(s_r)
            if abs_s < 0.10:
                effect = "Negligible"
            elif abs_s < 0.30:
                effect = "Weak"
            elif abs_s < 0.50:
                effect = "Moderate"
            else:
                effect = "Strong"
                
            interp = (
                f"Spearman rho={s_r:.3f} (p={s_p:.3e}), Pearson r={p_r:.3f} (p={p_p:.3e}). "
                f"{effect} association with {c_target}."
            )
            
            records.append({
                "feature": feat,
                "target": c_target,
                "target_type": "continuous_quality",
                "spearman_rho": round(float(s_r), 4),
                "spearman_p": round(float(s_p), 4),
                "pearson_r": round(float(p_r), 4),
                "pearson_p": round(float(p_p), 4),
                "kruskal_wallis_h": np.nan,
                "kruskal_wallis_p": np.nan,
                "effect_size": effect,
                "interpretation": interp
            })
            
    df_report = pd.DataFrame(records)
    return df_report, feature_group_stats


def run_feature_signal_analysis(
    feature_path: str,
    oracle_csv: str,
    epsilon_csv: str,
    het_csv: str,
    output_dir: str,
    selected_eps: float = 0.01
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute complete feature signal analysis."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    oracle_df = pd.read_csv(oracle_csv)
    epsilon_df = pd.read_csv(epsilon_csv)
    het_df = pd.read_csv(het_csv)
    
    df_aligned = load_and_align_features(
        feature_path, oracle_df, epsilon_df, het_df, selected_eps=selected_eps
    )
    
    df_report, group_stats = compute_feature_associations(df_aligned)
    
    report_csv = out_path / "feature_signal_analysis.csv"
    df_report.to_csv(report_csv, index=False)
    logger.info("Saved feature signal analysis table to %s", report_csv)
    
    summary = {
        "n_aligned_questions": len(df_aligned),
        "analyzed_features": FEATURE_NAMES,
        "selected_epsilon_star": selected_eps,
        "feature_group_statistics_by_best_k": group_stats,
        "top_associations": df_report.sort_values(
            by="spearman_rho", key=abs, ascending=False
        ).head(8).to_dict(orient="records")
    }
    
    sum_json = out_path / "feature_signal_summary.json"
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved feature signal summary to %s", sum_json)
    
    return df_report, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Feature Signal Analysis")
    parser.add_argument("--features", default="results/week2/processed/dev_retrieval_features.jsonl")
    parser.add_argument("--oracle", default="results/week3/processed/dev_quality_oracle.csv")
    parser.add_argument("--epsilon", default="results/week3/processed/epsilon_oracle_all.csv")
    parser.add_argument("--het", default="results/week3/processed/per_query_heterogeneity.csv")
    parser.add_argument("--output_dir", default="results/week3/processed")
    args = parser.parse_args()
    
    df_rep, sum_res = run_feature_signal_analysis(
        args.features, args.oracle, args.epsilon, args.het, args.output_dir
    )
    print("\nFeature Signal Table:")
    print(df_rep[["feature", "target", "spearman_rho", "spearman_p", "effect_size"]].to_string(index=False))
