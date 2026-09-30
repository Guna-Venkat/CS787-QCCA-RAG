"""Evaluation, metric calculation, and frozen response surface lookup for Week 4."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CONTEXT_TOKEN_COSTS = {
    0: 138.8,
    2: 534.5,
    4: 946.1,
    5: 1157.0,
    6: 1359.1,
    8: 1758.5,
    10: 2137.9
}


def normalize_qid(qid: Any) -> str:
    """Normalize question_id to clean integer string without .0 suffix."""
    s = str(qid).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


class FrozenResponseSurfaceLookup:
    """Fast, immutable lookup table for F1 scores from the Week-2 matrix."""
    
    def __init__(self, matrix_path: str = "results/week2/processed/dev_per_query_k_matrix.csv"):
        p = Path(matrix_path)
        if not p.exists():
            raise FileNotFoundError(f"Matrix artifact not found: {p}")
        self.df_matrix = pd.read_csv(p)
        self.lookup_: Dict[Tuple[str, int], float] = {}
        
        # Populate lookup map
        for _, row in self.df_matrix.iterrows():
            qid = normalize_qid(row["question_id"])
            for k in [0, 2, 4, 5, 6, 8, 10]:
                col_name = f"F1_k{k}"
                if col_name in row:
                    self.lookup_[(qid, k)] = float(row[col_name])
                    
    def get_f1(self, question_id: Any, k: int) -> float:
        """Retrieve frozen F1 score for (question_id, k)."""
        key = (normalize_qid(question_id), int(k))
        if key not in self.lookup_:
            raise KeyError(f"Missing frozen response record for question_id={question_id}, k={k}")
        return self.lookup_[key]
        
    def get_context_tokens(self, k: int) -> float:
        """Retrieve fixed empirical mean context tokens for budget k."""
        if k not in CONTEXT_TOKEN_COSTS:
            raise KeyError(f"Unknown budget k={k} for context token cost")
        return CONTEXT_TOKEN_COSTS[k]


def evaluate_allocator_predictions(
    df_preds: pd.DataFrame,
    pred_col: str,
    lookup: FrozenResponseSurfaceLookup,
    target_col: str = "target_k",
    static_k_baseline: int = 8,
    oracle_col: Optional[str] = "f1_quality_oracle"
) -> Dict[str, Any]:
    """Calculate comprehensive quality, efficiency, regret, and headroom recovery metrics."""
    f1_list = []
    tokens_list = []
    k_list = []
    eps_regret_list = []
    oracle_regret_list = []
    exact_target_match = []
    
    over_comp_count = 0
    under_comp_count = 0
    exact_count = 0
    
    for _, row in df_preds.iterrows():
        qid = str(row["question_id"])
        pred_k = int(row[pred_col])
        true_k = int(row[target_col])
        
        f1_pred = lookup.get_f1(qid, pred_k)
        f1_true = lookup.get_f1(qid, true_k)
        f1_list.append(f1_pred)
        tokens_list.append(lookup.get_context_tokens(pred_k))
        k_list.append(pred_k)
        
        # Regret vs epsilon target: F1(target) - F1(pred)
        # Note: can be negative if predicted k achieves higher F1 than target k
        eps_regret = max(0.0, f1_true - f1_pred)
        eps_regret_list.append(eps_regret)
        
        # Regret vs Quality Oracle
        if oracle_col in row and not pd.isna(row[oracle_col]):
            oracle_f1_val = float(row[oracle_col])
        else:
            # compute max over K_alloc = [2, 4, 5, 6, 8]
            oracle_f1_val = max(lookup.get_f1(qid, k_cand) for k_cand in [2, 4, 5, 6, 8])
            
        oracle_regret = max(0.0, oracle_f1_val - f1_pred)
        oracle_regret_list.append(oracle_regret)
        
        # Target accuracy
        is_match = (pred_k == true_k)
        exact_target_match.append(is_match)
        
        if pred_k < true_k:
            over_comp_count += 1
        elif pred_k > true_k:
            under_comp_count += 1
        else:
            exact_count += 1
            
    n_queries = len(df_preds)
    mean_f1 = float(np.mean(f1_list))
    mean_tokens = float(np.mean(tokens_list))
    mean_k = float(np.mean(k_list))
    mean_eps_regret = float(np.mean(eps_regret_list))
    mean_oracle_regret = float(np.mean(oracle_regret_list))
    target_acc = float(np.mean(exact_target_match))
    
    # Baseline comparison vs Static k=8
    static_tokens = lookup.get_context_tokens(static_k_baseline)
    context_reduction = float((static_tokens - mean_tokens) / static_tokens * 100.0)
    
    # Oracle headroom recovery
    # Static k=8 mean F1 on this set:
    static_f1s = [lookup.get_f1(str(r["question_id"]), static_k_baseline) for _, r in df_preds.iterrows()]
    mean_static_f1 = float(np.mean(static_f1s))
    
    oracle_f1s = [max(lookup.get_f1(str(r["question_id"]), k_cand) for k_cand in [2, 4, 5, 6, 8]) for _, r in df_preds.iterrows()]
    mean_oracle_f1 = float(np.mean(oracle_f1s))
    
    headroom_available = mean_oracle_f1 - mean_static_f1
    gain_over_static = mean_f1 - mean_static_f1
    
    if headroom_available > 1e-6:
        headroom_recovery_pct = float(gain_over_static / headroom_available * 100.0)
    else:
        headroom_recovery_pct = 0.0
        
    return {
        "pred_col": pred_col,
        "n_queries": n_queries,
        "mean_f1": round(mean_f1, 4),
        "mean_k": round(mean_k, 2),
        "mean_context_tokens": round(mean_tokens, 1),
        "context_reduction_pct_vs_k8": round(context_reduction, 2),
        "mean_epsilon_regret": round(mean_eps_regret, 4),
        "mean_oracle_regret": round(mean_oracle_regret, 4),
        "target_accuracy": round(target_acc, 4),
        "mean_static_f1": round(mean_static_f1, 4),
        "mean_oracle_f1": round(mean_oracle_f1, 4),
        "absolute_gain_vs_static": round(gain_over_static, 4),
        "oracle_headroom_recovered_pct": round(headroom_recovery_pct, 2),
        "error_distribution": {
            "exact_match_pct": round(exact_count / n_queries * 100.0, 2),
            "over_compression_pct": round(over_comp_count / n_queries * 100.0, 2),
            "under_compression_pct": round(under_comp_count / n_queries * 100.0, 2)
        }
    }
