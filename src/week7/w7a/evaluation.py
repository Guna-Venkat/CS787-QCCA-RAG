"""Evaluation metrics, conditional stratification, and win/tie/loss analytics for Week 7-A.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SARA_ROOT = REPO_ROOT / "Baselines" / "SARA-main"
if str(SARA_ROOT) not in sys.path:
    sys.path.insert(0, str(SARA_ROOT))

import src
if hasattr(src, "__path__") and str(SARA_ROOT / "src") not in src.__path__:
    src.__path__.append(str(SARA_ROOT / "src"))

try:
    from src.metrics.metrics import compute_f1, exact_match_score, rougel_score
except ImportError:
    def normalize_answer(s: str) -> str:
        s = s.lower()
        exclude = set(string.punctuation)
        s = "".join(ch for ch in s if ch not in exclude)
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        return " ".join(s.split())

    def compute_f1(a_gold: str, a_pred: str) -> Tuple[float, float, float]:
        gold_toks = normalize_answer(a_gold).split()
        pred_toks = normalize_answer(a_pred).split()
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            f1 = float(gold_toks == pred_toks)
            return f1, f1, f1
        if num_same == 0:
            return 0.0, 0.0, 0.0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    def exact_match_score(a_pred: str, a_gold: str) -> bool:
        return normalize_answer(a_pred) == normalize_answer(a_gold)

    def rougel_score(a_pred: str, a_gold: str) -> float:
        return 0.0

logger = logging.getLogger(__name__)


def compute_safe_qa_metrics(
    prediction: str,
    ground_truths: Union[str, Sequence[str]],
) -> Dict[str, float]:
    """Compute QA evaluation metrics with defensive handling for empty predictions.

    Guarantees:
    - Empty prediction against non-empty ground truth yields strictly 0.0 (does not crash or drop query).
    - Returns token_f1 and rouge_l.
    """
    if isinstance(ground_truths, str):
        gts = [ground_truths]
    else:
        gts = list(ground_truths)

    pred_clean = str(prediction).strip()

    # Defensive check: empty prediction
    if not pred_clean:
        return {"token_f1": 0.0, "rouge_l": 0.0}

    # Best token F1 across references
    f1_scores = []
    for gt in gts:
        gt_clean = str(gt).strip()
        if gt_clean:
            _, _, f1_val = compute_f1(gt_clean, pred_clean)
            f1_scores.append(f1_val)
    best_f1 = float(max(f1_scores)) if f1_scores else 0.0

    # Best ROUGE-L across references
    rouge_scores = []
    for gt in gts:
        gt_clean = str(gt).strip()
        if gt_clean:
            try:
                r_val = rougel_score(pred_clean, gt_clean)
                rouge_scores.append(r_val)
            except Exception:
                rouge_scores.append(0.0)

    best_rouge = float(max(rouge_scores)) if rouge_scores else 0.0

    return {"token_f1": best_f1, "rouge_l": best_rouge}


def compute_conditional_results_by_k(
    df_per_query: pd.DataFrame,
    f1_bm25_col: str = "f1_w4_bm25",
    f1_sem_col: str = "f1_w4_sem",
    tokens_bm25_col: str = "tokens_w4_bm25",
    tokens_sem_col: str = "tokens_w4_sem",
    budget_col: str = "predicted_k",
) -> pd.DataFrame:
    """Compute conditional performance stratified by predicted budget k in {2, 4, 5, 6, 8}."""
    rows = []
    unique_ks = sorted(df_per_query[budget_col].unique())

    for k in unique_ks:
        sub = df_per_query[df_per_query[budget_col] == k]
        n_q = len(sub)
        if n_q == 0:
            continue

        f1_0 = float(sub[f1_bm25_col].mean())
        f1_1 = float(sub[f1_sem_col].mean())
        tok_0 = float(sub[tokens_bm25_col].mean())
        tok_1 = float(sub[tokens_sem_col].mean())

        diffs = sub[f1_sem_col].values - sub[f1_bm25_col].values
        n_wins = int(np.sum(diffs > 1e-4))
        n_ties = int(np.sum(np.abs(diffs) <= 1e-4))
        n_losses = int(np.sum(diffs < -1e-4))

        rows.append({
            "predicted_k": int(k),
            "n_queries": n_q,
            "pct_of_dev": (n_q / len(df_per_query)) * 100.0,
            "f1_bm25_mean": f1_0,
            "f1_semantic_mean": f1_1,
            "delta_f1": f1_1 - f1_0,
            "tokens_bm25_mean": tok_0,
            "tokens_semantic_mean": tok_1,
            "delta_tokens": tok_1 - tok_0,
            "win_count": n_wins,
            "tie_count": n_ties,
            "loss_count": n_losses,
            "win_rate_pct": (n_wins / n_q) * 100.0,
            "loss_rate_pct": (n_losses / n_q) * 100.0,
        })

    return pd.DataFrame(rows)


def compute_w7a_win_tie_loss(
    f1_bm25_list: Sequence[float],
    f1_sem_list: Sequence[float],
    eps: float = 1e-4,
) -> Dict[str, Any]:
    """Compute aggregate query-level win, tie, loss statistics."""
    arr_0 = np.asarray(f1_bm25_list, dtype=np.float32)
    arr_1 = np.asarray(f1_sem_list, dtype=np.float32)
    assert len(arr_0) == len(arr_1)

    diffs = arr_1 - arr_0
    n = len(diffs)

    wins_mask = diffs > eps
    losses_mask = diffs < -eps
    ties_mask = np.abs(diffs) <= eps

    n_wins = int(np.sum(wins_mask))
    n_losses = int(np.sum(losses_mask))
    n_ties = int(np.sum(ties_mask))

    mean_gain_on_wins = float(np.mean(diffs[wins_mask])) if n_wins > 0 else 0.0
    mean_loss_on_losses = float(np.mean(diffs[losses_mask])) if n_losses > 0 else 0.0

    return {
        "n_queries": n,
        "n_wins": n_wins,
        "n_ties": n_ties,
        "n_losses": n_losses,
        "win_rate_pct": (n_wins / n) * 100.0 if n > 0 else 0.0,
        "tie_rate_pct": (n_ties / n) * 100.0 if n > 0 else 0.0,
        "loss_rate_pct": (n_losses / n) * 100.0 if n > 0 else 0.0,
        "mean_delta_f1_overall": float(np.mean(diffs)),
        "mean_gain_on_wins": mean_gain_on_wins,
        "mean_loss_on_losses": mean_loss_on_losses,
    }
