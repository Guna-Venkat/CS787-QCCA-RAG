"""Comprehensive Evaluation, Safety Metrics, and Stratification for Week 8 DAFiR.

Calculates:
1. Core QA metrics: Token F1, Exact Match, ROUGE-L.
2. Safety Metrics:
   - HELP precision: Fraction of promoted passages with Delta_i > 0.01.
   - HURT rate: Fraction of promoted passages with Delta_i < -0.01.
   - Oracle recall: Fraction where chosen candidate matches Oracle-1 chosen candidate.
   - Near-oracle recall: Fraction where chosen candidate Delta_i is within 0.01 of Oracle-1 max Delta.
   - Unnecessary promotion rate: Promoted when max_i(Delta_i) <= 0.01.
3. Promotion-Rank Distribution: Frequency of promotion across ranks 5..10 vs Oracle-1.
4. Stratified Query-Type Analysis: Extractive vs Abstractive vs Yes/No, query length, confidence.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_safety_and_controller_metrics(
    controller_decisions: Sequence[Dict[str, Any]],
    oracle_records_by_qid: Dict[str, Dict[str, Any]],
    epsilon_help: float = 0.01,
    epsilon_hurt: float = -0.01,
) -> Dict[str, float]:
    """Compute controller safety metrics comparing decisions against oracle promotion ground truth."""
    total_queries = len(controller_decisions)
    if total_queries == 0:
        return {}

    promoted_count = 0
    help_count = 0
    hurt_count = 0
    oracle_match_count = 0
    near_oracle_count = 0
    unnecessary_prom_count = 0

    for dec in controller_decisions:
        qid = str(dec["question_id"]).strip()
        action = dec.get("action", "NO_PROMOTION")
        promoted_idx = dec.get("chosen_candidate_idx")

        oracle_info = oracle_records_by_qid.get(qid, {})
        oracle_best_idx = oracle_info.get("best_candidate_idx")
        oracle_max_delta = oracle_info.get("max_delta", 0.0)
        candidate_deltas = oracle_info.get("candidate_deltas", {})

        if action == "PROMOTE" and promoted_idx is not None:
            promoted_count += 1
            actual_delta = candidate_deltas.get(promoted_idx, 0.0)

            if actual_delta > epsilon_help:
                help_count += 1
            elif actual_delta < epsilon_hurt:
                hurt_count += 1

            if promoted_idx == oracle_best_idx:
                oracle_match_count += 1

            if actual_delta >= (oracle_max_delta - epsilon_help):
                near_oracle_count += 1

            if oracle_max_delta <= epsilon_help:
                unnecessary_prom_count += 1

    help_precision = help_count / max(promoted_count, 1)
    hurt_rate = hurt_count / max(promoted_count, 1)
    oracle_recall = oracle_match_count / max(total_queries, 1)
    near_oracle_recall = near_oracle_count / max(total_queries, 1)
    unnecessary_rate = unnecessary_prom_count / max(promoted_count, 1)
    promotion_rate = promoted_count / max(total_queries, 1)

    return {
        "total_queries": float(total_queries),
        "promoted_queries": float(promoted_count),
        "promotion_rate": float(promotion_rate),
        "help_precision": float(help_precision),
        "hurt_rate": float(hurt_rate),
        "oracle_recall": float(oracle_recall),
        "near_oracle_recall": float(near_oracle_recall),
        "unnecessary_promotion_rate": float(unnecessary_rate),
    }


def analyze_promotion_rank_distribution(
    controller_promotions: Sequence[int],
    oracle_best_ranks: Sequence[int],
    candidate_ranks: Sequence[int] = (5, 6, 7, 8, 9, 10),
) -> pd.DataFrame:
    """Compare rank selection distribution of controller vs oracle."""
    c_counts = {r: 0 for r in candidate_ranks}
    o_counts = {r: 0 for r in candidate_ranks}

    for r in controller_promotions:
        if r in c_counts:
            c_counts[r] += 1

    for r in oracle_best_ranks:
        if r in o_counts:
            o_counts[r] += 1

    total_c = max(len(controller_promotions), 1)
    total_o = max(len(oracle_best_ranks), 1)

    rows = []
    for r in candidate_ranks:
        rows.append({
            "semantic_rank": r,
            "controller_count": c_counts[r],
            "controller_pct": c_counts[r] / total_c,
            "oracle_count": o_counts[r],
            "oracle_pct": o_counts[r] / total_o,
        })

    return pd.DataFrame(rows)


def stratify_by_query_type(
    results_df: pd.DataFrame,
    answer_types: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Breakdown evaluation metrics by QASPER answer type (extractive, abstractive, yes/no)."""
    df = results_df.copy()
    if answer_types is not None:
        df["answer_type"] = df["question_id"].astype(str).map(answer_types).fillna("unknown")
    elif "answer_type" not in df.columns:
        df["answer_type"] = "all"

    grouped = df.groupby("answer_type").agg(
        num_queries=("token_f1", "count"),
        mean_token_f1=("token_f1", "mean"),
        mean_exact_match=("exact_match", "mean"),
        mean_final_tokens=("final_context_tokens", "mean"),
        mean_cum_tokens=("cumulative_prompt_tokens", "mean"),
    ).reset_index()

    return grouped


def compute_evidence_utilization_metrics(
    promotion_records: Sequence[Dict[str, Any]],
    epsilon_help: float = 0.01,
    epsilon_hurt: float = -0.01,
) -> Dict[str, Any]:
    """Compute evidence potential and generator utilization bottleneck metrics.

    Evaluates:
    - P(AnswerChanged)
    - P(AnswerChanged | Delta > 0)
    - P(AnswerChanged | is_gold_evidence)
    - P(Delta > 0.01 | is_gold_evidence)
    - P(Delta == 0 | is_gold_evidence)
    - Promotion Utilization Failure Rate (UFR):
        UFR = P(y_i == y_0 | base_f1 < 1.0, is_gold_evidence == True)
    - 4-way taxonomy distribution:
        UTILIZED_HELPFUL, UNDERUTILIZED, TRUE_NEUTRAL, DISTRACTOR
    """
    total = len(promotion_records)
    if total == 0:
        return {}

    ans_changed_count = 0
    help_count = 0
    ans_changed_given_help = 0
    gold_ev_count = 0
    ans_changed_given_gold = 0
    help_given_gold = 0
    zero_delta_given_gold = 0

    ufr_denominator = 0  # base_f1 < 1.0 and is_gold_evidence == True
    ufr_numerator = 0    # unchanged prediction (y_i == y_0)

    tax_counts = {
        "UTILIZED_HELPFUL": 0,
        "UNDERUTILIZED": 0,
        "TRUE_NEUTRAL": 0,
        "DISTRACTOR": 0,
    }

    for r in promotion_records:
        delta_f1 = float(r.get("delta_f1", 0.0))
        base_f1 = float(r.get("base_f1", 0.0))
        is_gold = bool(r.get("is_gold_evidence", False))
        ans_changed = bool(r.get("answer_changed", False))

        # Check if answer changed
        if "answer_changed" not in r:
            base_p = str(r.get("base_prediction", "")).strip().lower()
            prom_p = str(r.get("promoted_prediction", "")).strip().lower()
            ans_changed = bool(base_p != prom_p)

        if ans_changed:
            ans_changed_count += 1

        is_helpful = delta_f1 > epsilon_help
        if is_helpful:
            help_count += 1
            if ans_changed:
                ans_changed_given_help += 1

        if is_gold:
            gold_ev_count += 1
            if ans_changed:
                ans_changed_given_gold += 1
            if is_helpful:
                help_given_gold += 1
            if abs(delta_f1) <= 1e-4:
                zero_delta_given_gold += 1

            # UFR condition: base answer was not already complete (base_f1 < 1.0)
            if base_f1 < 0.99:
                ufr_denominator += 1
                if not ans_changed:
                    ufr_numerator += 1

        # 4-way taxonomy assignment
        t_type = r.get("evidence_utilization_type")
        if not t_type:
            if delta_f1 < epsilon_hurt:
                t_type = "DISTRACTOR"
            elif is_gold and is_helpful:
                t_type = "UTILIZED_HELPFUL"
            elif is_gold and not is_helpful:
                t_type = "UNDERUTILIZED"
            else:
                t_type = "TRUE_NEUTRAL"

        if t_type in tax_counts:
            tax_counts[t_type] += 1

    p_ans_changed = ans_changed_count / max(total, 1)
    p_ans_changed_given_help = ans_changed_given_help / max(help_count, 1)
    p_ans_changed_given_gold = ans_changed_given_gold / max(gold_ev_count, 1)
    p_help_given_gold = help_given_gold / max(gold_ev_count, 1)
    p_zero_given_gold = zero_delta_given_gold / max(gold_ev_count, 1)
    ufr = ufr_numerator / max(ufr_denominator, 1)

    return {
        "total_promotions": total,
        "prob_answer_changed": float(p_ans_changed),
        "prob_answer_changed_given_help": float(p_ans_changed_given_help),
        "prob_answer_changed_given_gold_evidence": float(p_ans_changed_given_gold),
        "prob_help_given_gold_evidence": float(p_help_given_gold),
        "prob_zero_delta_given_gold_evidence": float(p_zero_given_gold),
        "promotion_utilization_failure_rate": float(ufr),
        "taxonomy_distribution": {
            k: {"count": v, "pct": float(v / max(total, 1))}
            for k, v in tax_counts.items()
        },
    }


def compute_query_grouped_selector_metrics(
    df: pd.DataFrame,
    selector_scores_col: Optional[str] = None,
    selector_obj: Optional[Any] = None,
    promotion_threshold: float = -999.0,  # Default to pure ranking among 6 candidates
    base_k4_f1: float = 0.38884,
    oracle_selective1_f1: float = 0.4808,
) -> Dict[str, Any]:
    """Compute query-grouped passage selection metrics and downstream headroom recovery.

    Metrics:
    - Top-1 Oracle Accuracy: Fraction where argmax(scores) == argmax(delta_f1)
    - Near-Oracle Selection Rate: Fraction where Delta* - Delta_chosen <= 0.01
    - Mean and Median Selector Regret: Delta* - Delta_chosen
    - HELP Precision: Fraction of promoted passages where Delta > 0.01
    - HURT Rate: Fraction of promoted passages where Delta < -0.01
    - Oracle-Gated Downstream F1: Base F1 + Delta of chosen candidate on escalation queries (Delta* > 0.01)
    - Selector Headroom Recovery (SHR): (F1_sim - F1_k4) / (F1_oracle_sel - F1_k4)
    """
    df_eval = df.copy()
    if "question_id" not in df_eval.columns:
        raise ValueError("DataFrame must contain 'question_id' column.")

    top1_matches = 0
    near_oracle_matches = 0
    regrets = []
    promoted_deltas = []
    oracle_gated_deltas = []
    num_queries = 0

    for qid, group in df_eval.groupby("question_id"):
        num_queries += 1
        group = group.reset_index(drop=True)
        deltas = group["delta_f1"].values
        best_delta = float(np.max(deltas))
        oracle_best_idx = int(np.argmax(deltas))

        # Get selector scores
        if selector_obj is not None:
            scores = selector_obj.predict_scores(group)
        elif selector_scores_col is not None and selector_scores_col in group.columns:
            scores = group[selector_scores_col].values
        else:
            raise ValueError("Either selector_obj or valid selector_scores_col must be provided.")

        chosen_idx = int(np.argmax(scores))
        chosen_score = float(scores[chosen_idx])

        # Action: check if STOP
        is_promote = chosen_score >= promotion_threshold
        chosen_delta = float(deltas[chosen_idx]) if is_promote else 0.0
        regret = max(0.0, best_delta - chosen_delta)

        # 1. Top-1 Oracle passage match
        if chosen_idx == oracle_best_idx:
            top1_matches += 1

        # 2. Near-oracle passage match (within 0.01 F1 of best)
        if best_delta - chosen_delta <= 0.01:
            near_oracle_matches += 1

        regrets.append(regret)
        if is_promote:
            promoted_deltas.append(chosen_delta)

        # 3. Clean selector-only simulation under oracle gate:
        # If query needed escalation (Delta* > 0.01), give selector all 6 candidates.
        # If safe-stop query (Delta* <= 0.01), keep k4 (gain = 0.0).
        if best_delta > 0.01:
            oracle_gated_deltas.append(chosen_delta)
        else:
            oracle_gated_deltas.append(0.0)

    top1_acc = top1_matches / max(num_queries, 1)
    near_oracle_rate = near_oracle_matches / max(num_queries, 1)
    mean_regret = float(np.mean(regrets)) if regrets else 0.0
    median_regret = float(np.median(regrets)) if regrets else 0.0

    n_promoted = len(promoted_deltas)
    help_count = sum(1 for d in promoted_deltas if d > 0.01)
    hurt_count = sum(1 for d in promoted_deltas if d < -0.01)
    help_precision = help_count / max(n_promoted, 1)
    hurt_rate = hurt_count / max(n_promoted, 1)

    # Downstream F1 with oracle gate
    mean_gated_gain = float(np.mean(oracle_gated_deltas)) if oracle_gated_deltas else 0.0
    final_downstream_f1 = base_k4_f1 + mean_gated_gain

    # Selector Headroom Recovery (SHR)
    oracle_headroom = max(oracle_selective1_f1 - base_k4_f1, 1e-6)
    achieved_gain = final_downstream_f1 - base_k4_f1
    shr = achieved_gain / oracle_headroom

    return {
        "num_queries": num_queries,
        "top1_oracle_accuracy": float(top1_acc),
        "near_oracle_selection_rate": float(near_oracle_rate),
        "mean_regret": float(mean_regret),
        "median_regret": float(median_regret),
        "help_precision": float(help_precision),
        "hurt_rate": float(hurt_rate),
        "final_downstream_f1": float(final_downstream_f1),
        "selector_headroom_recovery": float(shr),
    }


def compute_calibration_metrics(
    y_true_binary: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute Expected Calibration Error (ECE), Brier Score, and Decile Calibration Table."""
    brier = float(np.mean((y_prob - y_true_binary) ** 2))

    # Equal-width bins
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    decile_rows = []

    for b in range(n_bins):
        low, high = bin_limits[b], bin_limits[b + 1]
        mask = (y_prob >= low) & (y_prob < high if b < n_bins - 1 else y_prob <= high)
        n_in_bin = int(mask.sum())
        if n_in_bin > 0:
            mean_prob = float(np.mean(y_prob[mask]))
            mean_obs = float(np.mean(y_true_binary[mask]))
            bin_err = abs(mean_prob - mean_obs)
            ece += (n_in_bin / len(y_prob)) * bin_err
            decile_rows.append({
                "decile": b + 1,
                "bin_range": f"[{low:.1f}, {high:.1f}]",
                "count": n_in_bin,
                "mean_pred_prob": mean_prob,
                "observed_rate": mean_obs,
                "calibration_error": bin_err,
            })

    return {
        "ece": float(ece),
        "brier_score": float(brier),
        "decile_table": pd.DataFrame(decile_rows),
    }


