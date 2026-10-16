"""W8-E: Draft-Aware Selective Fidelity Controller (DAFiR-1 & DAFiR-2).

Controller Strategy:
1. Stage 1 (Sufficiency Gate):
     Evaluate draft uncertainty features with B0/B1/B2.
     If safe_stop (e.g. high confidence / low need-for-promotion), return draft immediately.
2. Stage 2 (Candidate Selection):
     For each candidate i in {5..10}, evaluate U4: P(HELP_i), P(HURT_i), pred_Delta_i.
     Score: S_i = P(HELP_i) - lambda_h * P(HURT_i) + lambda_d * pred_Delta_i
     Best candidate: i* = argmax_i S_i
     Promote only if: P(HELP_{i*}) > tau_h AND P(HURT_{i*}) < tau_d.
     Otherwise NO_PROMOTION.
3. Stage 3 (Two-Stage Promotion for DAFiR-2):
     After first promotion regeneration y_1, re-evaluate uncertainty and remaining candidates.
     Allow at most one second promotion if still insufficient.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def compute_candidate_utility_score(
    p_help: float,
    p_hurt: float,
    pred_delta: float,
    lambda_h: float = 1.0,
    lambda_d: float = 0.5,
) -> float:
    """Compute composite candidate utility score S_i."""
    return float(p_help - lambda_h * p_hurt + lambda_d * pred_delta)


def select_best_promotion_candidate(
    candidate_predictions: Union[Dict[int, Dict[str, float]], List[Dict[str, Any]]],
    lambda_h: float = 1.0,
    lambda_d: float = 0.5,
    tau_h: float = 0.4,
    tau_d: float = 0.35,
) -> Tuple[Optional[int], float, Dict[str, Any]]:
    """Select candidate with highest utility score satisfying safety constraints."""
    best_score = -float("inf")
    best_idx: Optional[int] = None
    scores_by_idx: Dict[int, float] = {}

    # Normalize to dict: idx -> dict
    cand_dict: Dict[int, Dict[str, float]] = {}
    if isinstance(candidate_predictions, list):
        for item in candidate_predictions:
            idx = item.get("candidate_idx", item.get("candidate_passage_index", 0))
            cand_dict[int(idx)] = {
                "p_help": float(item.get("prob_help", item.get("p_help", 0.0))),
                "p_hurt": float(item.get("prob_hurt", item.get("p_hurt", 0.0))),
                "pred_delta": float(item.get("pred_delta_f1", item.get("pred_delta", 0.0))),
            }
    else:
        for idx, item in candidate_predictions.items():
            cand_dict[int(idx)] = {
                "p_help": float(item.get("prob_help", item.get("p_help", 0.0))),
                "p_hurt": float(item.get("prob_hurt", item.get("p_hurt", 0.0))),
                "pred_delta": float(item.get("pred_delta_f1", item.get("pred_delta", 0.0))),
            }

    for idx, preds in cand_dict.items():
        p_help = preds["p_help"]
        p_hurt = preds["p_hurt"]
        p_delta = preds["pred_delta"]

        s_i = compute_candidate_utility_score(
            p_help=p_help,
            p_hurt=p_hurt,
            pred_delta=p_delta,
            lambda_h=lambda_h,
            lambda_d=lambda_d,
        )
        scores_by_idx[idx] = s_i

        if s_i > best_score:
            best_score = s_i
            best_idx = idx

    # Enforce threshold gate
    promoted_idx: Optional[int] = None
    if best_idx is not None:
        best_preds = cand_dict[best_idx]
        if best_preds["p_help"] > tau_h and best_preds["p_hurt"] < tau_d:
            promoted_idx = best_idx

    meta = {
        "candidate_scores": scores_by_idx,
        "selected_candidate": promoted_idx,
        "top_candidate": best_idx,
        "top_score": best_score,
        "threshold_passed": promoted_idx is not None,
    }
    return promoted_idx, best_score, meta


class DAFiRController:
    """DAFiR Controller managing safe-stopping, single-promotion, and two-stage promotion."""

    def __init__(
        self,
        stop_threshold: float = 0.45,
        lambda_h: float = 1.0,
        lambda_d: float = 0.5,
        tau_h: float = 0.4,
        tau_d: float = 0.35,
        max_promotions: int = 2,
        tau_help: Optional[float] = None,
        tau_hurt: Optional[float] = None,
    ):
        self.stop_threshold = stop_threshold
        self.lambda_h = lambda_h
        self.lambda_d = lambda_d
        self.tau_h = tau_help if tau_help is not None else tau_h
        self.tau_d = tau_hurt if tau_hurt is not None else tau_d
        self.max_promotions = max_promotions

    def should_escalate_draft(self, draft_features: Dict[str, Any]) -> bool:
        """Decide whether draft needs promotion or can safely stop based on uncertainty."""
        entropy = float(draft_features.get("mean_entropy", draft_features.get("entropy", 0.0)))
        margin = float(draft_features.get("mean_margin", draft_features.get("margin", 1.0)))

        # Confident draft: low entropy, high margin -> safe stop
        if entropy < 0.3 and margin > 2.0:
            return False

        p_needs = float(draft_features.get("p_needs_promotion", 0.5))
        return p_needs >= self.stop_threshold or entropy >= 0.3

    def evaluate_single_promotion(
        self,
        draft_features: Dict[str, Any],
        candidate_predictions: Union[Dict[int, Dict[str, float]], List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Evaluate Pass 1 promotion decision."""
        if not self.should_escalate_draft(draft_features):
            return {
                "action": "NO_PROMOTION",
                "chosen_candidate_idx": None,
                "reason": "safe_stop_confident_draft",
            }

        chosen_idx, best_score, meta = select_best_promotion_candidate(
            candidate_predictions=candidate_predictions,
            lambda_h=self.lambda_h,
            lambda_d=self.lambda_d,
            tau_h=self.tau_h,
            tau_d=self.tau_d,
        )

        if chosen_idx is not None:
            return {
                "action": "PROMOTE",
                "chosen_candidate_idx": chosen_idx,
                "score": best_score,
                "meta": meta,
            }
        else:
            return {
                "action": "NO_PROMOTION",
                "chosen_candidate_idx": None,
                "score": best_score,
                "meta": meta,
                "reason": "safety_threshold_not_met",
            }

    def evaluate_second_promotion(
        self,
        pass1_decision: Dict[str, Any],
        y1_features: Dict[str, Any],
        candidate_predictions_pass2: Union[Dict[int, Dict[str, float]], List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Evaluate optional Pass 2 promotion decision (DAFiR-2)."""
        if self.max_promotions < 2:
            return {"action": "NO_PROMOTION", "chosen_second_candidate_idx": None}

        if pass1_decision.get("action") != "PROMOTE":
            return {"action": "NO_PROMOTION", "chosen_second_candidate_idx": None}

        if not self.should_escalate_draft(y1_features):
            return {
                "action": "NO_PROMOTION",
                "chosen_second_candidate_idx": None,
                "reason": "safe_stop_after_pass1",
            }

        chosen_idx, best_score, meta = select_best_promotion_candidate(
            candidate_predictions=candidate_predictions_pass2,
            lambda_h=self.lambda_h,
            lambda_d=self.lambda_d,
            tau_h=self.tau_h,
            tau_d=self.tau_d,
        )

        if chosen_idx is not None:
            return {
                "action": "PROMOTE_SECOND",
                "chosen_second_candidate_idx": chosen_idx,
                "score": best_score,
                "meta": meta,
            }
        else:
            return {
                "action": "NO_PROMOTION",
                "chosen_second_candidate_idx": None,
                "reason": "safety_threshold_not_met",
            }
