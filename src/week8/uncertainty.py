"""W8-B: Draft Uncertainty Features (U1..U12) and Sufficiency Classification (B0, B1, B2).

Features:
- U1: Mean entropy
- U2: Maximum entropy
- U3: Mean top1-top2 logit margin
- U4: Minimum top1-top2 margin
- U5: Mean answer-token log probability
- U6: Normalized sequence log probability
- U7: Answer token length
- U8: EOS confidence
- U9: Empty-answer indicator
- U10: Repetition ratio
- U11: Lexical overlap draft <-> query
- U12: Semantic similarity draft <-> query

Models:
- B0: Logistic Regression
- B1: GBDT
- B2: Small MLP
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    auc,
)
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


def compute_uncertainty_features_for_row(
    draft_row: Dict[str, Any],
    semantic_scorer: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, float]:
    """Compute U1..U12 features from a single draft diagnostic record.
    
    Guarantees:
    - All non-empty generated answers have answer_len (U7) > 0.
    - U5 is length-normalized mean token log-probability (1/T sum log p_t).
    - U6 is cumulative sequence log-probability (sum_t log p_t).
    - U10 repetition ratio accounts for short QA answers.
    """
    entropies = draft_row.get("token_entropies", [])
    margins = draft_row.get("top1_top2_margins", [])
    logprobs = draft_row.get("token_logprobs", [])
    token_ids = draft_row.get("token_ids", [])
    draft_text = str(draft_row.get("draft_answer", "")).strip()
    query_text = str(draft_row.get("question", "")).strip()
    eos_conf = float(draft_row.get("eos_confidence", 0.0))

    # Robust token ID / answer length resolution:
    # 1. token_ids if non-empty
    # 2. Tokenize draft_text if tokenizer provided
    # 3. Exactly 1 step per generated token in token_entropies
    # 4. Word count of draft_text
    resolved_ids: List[int] = []
    if token_ids and len(token_ids) > 0:
        resolved_ids = list(token_ids)
    elif tokenizer is not None and len(draft_text) > 0:
        resolved_ids = tokenizer.encode(draft_text, add_special_tokens=False)
    elif entropies and len(entropies) > 0:
        # Each step in greedy decoding corresponds to exactly 1 generated token
        resolved_ids = list(range(len(entropies)))
    elif len(draft_text) > 0:
        resolved_ids = list(range(len(draft_text.split())))

    # Robust logprobs resolution:
    # If raw logprobs are missing from older cache, reconstruct from top1-top2 margin m:
    # log p(y_t) ~= -log(1 + exp(-m_t))
    resolved_logprobs = list(logprobs) if logprobs else []
    if not resolved_logprobs and margins:
        resolved_logprobs = [-float(np.log1p(np.exp(-min(float(m), 30.0)))) for m in margins]

    # U1: Mean entropy
    u1 = float(np.mean(entropies)) if entropies else 0.0
    # U2: Max entropy
    u2 = float(np.max(entropies)) if entropies else 0.0
    # U3: Mean top1-top2 margin
    u3 = float(np.mean(margins)) if margins else 0.0
    # U4: Min top1-top2 margin
    u4 = float(np.min(margins)) if margins else 0.0
    # U5: Mean answer-token log probability (length-normalized: 1/T sum log p_t)
    u5 = float(np.mean(resolved_logprobs)) if resolved_logprobs else 0.0
    # U6: Cumulative sequence log probability (sum_t log p_t)
    u6 = float(np.sum(resolved_logprobs)) if resolved_logprobs else (-100.0 if len(draft_text) == 0 else 0.0)
    # U7: Answer length in tokens (> 0 for any non-empty answer)
    u7 = float(len(resolved_ids)) if resolved_ids else (float(len(entropies)) if entropies else (0.0 if len(draft_text) == 0 else 1.0))
    # U8: EOS confidence
    u8 = eos_conf
    # U9: Empty answer indicator (1 if empty, 0 otherwise)
    u9 = 1.0 if len(draft_text) == 0 else 0.0
    # U10: Repetition ratio (1 - unique / total)
    rep_tokens = []
    if token_ids and len(token_ids) > 0:
        rep_tokens = list(token_ids)
    elif tokenizer is not None and len(draft_text) > 0:
        rep_tokens = tokenizer.encode(draft_text, add_special_tokens=False)
    elif len(draft_text) > 0:
        import re
        rep_tokens = re.findall(r'\w+', draft_text.lower())

    if len(rep_tokens) > 1:
        u10 = float(1.0 - (len(set(rep_tokens)) / float(len(rep_tokens))))
    else:
        u10 = 0.0

    # U11: Lexical overlap draft <-> query (token Jaccard)
    draft_toks = set(draft_text.lower().split())
    query_toks = set(query_text.lower().split())
    if draft_toks and query_toks:
        inter = len(draft_toks & query_toks)
        union = len(draft_toks | query_toks)
        u11 = float(inter / union) if union > 0 else 0.0
    else:
        u11 = 0.0

    # U12: Semantic similarity draft <-> query
    u12 = 0.0
    if semantic_scorer is not None and draft_text and query_text:
        try:
            scores = semantic_scorer.score_query_passages(query_text, [draft_text]).detach().cpu().numpy()
            u12 = float(scores[0])
        except Exception:
            u12 = 0.0

    return {
        "u1_entropy_mean": u1,
        "u2_entropy_max": u2,
        "u3_margin_mean": u3,
        "u4_margin_min": u4,
        "u5_logprob_mean": u5,
        "u6_seq_logprob": u6,
        "u6_norm_seq_logprob": u6,  # compatibility alias
        "u7_answer_len": u7,
        "u8_eos_conf": u8,
        "u9_empty_answer": u9,
        "u10_repetition_ratio": u10,
        "u11_lexical_overlap": u11,
        "u12_semantic_sim": u12,
    }


def compute_expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        in_bin = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(labels[in_bin])
            bin_conf = np.mean(probs[in_bin])
            ece += (bin_size / float(n)) * abs(bin_acc - bin_conf)
    return float(ece)


class SmallMLPClassifier(nn.Module):
    """Small 2-layer MLP for sufficiency prediction (B2)."""

    def __init__(self, in_features: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


from sklearn.model_selection import train_test_split


def evaluate_sufficiency_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate B0 (Logistic), B1 (GBDT), and B2 (MLP).
    
    Can be called either as:
      evaluate_sufficiency_models(X_train, y_train, X_val, y_val)
    or:
      evaluate_sufficiency_models(X, y)  # splits 80/20 internally
    """
    if X_val is None or y_val is None:
        stratify = None
        unique, counts = np.unique(y_train, return_counts=True)
        if len(unique) > 1 and np.min(counts) >= 2:
            stratify = y_train
        X_tr, X_v, y_tr, y_v = train_test_split(
            X_train, y_train, test_size=test_size, random_state=random_state, stratify=stratify
        )
    else:
        X_tr, y_tr, X_v, y_v = X_train, y_train, X_val, y_val

    results: Dict[str, Dict[str, float]] = {}

    # B0: Logistic Regression
    clf_b0 = LogisticRegression(max_iter=500, random_state=42)
    clf_b0.fit(X_tr, y_tr)
    if hasattr(clf_b0, "predict_proba") and len(clf_b0.classes_) > 1:
        probs_b0 = clf_b0.predict_proba(X_v)[:, 1]
    else:
        probs_b0 = np.full(len(y_v), 0.5)

    # B1: GBDT
    clf_b1 = HistGradientBoostingClassifier(random_state=42, max_iter=100)
    clf_b1.fit(X_tr, y_tr)
    if hasattr(clf_b1, "predict_proba") and len(clf_b1.classes_) > 1:
        probs_b1 = clf_b1.predict_proba(X_v)[:, 1]
    else:
        probs_b1 = np.full(len(y_v), 0.5)

    # B2: Small MLP
    device = "cpu"
    in_dim = X_tr.shape[1]
    mlp = SmallMLPClassifier(in_dim).to(device)
    optimizer = optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    xt_tensor = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt_tensor = torch.tensor(y_tr, dtype=torch.float32, device=device)
    mlp.train()
    for _ in range(50):
        optimizer.zero_grad()
        out = mlp(xt_tensor)
        loss = criterion(out, yt_tensor)
        loss.backward()
        optimizer.step()

    mlp.eval()
    xv_tensor = torch.tensor(X_v, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits_b2 = mlp(xv_tensor)
        probs_b2 = torch.sigmoid(logits_b2).detach().cpu().numpy()

    for name, p in [("B0_Logistic", probs_b0), ("B1_GBDT", probs_b1), ("B2_MLP", probs_b2)]:
        preds = (p >= 0.5).astype(int)
        acc = float(np.mean(preds == y_v))
        try:
            roc_auc = float(roc_auc_score(y_v, p))
        except Exception:
            roc_auc = 0.5
        try:
            prec, rec, _ = precision_recall_curve(y_v, p)
            pr_auc = float(auc(rec, prec))
        except Exception:
            pr_auc = 0.5
        brier = float(brier_score_loss(y_v, p))
        ece = compute_expected_calibration_error(p, y_v)

        results[name] = {
            "accuracy": round(acc, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
        }

    return results


# Aliases
UncertaintyClassifier = SmallMLPClassifier
train_uncertainty_sufficiency_models = evaluate_sufficiency_models

