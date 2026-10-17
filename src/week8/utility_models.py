"""W8-C & W8-D: Passage Utility Models (U0, U1, U2, U3, U4) and Baseline Selectors (S0..S4, Oracle).

Architectures:
- S0..S4: Heuristic baseline selectors (Rank-5, BM25, Semantic, Draft-Sim, MIG).
- Oracle: Upper-bound selector picking argmax Delta_i.
- U0: Tabular baseline (GBDT + Ridge/Logistic) with paper-grouped OOF training.
- U1: Pretrained Cross-Encoder on [q, D_i] with multitask + ranking heads.
- U2-F: Draft-Conditioned Cross-Encoder on [q, y_0, D_i] with frozen MiniLM backbone.
- U2-A: Draft-Conditioned Cross-Encoder on [q, y_0, D_i] with adapted backbone (top layers/LoRA).
- U3: Listwise Draft-Aware Selector observing all 6 candidates jointly.
- U4-A: Hybrid Fusion of U2 representation + Tabular / MIG features.
- U4-B: U4-A + Generator Attribution / Interaction features (A1..A6).

Ranking Formulation:
- Candidates i in {5, 6, 7, 8, 9, 10} for query q.
- Action space: A = {STOP, 5, 6, 7, 8, 9, 10}.
- Multitask loss with pairwise/listwise ranking objective and masked Delta LL.
- Strict token budgeting with candidate evidence prioritized over draft.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.week8.utility_dataset import (
    assert_no_feature_leakage,
    get_paper_grouped_splits,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token Budgeting and Truncation Policy
# ---------------------------------------------------------------------------
def truncate_and_budget_tokens(
    tokenizer: Any,
    query: str,
    draft: str,
    candidate: str,
    max_length: int = 512,
    max_query_tokens: int = 96,
    max_draft_tokens: int = 96,
) -> Tuple[str, Dict[str, Any]]:
    """Strict token budgeting for [q, y_0, D_i].

    Policy:
    - Query allocated up to max_query_tokens (default 96).
    - Draft allocated up to max_draft_tokens (default 96).
    - Candidate evidence gets full remaining budget (at least ~300 tokens).
    - Candidate evidence is strictly prioritized over draft if truncation needed.
    """
    q_tokens = tokenizer.encode(str(query), add_special_tokens=False)
    d_tokens = tokenizer.encode(str(draft), add_special_tokens=False) if draft else []
    c_tokens = tokenizer.encode(str(candidate), add_special_tokens=False)

    overhead = len(tokenizer.encode("[QUESTION] [DRAFT ANSWER] [CANDIDATE EVIDENCE]",
                                    add_special_tokens=False)) + 4
    available = max_length - overhead
    if available < 32:
        raise ValueError("Token budget is too small")
    original_lengths = (len(q_tokens), len(d_tokens), len(c_tokens))
    q_tokens = q_tokens[:min(max_query_tokens, available // 3)]
    remaining = available - len(q_tokens)
    draft_budget = min(len(d_tokens), max_draft_tokens,
                       max(0, remaining - min(len(c_tokens), remaining - min(16, len(d_tokens)))))
    d_tokens = d_tokens[:draft_budget]
    c_tokens = c_tokens[:remaining - len(d_tokens)]
    was_truncated = original_lengths != (len(q_tokens), len(d_tokens), len(c_tokens))

    q_text = tokenizer.decode(q_tokens, skip_special_tokens=True)
    d_text = tokenizer.decode(d_tokens, skip_special_tokens=True) if d_tokens else ""
    c_text = tokenizer.decode(c_tokens, skip_special_tokens=True)

    formatted = f"[QUESTION] {q_text} [DRAFT ANSWER] {d_text} [CANDIDATE EVIDENCE] {c_text}"

    diagnostics = {
        "query_tokens": len(q_tokens),
        "draft_tokens": len(d_tokens),
        "candidate_tokens": len(c_tokens),
        "total_tokens": len(q_tokens) + len(d_tokens) + len(c_tokens) + overhead,
        "was_truncated": was_truncated,
    }
    return formatted, diagnostics


def format_u1_text(query: str, candidate: str) -> str:
    """Format input for U1 (Pre-generation query/passage cross-encoder)."""
    return f"[QUESTION] {query.strip()} [CANDIDATE] {candidate.strip()}"


def format_u2_text(query: str, draft: str, candidate: str) -> str:
    """Format input for U2 (Draft-conditioned cross-encoder)."""
    d_str = draft.strip() if draft else "None"
    return f"[QUESTION] {query.strip()} [DRAFT ANSWER] {d_str} [CANDIDATE EVIDENCE] {candidate.strip()}"


# ---------------------------------------------------------------------------
# Multitask Heads Module
# ---------------------------------------------------------------------------
class MultitaskUtilityHeads(nn.Module):
    """Multitask heads: HELP (binary), HURT (binary), Delta F1 (regression),
    Oracle-Relative Regret (regression), Delta LL (masked), 3-way classification.
    """

    def __init__(
        self,
        in_features: Optional[int] = None,
        hidden_dim: int = 64,
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        in_dim = in_features if in_features is not None else (input_dim if input_dim is not None else 64)
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head_help = nn.Linear(hidden_dim, 1)        # Binary logit: P(HELP)
        self.head_hurt = nn.Linear(hidden_dim, 1)        # Binary logit: P(HURT)
        self.head_delta = nn.Linear(hidden_dim, 1)       # Continuous Delta F1
        self.head_regret = nn.Linear(hidden_dim, 1)      # Oracle-relative regret (Delta^* - Delta_i)
        self.head_delta_ll = nn.Linear(hidden_dim, 1)    # Dense continuous gold log-likelihood change
        self.head_beneficial_change = nn.Linear(hidden_dim, 1)
        self.head_3way = nn.Linear(hidden_dim, 3)        # 3 classes: 0=NEU, 1=HELP, 2=HURT

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.shared(x)
        logit_help = self.head_help(feat).squeeze(-1)
        logit_hurt = self.head_hurt(feat).squeeze(-1)
        pred_delta = self.head_delta(feat).squeeze(-1)
        pred_regret = F.relu(self.head_regret(feat).squeeze(-1))  # Regret is non-negative
        pred_delta_ll = self.head_delta_ll(feat).squeeze(-1)
        logit_beneficial = self.head_beneficial_change(feat).squeeze(-1)
        logits_3way = self.head_3way(feat)

        prob_help = torch.sigmoid(logit_help)
        prob_hurt = torch.sigmoid(logit_hurt)
        utility_score = prob_help - prob_hurt  # Composite ranking score

        return {
            "logit_help": logit_help,
            "logit_hurt": logit_hurt,
            "prob_help": prob_help,
            "prob_hurt": prob_hurt,
            "pred_delta": pred_delta,
            "pred_delta_f1": pred_delta,
            "pred_regret": pred_regret,
            "pred_delta_ll": pred_delta_ll,
            "logit_beneficial": logit_beneficial,
            "prob_beneficial": torch.sigmoid(logit_beneficial),
            "logits_3way": logits_3way,
            "label_logits": logits_3way,
            "utility_score": utility_score,
        }


# ---------------------------------------------------------------------------
# Ranking Losses
# ---------------------------------------------------------------------------
class PairwiseRankingLoss(nn.Module):
    """Pairwise ranking loss enforcing s_i > s_j whenever Delta_i > Delta_j + margin."""

    def __init__(self, margin: float = 0.01):
        super().__init__()
        self.margin = margin

    def forward(self, scores: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Args: scores (N,), deltas (N,) for candidates within the same query."""
        if len(scores) < 2:
            return torch.tensor(0.0, device=scores.device)
        # Pairwise differences
        diff_delta = deltas.unsqueeze(1) - deltas.unsqueeze(0)  # (N, N)
        diff_score = scores.unsqueeze(1) - scores.unsqueeze(0)  # (N, N)

        # Valid pairs where candidate i strictly outperforms candidate j
        valid_pairs = diff_delta > self.margin
        if not valid_pairs.any():
            return torch.tensor(0.0, device=scores.device)

        # Logistic loss: log(1 + exp(-(s_i - s_j)))
        loss = F.softplus(-diff_score[valid_pairs]).mean()
        return loss


class ListNetRankingLoss(nn.Module):
    """ListNet listwise ranking loss comparing softmax distributions."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, scores: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        if len(scores) < 2:
            return torch.tensor(0.0, device=scores.device)
        p_true = F.softmax(deltas / self.temperature, dim=-1)
        p_pred = F.log_softmax(scores / self.temperature, dim=-1)
        return -torch.sum(p_true * p_pred)


class MultitaskUtilityLoss(nn.Module):
    """Combined multitask loss with ranking objective and masked Delta LL."""

    def __init__(
        self,
        lambda_h: float = 1.0,
        lambda_hu: float = 1.0,
        lambda_delta: float = 1.0,
        lambda_regret: float = 0.5,
        lambda_ll: float = 0.25,
        lambda_3way: float = 0.5,
        lambda_rank: float = 1.0,
    ):
        super().__init__()
        self.lambda_h = lambda_h
        self.lambda_hu = lambda_hu
        self.lambda_delta = lambda_delta
        self.lambda_regret = lambda_regret
        self.lambda_ll = lambda_ll
        self.lambda_3way = lambda_3way
        self.lambda_rank = lambda_rank

        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.ranking_loss = PairwiseRankingLoss(margin=0.01)

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        l_h = self.bce(preds["logit_help"], targets["target_help_binary"])
        l_hu = self.bce(preds["logit_hurt"], targets["target_hurt_binary"])
        l_delta = self.mse(preds["pred_delta"], targets["target_continuous_delta"])

        # Regret loss
        l_regret = self.mse(preds["pred_regret"], targets["target_regret"]) if "target_regret" in targets else torch.tensor(0.0, device=l_h.device)

        # Masked Delta LL loss
        if "target_mask_ll" in targets and targets["target_mask_ll"].sum() > 0:
            mask = targets["target_mask_ll"] > 0.5
            l_ll = self.mse(preds["pred_delta_ll"][mask], targets["target_dense_gold_ll"][mask])
        else:
            l_ll = torch.tensor(0.0, device=l_h.device)

        l_3way = self.ce(preds["logits_3way"], targets["target_3way_label"])

        # Ranking loss across query candidates
        l_rank = self.ranking_loss(preds["utility_score"], targets["target_continuous_delta"])

        total = (
            self.lambda_h * l_h
            + self.lambda_hu * l_hu
            + self.lambda_delta * l_delta
            + self.lambda_regret * l_regret
            + self.lambda_ll * l_ll
            + self.lambda_3way * l_3way
            + self.lambda_rank * l_rank
        )

        return {
            "loss_total": total,
            "loss_help": l_h,
            "loss_hurt": l_hu,
            "loss_delta": l_delta,
            "loss_regret": l_regret,
            "loss_ll": l_ll,
            "loss_3way": l_3way,
            "loss_rank": l_rank,
        }


# ---------------------------------------------------------------------------
# Base Passage Selector
# ---------------------------------------------------------------------------
class BasePassageSelector:
    """Abstract base class for all Week-8 passage selectors.

    Action space: A = {STOP, 5, 6, 7, 8, 9, 10}
    """

    def predict_scores(self, df_query: pd.DataFrame) -> np.ndarray:
        """Return score array of length len(df_query). Higher score = higher utility."""
        raise NotImplementedError

    def select_passage(
        self,
        df_query: pd.DataFrame,
        promotion_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Select single passage or STOP if maximum predicted utility is below threshold."""
        scores = self.predict_scores(df_query)
        if len(scores) == 0:
            return {"action": "STOP", "chosen_idx": None, "chosen_rank": None, "score": -999.0, "scores": scores}

        best_i = int(np.argmax(scores))
        best_score = float(scores[best_i])

        rank_col = "semantic_rank"
        chosen_rank = int(df_query.iloc[best_i][rank_col]) if rank_col in df_query.columns else (5 + best_i)

        if best_score < promotion_threshold:
            action = "STOP"
            chosen_idx = None
            chosen_rank = None
        else:
            action = "PROMOTE"
            chosen_idx = best_i

        return {
            "action": action,
            "chosen_idx": chosen_idx,
            "chosen_rank": chosen_rank,
            "score": best_score,
            "scores": scores,
        }


# ---------------------------------------------------------------------------
# Simple Heuristic Selectors (S0..S4, Oracle)
# ---------------------------------------------------------------------------
class S0_Rank5Selector(BasePassageSelector):
    """S0: Always promotes semantic rank 5."""

    def predict_scores(self, df_query: pd.DataFrame) -> np.ndarray:
        scores = np.zeros(len(df_query))
        rank_col = "semantic_rank"
        if rank_col in df_query.columns:
            ranks = df_query[rank_col].values
            for i, r in enumerate(ranks):
                if int(r) == 5:
                    scores[i] = 1.0
                    return scores
        scores[0] = 1.0
        return scores


class S1_BM25Selector(BasePassageSelector):
    """S1: Selects candidate with highest BM25 score."""

    def predict_scores(self, df_query: pd.DataFrame) -> np.ndarray:
        if "bm25_score" in df_query.columns:
            return df_query["bm25_score"].fillna(-999.0).values
        elif "bm25_rank" in df_query.columns:
            return -df_query["bm25_rank"].fillna(999.0).values
        return np.zeros(len(df_query))


class S2_SemanticSelector(BasePassageSelector):
    """S2: Selects candidate with highest MiniLM semantic similarity."""

    def predict_scores(self, df_query: pd.DataFrame) -> np.ndarray:
        if "semantic_score" in df_query.columns:
            return df_query["semantic_score"].fillna(-999.0).values
        elif "semantic_rank" in df_query.columns:
            return -df_query["semantic_rank"].fillna(999.0).values
        return np.zeros(len(df_query))


class S3_DraftSimilaritySelector(BasePassageSelector):
    """S3: Selects candidate with highest draft <-> candidate similarity."""

    def __init__(self, semantic_scorer: Any = None):
        self.scorer = semantic_scorer

    def predict_scores(self, df_query: pd.DataFrame) -> np.ndarray:
        if "draft_relevance_ry" in df_query.columns:
            return df_query["draft_relevance_ry"].fillna(0.0).values
        # Fallback: compute Jaccard lexical similarity between draft and candidate
        draft_str = str(df_query.iloc[0].get("draft_answer", "")).lower()
        d_words = set(draft_str.split())
        scores = []
        for _, row in df_query.iterrows():
            c_text = str(row.get("candidate_text", "")).lower()
            c_words = set(c_text.split())
            inter = len(d_words & c_words)
            union = len(d_words | c_words)
            scores.append(inter / max(union, 1))
        return np.array(scores)


class S4_MarginalInfoSelector(BasePassageSelector):
    """S4: Selects candidate with highest Marginal Information Gain (MIG)."""

    def predict_scores(self, df_query: pd.DataFrame) -> np.ndarray:
        if "mig_score" in df_query.columns:
            return df_query["mig_score"].fillna(-999.0).values
        raise ValueError("S4 requires computed mig_score; no zero-score fallback")


class OracleSelector(BasePassageSelector):
    """Oracle: Upper-bound selector picking candidate with true maximum Delta F1."""

    def predict_scores(self, df_query: pd.DataFrame) -> np.ndarray:
        return df_query["delta_f1"].fillna(-999.0).values


# ---------------------------------------------------------------------------
# U0: Tabular Utility Model
# ---------------------------------------------------------------------------
class UtilityModelU0(BasePassageSelector):
    """U0: Tabular baseline using GBDT regressors/classifiers on non-leaking features."""

    def __init__(self, random_state: int = 42):
        self.clf_help = HistGradientBoostingClassifier(random_state=random_state, early_stopping=False)
        self.clf_hurt = HistGradientBoostingClassifier(random_state=random_state, early_stopping=False)
        self.reg_delta = HistGradientBoostingRegressor(random_state=random_state, early_stopping=False)
        self.reg_regret = HistGradientBoostingRegressor(random_state=random_state, early_stopping=False)
        self.clf_3way = HistGradientBoostingClassifier(random_state=random_state, early_stopping=False)
        self.feature_cols: List[str] = []

    def fit(
        self,
        X: np.ndarray,
        y_help: np.ndarray,
        y_hurt: np.ndarray,
        y_delta: np.ndarray,
        y_3way: np.ndarray,
        y_regret: Optional[np.ndarray] = None,
        feature_cols: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        if feature_cols:
            assert_no_feature_leakage(feature_cols)
            self.feature_cols = feature_cols

        if len(np.unique(y_help)) > 1:
            self.clf_help.fit(X, y_help)
            self.fitted_help = True
        else:
            self.fitted_help = False
            self.const_help = float(y_help[0]) if len(y_help) > 0 else 0.0

        if len(np.unique(y_hurt)) > 1:
            self.clf_hurt.fit(X, y_hurt)
            self.fitted_hurt = True
        else:
            self.fitted_hurt = False
            self.const_hurt = float(y_hurt[0]) if len(y_hurt) > 0 else 0.0

        self.reg_delta.fit(X, y_delta)

        if y_regret is not None:
            self.reg_regret.fit(X, y_regret)
            self.fitted_regret = True
        else:
            self.fitted_regret = False

        if len(np.unique(y_3way)) > 1:
            self.clf_3way.fit(X, y_3way)
            self.fitted_3way = True
        else:
            self.fitted_3way = False
            self.const_3way = int(y_3way[0]) if len(y_3way) > 0 else 0

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        n = len(X)
        p_help = self.clf_help.predict_proba(X)[:, 1] if getattr(self, "fitted_help", False) else np.full(n, getattr(self, "const_help", 0.0))
        p_hurt = self.clf_hurt.predict_proba(X)[:, 1] if getattr(self, "fitted_hurt", False) else np.full(n, getattr(self, "const_hurt", 0.0))
        pred_delta = self.reg_delta.predict(X)
        pred_regret = self.reg_regret.predict(X) if getattr(self, "fitted_regret", False) else np.zeros(n)

        # Composite utility score: P(HELP) - P(HURT)
        utility_score = p_help - p_hurt

        return {
            "prob_help": p_help,
            "prob_hurt": p_hurt,
            "pred_delta": pred_delta,
            "pred_regret": pred_regret,
            "utility_score": utility_score,
        }

    def predict_scores(self, df_query: pd.DataFrame) -> np.ndarray:
        if not self.feature_cols:
            raise ValueError("U0 feature_cols not set.")
        X = df_query[self.feature_cols].fillna(0.0).values
        preds = self.predict(X)
        return preds["utility_score"]


def train_u0_paper_grouped_oof(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 5,
) -> Tuple[Dict[str, np.ndarray], UtilityModelU0]:
    """Train U0 with paper-grouped Out-Of-Fold (OOF) cross-validation on FIT split.

    Guarantees zero paper leakage across folds.
    Returns:
        (oof_predictions_dict, final_fitted_u0_model)
    """
    assert_no_feature_leakage(feature_cols)

    oof_preds = {
        "prob_help": np.zeros(len(df)),
        "prob_hurt": np.zeros(len(df)),
        "pred_delta": np.zeros(len(df)),
        "pred_regret": np.zeros(len(df)),
        "utility_score": np.zeros(len(df)),
    }

    X = df[feature_cols].fillna(0.0).values
    y_help = df["target_help_binary"].values
    y_hurt = df["target_hurt_binary"].values
    y_delta = df["target_continuous_delta"].values
    y_3way = df["target_3way_label"].values
    y_regret = df["target_regret"].values if "target_regret" in df.columns else None

    splits = list(get_paper_grouped_splits(df, n_splits=n_splits))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        u0_fold = UtilityModelU0()
        u0_fold.fit(
            X[train_idx],
            y_help[train_idx],
            y_hurt[train_idx],
            y_delta[train_idx],
            y_3way[train_idx],
            y_regret=y_regret[train_idx] if y_regret is not None else None,
            feature_cols=feature_cols,
        )
        fold_preds = u0_fold.predict(X[val_idx])
        for k in oof_preds:
            oof_preds[k][val_idx] = fold_preds[k]

    # Final model fit on all FIT rows
    final_u0 = UtilityModelU0()
    final_u0.fit(X, y_help, y_hurt, y_delta, y_3way, y_regret=y_regret, feature_cols=feature_cols)
    return oof_preds, final_u0


# ---------------------------------------------------------------------------
# U1 & U2: Cross-Encoder Architectures
# ---------------------------------------------------------------------------
class CrossEncoderUtilityModel(BasePassageSelector, nn.Module):
    """Pretrained Cross-Encoder with multitask + ranking heads.

    Supports:
    - U1: Pre-generation Cross-Encoder on [q, D_i]
    - U2-F: Draft-Conditioned Cross-Encoder with frozen backbone
    - U2-A: Parameter-efficient adapted backbone (unfrozen top transformer layers)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        freeze_backbone: bool = False,
        adapted_top_layers: int = 0,
        mode: str = "U2",  # 'U1' or 'U2'
    ):
        super().__init__()
        self.mode = mode
        self.model_name = model_name
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif adapted_top_layers > 0:
            # Freeze all but top N encoder layers
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Unfreeze top N transformer layers
            if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
                for layer in self.backbone.encoder.layer[-adapted_top_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True

        hidden_size = self.backbone.config.hidden_size
        self.heads = MultitaskUtilityHeads(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**kwargs)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        preds = self.heads(cls_rep)
        preds["ce_embedding"] = cls_rep
        return preds

    def predict_scores(self, df_query: pd.DataFrame) -> np.ndarray:
        """Tokenize and score candidates for a query in evaluation mode."""
        self.eval()
        device = next(self.parameters()).device
        texts = []
        q = str(df_query.iloc[0].get("question", ""))
        d = str(df_query.iloc[0].get("draft_answer", ""))

        for _, row in df_query.iterrows():
            c = str(row.get("candidate_text", ""))
            if self.mode == "U1":
                texts.append(truncate_and_budget_tokens(self.tokenizer, q, "", c)[0])
            else:
                formatted, _ = truncate_and_budget_tokens(self.tokenizer, q, d, c)
                texts.append(formatted)

        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            preds = self.forward(**enc)
            scores = preds["utility_score"].cpu().numpy()
        return scores


# ---------------------------------------------------------------------------
# U4: Hybrid Fusion Utility Models (U4-A, U4-B)
# ---------------------------------------------------------------------------
class FullHybridUtilityModelU4(BasePassageSelector, nn.Module):
    """U4: Fuses U2 CrossEncoder embedding + non-leaking numeric features via MLP.

    Variants:
    - U4-A: U2 embedding + tabular/MIG features
    - U4-B: U4-A + generator attribution features (A1..A6)
    """

    def __init__(
        self,
        cross_encoder_dim: int = 384,
        numeric_dim: int = 11,
        hidden_dim: int = 128,
        has_attribution: bool = False,
        ce_dim: Optional[int] = None,
        num_numeric_features: Optional[int] = None,
    ):
        super().__init__()
        self.has_attribution = has_attribution
        cross_encoder_dim = ce_dim if ce_dim is not None else cross_encoder_dim
        numeric_dim = num_numeric_features if num_numeric_features is not None else numeric_dim
        total_in = cross_encoder_dim + numeric_dim

        self.fusion = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.heads = MultitaskUtilityHeads(hidden_dim)

    def forward(
        self,
        ce_embedding: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        fused = torch.cat([ce_embedding, numeric_features], dim=-1)
        feat = self.fusion(fused)
        return self.heads(feat)


# ---------------------------------------------------------------------------
# Cross-Encoder Input Preparation Helper
# ---------------------------------------------------------------------------
def prepare_cross_encoder_inputs(
    tokenizer: Any = None,
    queries: Optional[Union[List[str], pd.Series, pd.DataFrame, str]] = None,
    candidates: Optional[Union[List[str], pd.Series, str]] = None,
    drafts: Optional[Union[List[str], pd.Series, str]] = None,
    max_length: int = 512,
    device: Optional[Union[str, torch.device]] = None,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> Dict[str, torch.Tensor]:
    """Tokenize query-candidate (U1) or query-draft-candidate (U2) inputs with strict budgeting."""
    if tokenizer is not None and isinstance(tokenizer, pd.DataFrame):
        queries, tokenizer = tokenizer, queries

    if isinstance(queries, pd.DataFrame):
        df = queries
        q_col = next((c for c in ["question", "query", "question_text"] if c in df.columns), None)
        c_col = next((c for c in ["candidate_text", "candidate", "passage", "passage_text"] if c in df.columns), None)
        d_col = next((c for c in ["draft_answer", "draft", "base_prediction"] if c in df.columns), None)

        q_list = df[q_col].astype(str).tolist() if q_col else [""] * len(df)
        c_list = df[c_col].astype(str).tolist() if c_col else [""] * len(df)
        d_list = df[d_col].astype(str).tolist() if d_col else None
        return prepare_cross_encoder_inputs(
            tokenizer=tokenizer,
            queries=q_list,
            candidates=c_list,
            drafts=d_list,
            max_length=max_length,
            device=device,
            model_name=model_name,
        )

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if isinstance(queries, str):
        queries = [queries]
    elif isinstance(queries, pd.Series):
        queries = queries.tolist()
    elif queries is None:
        queries = []

    if isinstance(candidates, str):
        candidates = [candidates]
    elif isinstance(candidates, pd.Series):
        candidates = candidates.tolist()
    elif candidates is None:
        candidates = [""] * len(queries)

    if isinstance(drafts, str):
        drafts = [drafts] * len(queries)
    elif isinstance(drafts, pd.Series):
        drafts = drafts.tolist()

    texts = []
    for i in range(len(queries)):
        q = str(queries[i])
        c = str(candidates[i]) if i < len(candidates) else ""
        if drafts is not None and i < len(drafts) and drafts[i] is not None:
            formatted, _ = truncate_and_budget_tokens(tokenizer, q, str(drafts[i]), c, max_length=max_length)
            texts.append(formatted)
        else:
            texts.append(format_u1_text(q, c))

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if device is not None:
        encoded = {k: v.to(device) for k, v in encoded.items()}
    return encoded


# Aliases
TabularUtilityModel_U0 = UtilityModelU0
PassageCrossEncoder_U1 = CrossEncoderUtilityModel
DraftConditionedCrossEncoder_U2 = CrossEncoderUtilityModel
FullHybridPassageUtilityModel_U4 = FullHybridUtilityModelU4
HybridFusionUtilityModelU4 = FullHybridUtilityModelU4
