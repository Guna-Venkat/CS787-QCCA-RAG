"""Week 5: Nonlinear Stage-2 ML Modeling Benchmark.

Investigates whether small nonlinear models (Decision Tree, Random Forest,
XGBoost, LightGBM, HistGradientBoosting, and Small MLP) can improve upon
the Balanced Logistic Regression baseline for the QCCA-V2 Stage-2 (4 -> 6)
expansion decision under strict paper-disjoint GroupKFold cross-validation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Frozen Context Token Costs (empirical QASPER dev means)
CONTEXT_TOKEN_COSTS = {
    2: 534.5,
    4: 946.1,
    5: 1157.0,
    6: 1359.1,
    8: 1758.5,
}

# The 5 validated Stage-2 core features
STAGE_2_CORE_FEATURES = [
    "bm25_mean_score",
    "lex_coverage_top2",
    "sem_sim_mean_top6",
    "evidence_query_cluster_span",
    "evidence_lexical_overlap_mean",
]

# The validated 2-feature subset
STAGE_2_SUBSET_FEATURES = [
    "bm25_mean_score",
    "lex_coverage_top2",
]

# Standard threshold grid
DEFAULT_THRESHOLDS = [
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70
]


# =============================================================================
# DATA LOADING & NORMALIZATION
# =============================================================================

def normalize_qid(qid: Any) -> str:
    """Normalize question IDs for consistent merging across tables."""
    s = str(qid).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def load_week5_dev_data(
    feature_matrix_path: str = "results/week4/feature_discovery/feature_matrix.csv",
    dev_matrix_path: str = "results/week2/processed/dev_per_query_k_matrix.csv",
    oracle_path: str = "results/week3/processed/epsilon_oracle_all.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load development split features and response surfaces, constructing Stage-2 targets.
    
    Returns:
        (df_features, df_targets, df_merged)
    """
    p_feat = Path(feature_matrix_path)
    p_dev = Path(dev_matrix_path)
    p_orc = Path(oracle_path)

    if not p_feat.is_file():
        raise FileNotFoundError(f"Feature matrix not found: {p_feat}")
    if not p_dev.is_file():
        raise FileNotFoundError(f"Dev response matrix not found: {p_dev}")

    df_f = pd.read_csv(p_feat)
    df_t = pd.read_csv(p_dev)

    df_f["question_id"] = df_f["question_id"].apply(normalize_qid)
    df_f["paper_id"] = df_f["paper_id"].astype(str).str.strip()

    df_t["question_id"] = df_t["question_id"].apply(normalize_qid)
    df_t["paper_id"] = df_t["paper_id"].astype(str).str.strip()

    # Merge response surface with features
    df_merged = pd.merge(
        df_f,
        df_t[["question_id", "F1_k2", "F1_k4", "F1_k5", "F1_k6", "F1_k8"]],
        on="question_id",
        how="inner",
    )

    # Optional merge with epsilon oracle
    if p_orc.is_file():
        df_o = pd.read_csv(p_orc)
        df_o["question_id"] = df_o["question_id"].apply(normalize_qid)
        f1_col = "oracle_f1" if "oracle_f1" in df_o.columns else ("f1_oracle" if "f1_oracle" in df_o.columns else None)
        if f1_col:
            df_merged = pd.merge(
                df_merged,
                df_o[["question_id", f1_col]].rename(columns={f1_col: "oracle_f1"}),
                on="question_id",
                how="left",
            )
        else:
            df_merged["oracle_f1"] = df_merged[["F1_k2", "F1_k4", "F1_k5", "F1_k6", "F1_k8"]].max(axis=1)
    else:
        df_merged["oracle_f1"] = df_merged[["F1_k2", "F1_k4", "F1_k5", "F1_k6", "F1_k8"]].max(axis=1)

    # Marginal transition definitions for Stage 2 (4 -> 6)
    df_merged["G_4_6"] = df_merged["F1_k6"] - df_merged["F1_k4"]
    df_merged["Y_4_6"] = (df_merged["G_4_6"] > 0.01).astype(int)

    return df_f, df_t, df_merged


# =============================================================================
# POLICY C1 SIMULATION & EVALUATION
# =============================================================================

def simulate_policy_c1(
    df_merged: pd.DataFrame,
    oof_probs: np.ndarray,
    threshold: float = 0.50,
    policy_name: str = "Policy C1",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Simulate Policy C1 on out-of-fold probability predictions.
    
    Rule:
    1. Start query at k=4.
    2. If P(expand) >= threshold, allocate k=6.
    3. Else stop at k=4.
    """
    n = len(df_merged)
    if len(oof_probs) != n:
        raise ValueError(f"Length mismatch: {len(oof_probs)} probs vs {n} queries.")

    expand_mask = oof_probs >= threshold
    selected_k = np.where(expand_mask, 6, 4)
    actual_f1s = np.where(expand_mask, df_merged["F1_k6"].values, df_merged["F1_k4"].values)
    actual_tokens = np.where(expand_mask, CONTEXT_TOKEN_COSTS[6], CONTEXT_TOKEN_COSTS[4])

    oracle_f1 = df_merged["oracle_f1"].values if "oracle_f1" in df_merged.columns else np.maximum(df_merged["F1_k4"], df_merged["F1_k6"])
    regret = np.maximum(0.0, oracle_f1 - actual_f1s)

    y_true = df_merged["Y_4_6"].values.astype(int)
    gains = df_merged["G_4_6"].values.astype(float)

    # Quadrant breakdown
    tp_mask = expand_mask & (y_true == 1)
    tn_mask = (~expand_mask) & (y_true == 0)
    fp_mask = expand_mask & (y_true == 0)   # False Expansion (Unneeded context)
    fn_mask = (~expand_mask) & (y_true == 1) # False Stop (Missed gain)

    tp_count = int(np.sum(tp_mask))
    tn_count = int(np.sum(tn_mask))
    fp_count = int(np.sum(fp_mask))
    fn_count = int(np.sum(fn_mask))

    # Mean captured gain and distractor penalty
    avg_gain_captured = float(np.mean(gains[tp_mask])) if tp_count > 0 else 0.0
    distractor_penalties = gains[fp_mask][gains[fp_mask] < 0]
    avg_distractor_penalty = float(np.mean(np.abs(distractor_penalties))) if len(distractor_penalties) > 0 else 0.0

    mean_f1 = float(np.mean(actual_f1s))
    mean_k = float(np.mean(selected_k))
    mean_tokens = float(np.mean(actual_tokens))
    context_red_k8 = float((CONTEXT_TOKEN_COSTS[8] - mean_tokens) / CONTEXT_TOKEN_COSTS[8] * 100.0)
    mean_regret = float(np.mean(regret))
    zero_reg_pct = float(np.mean(regret == 0.0) * 100.0)

    summary = {
        "policy_name": policy_name,
        "threshold": threshold,
        "mean_f1": round(mean_f1, 4),
        "mean_k": round(mean_k, 2),
        "mean_context_tokens": round(mean_tokens, 1),
        "context_reduction_pct_vs_k8": round(context_red_k8, 2),
        "mean_regret": round(mean_regret, 4),
        "fraction_zero_regret": round(zero_reg_pct, 2),
        "expansion_rate_pct": round(float(np.mean(expand_mask) * 100.0), 2),
        "tp_count": tp_count,
        "tn_count": tn_count,
        "fp_count": fp_count,
        "fn_count": fn_count,
        "false_expansion_pct": round(float(fp_count / n * 100.0), 2),
        "false_stop_pct": round(float(fn_count / n * 100.0), 2),
        "true_positive_rate_pct": round(float(tp_count / max(1, tp_count + fn_count) * 100.0), 2),
        "avg_gain_captured": round(avg_gain_captured, 4),
        "avg_distractor_penalty": round(avg_distractor_penalty, 4),
    }

    df_query_results = pd.DataFrame({
        "question_id": df_merged["question_id"],
        "paper_id": df_merged["paper_id"],
        "selected_k": selected_k,
        "policy_f1": np.round(actual_f1s, 4),
        "context_tokens": actual_tokens,
        "regret": np.round(regret, 4),
        "is_expanded": expand_mask.astype(int),
        "is_tp": tp_mask.astype(int),
        "is_tn": tn_mask.astype(int),
        "is_fp": fp_mask.astype(int),
        "is_fn": fn_mask.astype(int),
    })

    return df_query_results, summary


# =============================================================================
# MODEL ARCHITECTURES & WRAPPERS
# =============================================================================

class SmallPyTorchMLP(BaseEstimator, ClassifierMixin):
    """Small PyTorch Multi-Layer Perceptron with Class-Weighted BCE Loss."""

    def __init__(
        self,
        hidden_sizes: Tuple[int, ...] = (8,),
        learning_rate: float = 0.01,
        weight_decay: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        pos_weight: float = 4.13,
        patience: int = 15,
        random_state: int = 42,
    ):
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.pos_weight = pos_weight
        self.patience = patience
        self.random_state = random_state
        self.model_ = None
        self.classes_ = np.array([0, 1])

    def _build_network(self, in_features: int) -> nn.Module:
        torch.manual_seed(self.random_state)
        layers = []
        prev_dim = in_features
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if not HAS_TORCH:
            raise ImportError("PyTorch is not available.")
        
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        in_dim = X_arr.shape[1]
        self.model_ = self._build_network(in_dim)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight], dtype=torch.float32))
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        dataset = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model_(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            epoch_loss /= len(X_arr)
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        self.model_.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X_arr)).numpy().flatten()
            probs_1 = 1.0 / (1.0 + np.exp(-logits))
        probs_0 = 1.0 - probs_1
        return np.column_stack([probs_0, probs_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


# =============================================================================
# LEAKAGE-FREE INNER-TUNED TRAINING & OUT-OF-FOLD EVALUATION
# =============================================================================

def get_candidate_model_configs(model_family: str, seed: int = 42) -> List[Dict[str, Any]]:
    """Return controlled hyperparameter candidates for each model family."""
    if model_family == "logistic_regression":
        return [{"class_weight": "balanced", "C": 1.0}]
    
    elif model_family == "decision_tree":
        configs = []
        for depth in [2, 3, 4, 5]:
            for min_leaf in [5, 10]:
                configs.append({
                    "max_depth": depth,
                    "min_samples_leaf": min_leaf,
                    "class_weight": "balanced",
                })
        return configs

    elif model_family == "random_forest":
        configs = []
        for depth in [2, 3, 4]:
            for min_leaf in [3, 5]:
                configs.append({
                    "n_estimators": 50,
                    "max_depth": depth,
                    "min_samples_leaf": min_leaf,
                    "class_weight": "balanced",
                })
        return configs

    elif model_family == "hist_gradient_boosting":
        configs = []
        for depth in [2, 3]:
            for lr in [0.03, 0.05]:
                configs.append({
                    "max_iter": 50,
                    "max_depth": depth,
                    "learning_rate": lr,
                    "class_weight": "balanced",
                    "min_samples_leaf": 5,
                })
        return configs

    elif model_family == "lightgbm":
        if not HAS_LIGHTGBM:
            return []
        configs = []
        for depth in [2, 3]:
            for lr in [0.03, 0.05]:
                configs.append({
                    "n_estimators": 50,
                    "max_depth": depth,
                    "num_leaves": 2 ** depth - 1,
                    "learning_rate": lr,
                    "min_child_samples": 5,
                    "class_weight": "balanced",
                    "verbose": -1,
                })
        return configs

    elif model_family == "xgboost":
        if not HAS_XGBOOST:
            return []
        configs = []
        for depth in [2, 3]:
            for lr in [0.03, 0.05]:
                configs.append({
                    "n_estimators": 50,
                    "max_depth": depth,
                    "learning_rate": lr,
                    "scale_pos_weight": 4.13,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "eval_metric": "logloss",
                })
        return configs

    elif model_family == "mlp":
        if not HAS_TORCH:
            return []
        return [
            {"hidden_sizes": (8,), "learning_rate": 0.01, "weight_decay": 1e-3},
            {"hidden_sizes": (16, 8), "learning_rate": 0.008, "weight_decay": 1e-3},
        ]
    else:
        raise ValueError(f"Unknown model_family: {model_family}")


def instantiate_model(model_family: str, params: Dict[str, Any], seed: int = 42) -> Any:
    """Instantiate a model pipeline given family and hyperparameter config."""
    if model_family == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", random_state=seed, **params))
        ])
    elif model_family == "decision_tree":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(random_state=seed, **params))
        ])
    elif model_family == "random_forest":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=seed, **params))
        ])
    elif model_family == "hist_gradient_boosting":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(random_state=seed, **params))
        ])
    elif model_family == "lightgbm":
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed.")
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", lgb.LGBMClassifier(random_state=seed, **params))
        ])
    elif model_family == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed.")
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", xgb.XGBClassifier(random_state=seed, **params))
        ])
    elif model_family == "mlp":
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed.")
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SmallPyTorchMLP(random_state=seed, **params))
        ])
    else:
        raise ValueError(f"Unknown model_family: {model_family}")


def train_eval_model_oof(
    df_merged: pd.DataFrame,
    model_family: str,
    feature_cols: List[str] = STAGE_2_CORE_FEATURES,
    target_col: str = "Y_4_6",
    n_outer_splits: int = 5,
    n_inner_splits: int = 3,
    calibrate: bool = False,
    calibration_method: str = "sigmoid",
    seed: int = 42,
) -> Dict[str, Any]:
    """Perform leak-free paper-disjoint GroupKFold OOF evaluation with inner-CV tuning."""
    gkf_outer = GroupKFold(n_splits=n_outer_splits)
    outer_groups = df_merged["paper_id"].values

    n = len(df_merged)
    oof_probs = np.zeros(n, dtype=float)
    oof_preds = np.zeros(n, dtype=int)
    oof_preds_tuned_thresh = np.zeros(n, dtype=int)
    fold_ids = np.zeros(n, dtype=int)

    y_true = df_merged[target_col].values.astype(int)
    X_df = df_merged[feature_cols].copy()

    candidate_configs = get_candidate_model_configs(model_family, seed=seed)
    if len(candidate_configs) == 0:
        raise RuntimeError(f"No candidate configs available for {model_family} (missing dependencies?)")

    fold_best_configs = []
    fold_best_thresholds = []

    for fold, (train_idx, test_idx) in enumerate(gkf_outer.split(X_df, y_true, groups=outer_groups)):
        X_train_outer, y_train_outer = X_df.iloc[train_idx], y_true[train_idx]
        train_groups_outer = outer_groups[train_idx]
        X_test_outer, y_test_outer = X_df.iloc[test_idx], y_true[test_idx]

        fold_ids[test_idx] = fold

        # -------------------------------------------------------------
        # Inner-CV: Select Hyperparameters Leak-Free on Training Fold
        # -------------------------------------------------------------
        best_cfg = candidate_configs[0]
        if len(candidate_configs) > 1:
            gkf_inner = GroupKFold(n_splits=min(n_inner_splits, len(np.unique(train_groups_outer))))
            config_scores = []

            for cfg in candidate_configs:
                inner_cv_scores = []
                for in_tr_idx, in_val_idx in gkf_inner.split(X_train_outer, y_train_outer, groups=train_groups_outer):
                    X_in_tr, y_in_tr = X_train_outer.iloc[in_tr_idx], y_train_outer[in_tr_idx]
                    X_in_val, y_in_val = X_train_outer.iloc[in_val_idx], y_train_outer[in_val_idx]

                    # Skip fold if single class in training
                    if len(np.unique(y_in_tr)) < 2:
                        continue

                    model = instantiate_model(model_family, cfg, seed=seed + fold)
                    model.fit(X_in_tr, y_in_tr)
                    probs_pred = model.predict_proba(X_in_val)
                    in_probs = probs_pred[:, 1] if probs_pred.shape[1] > 1 else np.full(len(X_in_val), float(y_in_tr[0]))
                    try:
                        score = average_precision_score(y_in_val, in_probs)
                    except Exception:
                        score = 0.0
                    inner_cv_scores.append(score)

                mean_score = float(np.mean(inner_cv_scores)) if len(inner_cv_scores) > 0 else 0.0
                config_scores.append(mean_score)

            best_idx = int(np.argmax(config_scores))
            best_cfg = candidate_configs[best_idx]

        fold_best_configs.append(best_cfg)

        # -------------------------------------------------------------
        # Train Outer Model with Best Config
        # -------------------------------------------------------------
        final_model = instantiate_model(model_family, best_cfg, seed=seed + fold)

        if calibrate:
            # Calibrate model using inner CV strictly on training fold
            inner_splits = list(GroupKFold(n_splits=3).split(X_train_outer, y_train_outer, groups=train_groups_outer))
            final_model = CalibratedClassifierCV(
                estimator=final_model,
                method=calibration_method,
                cv=inner_splits,
            )
            final_model.fit(X_train_outer, y_train_outer)
        else:
            final_model.fit(X_train_outer, y_train_outer)

        # Outer test fold prediction
        test_probs_all = final_model.predict_proba(X_test_outer)
        test_probs = test_probs_all[:, 1] if test_probs_all.shape[1] > 1 else np.full(len(X_test_outer), float(y_train_outer[0]))
        oof_probs[test_idx] = test_probs
        oof_preds[test_idx] = (test_probs >= 0.50).astype(int)

        # -------------------------------------------------------------
        # Inner-CV Threshold Selection (Strictly on Training Fold)
        # -------------------------------------------------------------
        # Generate out-of-fold predictions on training fold to select optimal threshold
        gkf_thresh = GroupKFold(n_splits=min(3, len(np.unique(train_groups_outer))))
        inner_probs_arr = np.zeros(len(X_train_outer))
        for in_tr_idx, in_val_idx in gkf_thresh.split(X_train_outer, y_train_outer, groups=train_groups_outer):
            if len(np.unique(y_train_outer[in_tr_idx])) < 2:
                inner_probs_arr[in_val_idx] = float(np.mean(y_train_outer[in_tr_idx]))
                continue
            sub_m = instantiate_model(model_family, best_cfg, seed=seed + fold)
            sub_m.fit(X_train_outer.iloc[in_tr_idx], y_train_outer[in_tr_idx])
            in_p = sub_m.predict_proba(X_train_outer.iloc[in_val_idx])
            inner_probs_arr[in_val_idx] = in_p[:, 1] if in_p.shape[1] > 1 else in_p[:, 0]

        thresh_scores = []
        for t in DEFAULT_THRESHOLDS:
            p_bin = (inner_probs_arr >= t).astype(int)
            bacc = balanced_accuracy_score(y_train_outer, p_bin)
            thresh_scores.append(bacc)

        best_t = DEFAULT_THRESHOLDS[int(np.argmax(thresh_scores))]
        fold_best_thresholds.append(best_t)
        oof_preds_tuned_thresh[test_idx] = (test_probs >= best_t).astype(int)

    # -----------------------------------------------------------------
    # Compute Classification Metrics Across All Queries
    # -----------------------------------------------------------------
    try:
        roc_auc = float(roc_auc_score(y_true, oof_probs))
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true, oof_probs))
    except Exception:
        pr_auc = float(np.mean(y_true))

    bal_acc = float(balanced_accuracy_score(y_true, oof_preds))
    prec = float(precision_score(y_true, oof_preds, zero_division=0))
    rec = float(recall_score(y_true, oof_preds, zero_division=0))
    f1_cls = float(f1_score(y_true, oof_preds, zero_division=0))
    brier = float(brier_score_loss(y_true, oof_probs))

    # Downstream Policy C1 evaluation at default threshold 0.50
    df_policy_res, policy_summary = simulate_policy_c1(
        df_merged,
        oof_probs,
        threshold=0.50,
        policy_name=f"{model_family} (t=0.50)",
    )

    # Downstream Policy C1 evaluation at inner-tuned threshold
    mean_tuned_thresh = float(np.mean(fold_best_thresholds))
    df_policy_res_tuned, policy_summary_tuned = simulate_policy_c1(
        df_merged,
        oof_probs,
        threshold=mean_tuned_thresh,
        policy_name=f"{model_family} (tuned t={mean_tuned_thresh:.2f})",
    )

    return {
        "model_family": model_family,
        "feature_count": len(feature_cols),
        "calibrated": calibrate,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "balanced_acc": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1_cls, 4),
        "brier_score": round(brier, 4),
        "oof_probs": oof_probs,
        "oof_preds": oof_preds,
        "oof_preds_tuned_thresh": oof_preds_tuned_thresh,
        "fold_best_configs": fold_best_configs,
        "fold_best_thresholds": fold_best_thresholds,
        "policy_summary": policy_summary,
        "policy_summary_tuned": policy_summary_tuned,
        "df_policy_results": df_policy_res,
        "df_policy_results_tuned": df_policy_res_tuned,
    }


# =============================================================================
# THRESHOLD SENSITIVITY SWEEPER
# =============================================================================

def evaluate_model_threshold_grid(
    df_merged: pd.DataFrame,
    oof_probs: np.ndarray,
    model_name: str,
    thresholds: List[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Evaluate Policy C1 performance across a grid of decision thresholds."""
    records = []
    for t in thresholds:
        _, sm = simulate_policy_c1(df_merged, oof_probs, threshold=t, policy_name=model_name)
        records.append({
            "model_name": model_name,
            "threshold": t,
            "mean_f1": sm["mean_f1"],
            "mean_k": sm["mean_k"],
            "mean_context_tokens": sm["mean_context_tokens"],
            "context_reduction_pct_vs_k8": sm["context_reduction_pct_vs_k8"],
            "mean_regret": sm["mean_regret"],
            "fraction_zero_regret": sm["fraction_zero_regret"],
            "expansion_rate_pct": sm["expansion_rate_pct"],
            "false_expansion_pct": sm["false_expansion_pct"],
            "false_stop_pct": sm["false_stop_pct"],
            "avg_gain_captured": sm["avg_gain_captured"],
            "avg_distractor_penalty": sm["avg_distractor_penalty"],
        })
    return pd.DataFrame(records)


# =============================================================================
# STATISTICAL COMPARISONS (PAPER-CLUSTERED PAIRED BOOTSTRAP)
# =============================================================================

def run_paper_clustered_paired_bootstrap(
    df_merged: pd.DataFrame,
    candidate_policy_res: pd.DataFrame,
    baseline_policy_res: pd.DataFrame,
    baseline_name: str = "Balanced Logistic Regression",
    candidate_name: str = "Candidate Model",
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run paper-clustered paired bootstrap comparing candidate policy against baseline."""
    np.random.seed(seed)
    unique_papers = np.array(df_merged["paper_id"].unique())
    n_papers = len(unique_papers)

    paper_to_indices = {p: np.where(df_merged["paper_id"] == p)[0] for p in unique_papers}

    delta_f1_list = []
    delta_tokens_list = []
    delta_regret_list = []

    cand_f1 = candidate_policy_res["policy_f1"].values
    cand_tok = candidate_policy_res["context_tokens"].values
    cand_reg = candidate_policy_res["regret"].values

    base_f1 = baseline_policy_res["policy_f1"].values
    base_tok = baseline_policy_res["context_tokens"].values
    base_reg = baseline_policy_res["regret"].values

    for _ in range(n_bootstrap):
        sampled_papers = np.random.choice(unique_papers, size=n_papers, replace=True)
        sampled_idx = np.concatenate([paper_to_indices[p] for p in sampled_papers])

        delta_f1 = float(np.mean(cand_f1[sampled_idx]) - np.mean(base_f1[sampled_idx]))
        delta_tok = float(np.mean(cand_tok[sampled_idx]) - np.mean(base_tok[sampled_idx]))
        delta_reg = float(np.mean(cand_reg[sampled_idx]) - np.mean(base_reg[sampled_idx]))

        delta_f1_list.append(delta_f1)
        delta_tokens_list.append(delta_tok)
        delta_regret_list.append(delta_reg)

    ci_f1 = np.percentile(delta_f1_list, [2.5, 97.5])
    ci_tok = np.percentile(delta_tokens_list, [2.5, 97.5])
    ci_reg = np.percentile(delta_regret_list, [2.5, 97.5])

    mean_delta_f1 = float(np.mean(delta_f1_list))
    mean_delta_tok = float(np.mean(delta_tokens_list))
    mean_delta_reg = float(np.mean(delta_regret_list))

    is_f1_significant = bool(ci_f1[0] > 0.0 or ci_f1[1] < 0.0)
    is_tok_significant = bool(ci_tok[0] > 0.0 or ci_tok[1] < 0.0)

    return {
        "candidate": candidate_name,
        "baseline": baseline_name,
        "mean_delta_f1": round(mean_delta_f1, 4),
        "ci_95_f1_low": round(float(ci_f1[0]), 4),
        "ci_95_f1_high": round(float(ci_f1[1]), 4),
        "statistically_detectable_f1": is_f1_significant,
        "mean_delta_tokens": round(mean_delta_tok, 1),
        "ci_95_tokens_low": round(float(ci_tok[0]), 1),
        "ci_95_tokens_high": round(float(ci_tok[1]), 1),
        "statistically_detectable_tokens": is_tok_significant,
        "mean_delta_regret": round(mean_delta_reg, 4),
        "ci_95_regret_low": round(float(ci_reg[0]), 4),
        "ci_95_regret_high": round(float(ci_reg[1]), 4),
    }


# =============================================================================
# FEATURE INTERACTION & 2D GRID ANALYSIS
# =============================================================================

def analyze_feature_interactions(
    df_merged: pd.DataFrame,
    model_family: str,
    feature_cols: List[str] = STAGE_2_CORE_FEATURES,
    feat_x: str = "bm25_mean_score",
    feat_y: str = "evidence_lexical_overlap_mean",
    grid_size: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D response surface over BM25 score vs pairwise lexical overlap.
    
    Tests whether the model learns:
    high BM25 + high overlap -> HALT (distractor redundancy)
    vs
    high BM25 + low overlap -> EXPAND (complementary evidence).
    """
    X_train = df_merged[feature_cols].copy()
    y_train = df_merged["Y_4_6"].values.astype(int)

    cfg = get_candidate_model_configs(model_family, seed=seed)[0]
    model = instantiate_model(model_family, cfg, seed=seed)
    model.fit(X_train, y_train)

    x_vals = np.linspace(df_merged[feat_x].min(), df_merged[feat_x].max(), grid_size)
    y_vals = np.linspace(df_merged[feat_y].min(), df_merged[feat_y].max(), grid_size)
    xx, yy = np.meshgrid(x_vals, y_vals)

    # Median baseline for other features
    grid_points = []
    for xi, yi in zip(xx.ravel(), yy.ravel()):
        row = {}
        for col in feature_cols:
            if col == feat_x:
                row[col] = xi
            elif col == feat_y:
                row[col] = yi
            else:
                row[col] = float(df_merged[col].median())
        grid_points.append(row)

    df_grid = pd.DataFrame(grid_points)
    probs_grid = model.predict_proba(df_grid)[:, 1].reshape(xx.shape)

    return xx, yy, probs_grid
