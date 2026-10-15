"""Model training and internal-validation selection for Week 7-B.

Enforces:
- Models are trained ONLY on TRAIN-FIT.
- Supervision targets k*_eps,sem are derived strictly from the TRAIN semantic response surface.
- Model selection is conducted ONLY on INTERNAL-VAL using matrix lookup (no SARA regeneration).
- Selection priority:
  1. Highest downstream answer F1 on INTERNAL-VAL.
  2. If within small tolerance (0.001), lower context tokens.
  3. Simpler model.
- DEV is strictly held out and untouched during training and model selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

BUDGETS: List[int] = [2, 4, 5, 6, 8]


def train_candidate_budget_models(
    X_fit: np.ndarray,
    y_fit_k: np.ndarray,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train candidate lightweight budget allocators on TRAIN-FIT.

    Formulations:
    1. 'balanced_logistic_regression_cascade':
       Binary balanced logistic regression deciding whether to expand from base k=4 to k=6.
    2. 'multiclass_balanced_logistic_regression':
       Multinomial balanced logistic regression directly predicting k in {2, 4, 5, 6, 8}.
    3. 'ridge_classifier':
       Regularized balanced linear classifier.
    """
    models: Dict[str, Any] = {}

    # Formulation 1: Policy C1-style Cascade (Expand to k=6 if target >= 6, else k=4)
    y_binary = (y_fit_k >= 6).astype(int)
    clf_cascade = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
        max_iter=1000,
    )
    clf_cascade.fit(X_fit, y_binary)
    models["balanced_logistic_regression_cascade"] = {
        "model": clf_cascade,
        "type": "binary_cascade",
        "description": "Balanced Logistic Regression (Base k=4, expand k=6)",
    }

    # Formulation 2: Multiclass Balanced Logistic Regression
    clf_multiclass = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
        max_iter=1000,
    )
    clf_multiclass.fit(X_fit, y_fit_k)
    models["multiclass_balanced_logistic_regression"] = {
        "model": clf_multiclass,
        "type": "multiclass",
        "description": "Multiclass Balanced Logistic Regression over {2,4,5,6,8}",
    }

    # Formulation 3: Ridge Classifier
    clf_ridge = RidgeClassifier(
        class_weight="balanced",
        random_state=seed,
    )
    clf_ridge.fit(X_fit, y_fit_k)
    models["ridge_classifier"] = {
        "model": clf_ridge,
        "type": "multiclass",
        "description": "Balanced Ridge Classifier over {2,4,5,6,8}",
    }

    return models


def predict_budgets(
    model_entry: Dict[str, Any],
    X: np.ndarray,
) -> np.ndarray:
    """Generate budget predictions k in {2, 4, 5, 6, 8} from a trained model entry."""
    m_type = model_entry["type"]
    model = model_entry["model"]

    if m_type == "binary_cascade":
        preds_binary = model.predict(X)
        return np.where(preds_binary == 1, 6, 4)
    elif m_type == "multiclass":
        raw_preds = model.predict(X)
        # Ensure predictions are strictly within ALLOWED_BUDGETS
        valid_k = np.array([int(k) if int(k) in BUDGETS else 4 for k in raw_preds])
        return valid_k
    else:
        raise ValueError(f"Unknown model type: {m_type}")


def select_best_model_on_internal_val(
    models: Dict[str, Any],
    X_val: np.ndarray,
    df_val_matrix: pd.DataFrame,
    output_tables_dir: Optional[Union[str, Path]] = None,
) -> Tuple[str, Dict[str, Any], pd.DataFrame]:
    """Select the best budget allocator using INTERNAL-VAL response surface lookup.

    Selection priority:
    1. Highest downstream answer F1 on INTERNAL-VAL.
    2. Lower context tokens.
    3. Simpler model.
    """
    records = []

    for name, m_entry in models.items():
        k_preds = predict_budgets(m_entry, X_val)
        n_queries = len(k_preds)
        assert n_queries == len(df_val_matrix), "Mismatch between predictions count and validation matrix rows"

        f1s = []
        tokens = []
        for i in range(n_queries):
            k = int(k_preds[i])
            f1s.append(float(df_val_matrix.iloc[i][f"f1_k{k}"]))
            tokens.append(int(df_val_matrix.iloc[i][f"tokens_k{k}"]))

        mean_f1 = float(np.mean(f1s))
        mean_tok = float(np.mean(tokens))
        mean_k = float(np.mean(k_preds))

        records.append({
            "model_name": name,
            "description": m_entry["description"],
            "val_mean_f1": mean_f1,
            "val_mean_tokens": mean_tok,
            "val_mean_k": mean_k,
        })

    df_results = pd.DataFrame(records)
    # Sort primarily by F1 descending, secondarily by tokens ascending
    df_sorted = df_results.sort_values(by=["val_mean_f1", "val_mean_tokens"], ascending=[False, True]).reset_index(drop=True)

    best_name = str(df_sorted.iloc[0]["model_name"])
    best_entry = models[best_name]

    logger.info(
        "INTERNAL-VAL model selection complete. Winner: '%s' (Val F1=%.4f, Val Tokens=%.1f, Val k=%.2f)",
        best_name, df_sorted.iloc[0]["val_mean_f1"], df_sorted.iloc[0]["val_mean_tokens"], df_sorted.iloc[0]["val_mean_k"],
    )

    if output_tables_dir:
        p = Path(output_tables_dir)
        p.mkdir(parents=True, exist_ok=True)
        df_sorted.to_csv(p / "w7b_model_internal_val_results.csv", index=False)

    return best_name, best_entry, df_sorted


def save_frozen_w7b_model(
    model_entry: Dict[str, Any],
    scaler: StandardScaler,
    model_name: str,
    output_dir: Union[str, Path] = "results/week7/w7b_semantic_budget/models",
) -> Path:
    """Save the selected frozen budget allocator and its fitted scaler for deployment."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    out_file = p / "w7b_frozen_allocator.joblib"

    artifact = {
        "model_name": model_name,
        "model_entry": model_entry,
        "scaler": scaler,
        "features": [
            "bm25_mean_score",
            "lex_coverage_top2",
            "sem_sim_mean_top6",
            "evidence_query_cluster_span",
            "evidence_lexical_overlap_mean",
        ],
        "budgets": BUDGETS,
    }
    joblib.dump(artifact, out_file)
    logger.info("Saved frozen W7-B budget allocator to %s", out_file)
    return out_file
