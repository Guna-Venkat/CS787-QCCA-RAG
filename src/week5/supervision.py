"""Supervision data management and oracle target derivation for Week 5.

Provides utilities for:
- Normalizing question IDs
- Loading canonical dev and train supervision tables
- Computing best F1, regret, and epsilon-good indicator vectors
- Deriving minimum-k epsilon oracle targets
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_K_VALUES = [2, 4, 5, 6, 8]
CONTEXT_TOKEN_COSTS = {
    2: 534.5,
    4: 946.1,
    5: 1157.0,
    6: 1359.1,
    8: 1758.5,
}


def normalize_id(val: Any) -> str:
    """Normalize question/paper IDs to clean strings without trailing .0."""
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def derive_canonical_supervision_record(
    question_id: str,
    paper_id: str,
    f1_by_k: Dict[int, float],
    epsilon: float = 0.01,
) -> Dict[str, Any]:
    """Derive canonical supervision row for a single query from fixed-k response surface."""
    f1s = {k: float(f1_by_k[k]) for k in VALID_K_VALUES}
    best_f1 = max(f1s.values())
    
    regrets = {k: max(0.0, best_f1 - f1s[k]) for k in VALID_K_VALUES}
    eps_goods = {k: int(regrets[k] <= epsilon + 1e-9) for k in VALID_K_VALUES}
    
    feasible_ks = [k for k in VALID_K_VALUES if eps_goods[k] == 1]
    oracle_k = min(feasible_ks) if feasible_ks else 8

    rec = {
        "question_id": normalize_id(question_id),
        "paper_id": normalize_id(paper_id),
        "best_f1": best_f1,
        "oracle_k_eps001": oracle_k,
    }
    for k in VALID_K_VALUES:
        rec[f"f1_k{k}"] = f1s[k]
        rec[f"regret_k{k}"] = regrets[k]
        rec[f"eps_good_k{k}"] = eps_goods[k]

    return rec


def load_dev_supervision(
    supervision_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Load canonical DEV supervision table."""
    path = Path(supervision_path) if supervision_path else REPO_ROOT / "results/week5/supervision/dev_allocation_supervision.csv"
    if not path.exists():
        raise FileNotFoundError(f"DEV supervision table not found at: {path}")
    df = pd.read_csv(path)
    df["question_id"] = df["question_id"].apply(normalize_id)
    df["paper_id"] = df["paper_id"].apply(normalize_id)
    return df
