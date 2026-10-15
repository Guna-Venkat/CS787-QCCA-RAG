"""Target definitions and supervisory label generation for Week 7-D.

Provides:
- Downstream SARA-derived supervisory targets:
    - y_down: 1[F(q, 2) >= F*(q) - epsilon] (Safe downshift to k=2)
    - y_up: 1[max(F5, F6, F8) - F4 > epsilon] (Selective upshift beyond k=4)
    - deltas: Delta_k = F(q, k) - F(q, 4) for k in {5, 6, 8}
    - admissibility_vector: [z_2, z_4, z_5, z_6, z_8]
    - oracle_k: minimum-cost epsilon-admissible budget
    - aux_k_index: integer index in {0, 1, 2, 3, 4} corresponding to oracle_k
- Passage-level utility target infrastructure (for W7-E multi-fidelity handoff).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BUDGETS: List[int] = [2, 4, 5, 6, 8]
BUDGET_TO_INDEX: Dict[int, int] = {2: 0, 4: 1, 5: 2, 6: 3, 8: 4}
INDEX_TO_BUDGET: Dict[int, int] = {0: 2, 1: 4, 2: 5, 3: 6, 4: 8}


def compute_sara_multitask_targets(
    f1_by_k: Dict[int, float],
    tokens_by_k: Optional[Dict[int, float]] = None,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    anchor_k: int = 4,
    epsilon: float = 0.01,
) -> Dict[str, Any]:
    """Compute downstream SARA utility targets for a single query."""
    max_f1 = max(f1_by_k.values())

    # Admissibility vector
    adm_set = {k for k in budgets if f1_by_k[k] >= max_f1 - epsilon}
    z_vector = [1.0 if k in adm_set else 0.0 for k in budgets]

    # Minimum-cost oracle k
    if tokens_by_k:
        oracle_k = min(adm_set, key=lambda k: (tokens_by_k[k], k)) if adm_set else max(budgets)
    else:
        oracle_k = min(adm_set) if adm_set else max(budgets)

    # Safe downshift target (y_down)
    y_down = 1.0 if f1_by_k[2] >= max_f1 - epsilon else 0.0

    # Selective upshift target (y_up)
    f_up_star = max(f1_by_k[5], f1_by_k[6], f1_by_k[8])
    gain_up = f_up_star - f1_by_k[anchor_k]
    y_up = 1.0 if gain_up > epsilon else 0.0

    # Marginal gain deltas relative to k4
    delta_5 = f1_by_k[5] - f1_by_k[anchor_k]
    delta_6 = f1_by_k[6] - f1_by_k[anchor_k]
    delta_8 = f1_by_k[8] - f1_by_k[anchor_k]

    return {
        "y_down": y_down,
        "y_up": y_up,
        "delta_5": delta_5,
        "delta_6": delta_6,
        "delta_8": delta_8,
        "deltas": [delta_5, delta_6, delta_8],
        "admissibility_vector": z_vector,
        "oracle_k": oracle_k,
        "aux_k_index": BUDGET_TO_INDEX.get(oracle_k, 1),
        "max_f1": max_f1,
        "gain_up": gain_up,
    }


def build_w7d_targets_dataframe(
    response_matrix_df: pd.DataFrame,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    anchor_k: int = 4,
    epsilon: float = 0.01,
) -> pd.DataFrame:
    """Build a comprehensive DataFrame of SARA-derived targets for all queries in response matrix."""
    records: List[Dict[str, Any]] = []

    for _, row in response_matrix_df.iterrows():
        qid = str(row["question_id"]).strip()
        pid = str(row["paper_id"]).strip()

        f1_by_k = {k: float(row[f"f1_k{k}"]) for k in budgets}
        tokens_by_k = {k: float(row[f"tokens_k{k}"]) for k in budgets}

        targets = compute_sara_multitask_targets(
            f1_by_k=f1_by_k,
            tokens_by_k=tokens_by_k,
            budgets=budgets,
            anchor_k=anchor_k,
            epsilon=epsilon,
        )

        rec = {
            "question_id": qid,
            "paper_id": pid,
            "target_y_down": targets["y_down"],
            "target_y_up": targets["y_up"],
            "target_delta_5": targets["delta_5"],
            "target_delta_6": targets["delta_6"],
            "target_delta_8": targets["delta_8"],
            "target_z_2": targets["admissibility_vector"][0],
            "target_z_4": targets["admissibility_vector"][1],
            "target_z_5": targets["admissibility_vector"][2],
            "target_z_6": targets["admissibility_vector"][3],
            "target_z_8": targets["admissibility_vector"][4],
            "target_oracle_k": targets["oracle_k"],
            "target_aux_k_index": targets["aux_k_index"],
            "target_max_f1": targets["max_f1"],
            "target_gain_up": targets["gain_up"],
        }
        for k in budgets:
            rec[f"f1_k{k}"] = f1_by_k[k]
            rec[f"tokens_k{k}"] = tokens_by_k[k]

        records.append(rec)

    return pd.DataFrame(records)
