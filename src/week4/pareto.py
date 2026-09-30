"""Pareto frontier calculation and dominance identification for Week 4."""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_pareto_frontier(df_points: pd.DataFrame) -> pd.DataFrame:
    """Identify Pareto non-dominated points in (Context Tokens, Token F1) space.
    
    Objectives:
    - Context tokens (X): MINIMIZE (lower is better)
    - Token F1 (Y): MAXIMIZE (higher is better)
    """
    df = df_points.copy()
    is_non_dominated = []
    
    for i, row_a in df.iterrows():
        xa, ya = row_a["mean_context_tokens"], row_a["mean_f1"]
        dominated = False
        for j, row_b in df.iterrows():
            if i == j:
                continue
            xb, yb = row_b["mean_context_tokens"], row_b["mean_f1"]
            # B dominates A if B has <= tokens AND >= F1, with at least one strict inequality
            if (xb <= xa and yb >= ya) and (xb < xa or yb > ya):
                dominated = True
                break
        is_non_dominated.append(not dominated)
        
    df["is_pareto_optimal"] = is_non_dominated
    df = df.sort_values(by="mean_context_tokens").reset_index(drop=True)
    return df
