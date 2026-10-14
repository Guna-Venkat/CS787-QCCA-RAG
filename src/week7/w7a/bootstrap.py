"""Paper-clustered bootstrap hypothesis testing for Week 7-A.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.week6.evaluation import paper_clustered_bootstrap_comparison

logger = logging.getLogger(__name__)


def run_w7a_bootstrap_analysis(
    paper_ids: Sequence[str],
    f1_a: Sequence[float],
    f1_b: Sequence[float],
    name_a: str,
    name_b: str,
    tokens_a: Optional[Sequence[float]] = None,
    tokens_b: Optional[Sequence[float]] = None,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute paper-clustered bootstrap comparison for both F1 and Context Tokens."""
    res_f1 = paper_clustered_bootstrap_comparison(
        paper_ids=list(paper_ids),
        f1_a=list(f1_a),
        f1_b=list(f1_b),
        name_a=name_a,
        name_b=name_b,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )

    out: Dict[str, Any] = {
        "comparison": res_f1["comparison"],
        "policy_A": res_f1["policy_A"],
        "policy_B": res_f1["policy_B"],
        "delta_f1": res_f1["delta_f1"],
        "f1_ci_lower": res_f1["ci_lower"],
        "f1_ci_upper": res_f1["ci_upper"],
        "p_value": res_f1["p_value"],
        "significant": res_f1["significant"],
    }

    if tokens_a is not None and tokens_b is not None:
        res_tok = paper_clustered_bootstrap_comparison(
            paper_ids=list(paper_ids),
            f1_a=list(tokens_a),
            f1_b=list(tokens_b),
            name_a=name_a,
            name_b=name_b,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            seed=seed,
        )
        out["delta_tokens"] = res_tok["delta_f1"]
        out["tokens_ci_lower"] = res_tok["ci_lower"]
        out["tokens_ci_upper"] = res_tok["ci_upper"]

    return out
