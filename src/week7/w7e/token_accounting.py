"""Token accounting and compression statistics for W7-E multi-fidelity representations.

Computes:
- Passage-level and query-level token counts using generator tokenizer.
- Extractive compression ratio: CR_i = tokens(MEDIUM_i) / tokens(FULL_i).
- Token reduction percentage: (1 - CR_i) * 100%.
- Distribution metrics (mean, median, p25, p75, p90, max).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_medium_compression_ratio(medium_tokens: int, original_tokens: int) -> float:
    """Compute extractive compression ratio CR_i = tokens(MEDIUM_i) / max(1, tokens(FULL_i))."""
    if original_tokens <= 0:
        return 0.0
    return float(medium_tokens / original_tokens)


def compute_compression_statistics(
    df_medium_records: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Compute summary statistics for MEDIUM compression across all extracted passages.

    Expects columns:
    - original_tokens
    - medium_tokens
    - compression_ratio
    - token_cap
    """
    if df_medium_records.empty:
        return {}, pd.DataFrame()

    caps = sorted(df_medium_records["token_cap"].unique())
    summary_rows: List[Dict[str, Any]] = []

    for cap in caps:
        sub = df_medium_records[df_medium_records["token_cap"] == cap]
        orig_arr = sub["original_tokens"].values.astype(float)
        med_arr = sub["medium_tokens"].values.astype(float)
        cr_arr = sub["compression_ratio"].values.astype(float)

        summary_rows.append({
            "token_cap": int(cap),
            "n_passages": len(sub),
            "orig_tokens_mean": float(np.mean(orig_arr)),
            "orig_tokens_median": float(np.median(orig_arr)),
            "medium_tokens_mean": float(np.mean(med_arr)),
            "medium_tokens_median": float(np.median(med_arr)),
            "medium_tokens_p90": float(np.percentile(med_arr, 90)),
            "medium_tokens_max": float(np.max(med_arr)),
            "cr_mean": float(np.mean(cr_arr)),
            "cr_median": float(np.median(cr_arr)),
            "cr_p25": float(np.percentile(cr_arr, 25)),
            "cr_p75": float(np.percentile(cr_arr, 75)),
            "cr_p90": float(np.percentile(cr_arr, 90)),
            "token_reduction_pct_mean": float((1.0 - np.mean(cr_arr)) * 100.0),
        })

    df_summary = pd.DataFrame(summary_rows)
    stats_dict = {f"M{r['token_cap']}": r for r in summary_rows}
    return stats_dict, df_summary
