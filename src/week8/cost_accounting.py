"""Comprehensive cost and compute accounting for multi-pass DAFiR systems.

Tracks and reports:
1. Final-context tokens (context tokens used in the final answering generation).
2. Cumulative prompt tokens (sum of prompt tokens across all generation passes).
3. Generated tokens (sum of generated tokens across all generation passes).
4. End-to-end latency (sum of preprocessing, routing, and generation latencies).
5. Number of generation passes (1 for draft safe-stop, 2 for DAFiR-1, up to 3 for DAFiR-2).
6. GPU peak memory consumption.
7. Promotion rate (fraction of queries with >= 1 promotion).
8. Mean number of promotions per query.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass
class QueryCostRecord:
    """Detailed cost record for a single query execution."""

    question_id: str
    paper_id: str
    policy_name: str
    num_passes: int
    promoted_indices: List[int] = field(default_factory=list)
    final_context_tokens: int = 0
    cumulative_prompt_tokens: int = 0
    total_generated_tokens: int = 0
    e2e_latency_sec: float = 0.0
    gpu_peak_memory_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_cost_records(
    records: Sequence[Union[Dict[str, Any], QueryCostRecord]],
) -> Dict[str, float]:
    """Compute aggregate cost metrics over a collection of query cost records."""
    if not records:
        return {
            "mean_final_context_tokens": 0.0,
            "mean_cumulative_prompt_tokens": 0.0,
            "mean_generated_tokens": 0.0,
            "mean_e2e_latency_sec": 0.0,
            "mean_num_passes": 0.0,
            "promotion_rate": 0.0,
            "mean_num_promotions": 0.0,
            "gpu_peak_memory_mb": 0.0,
        }

    dict_records = [r.to_dict() if isinstance(r, QueryCostRecord) else r for r in records]
    df = pd.DataFrame(dict_records)

    final_ctx = float(df["final_context_tokens"].mean())
    cum_prompt = float(df["cumulative_prompt_tokens"].mean())
    gen_tokens = float(df.get("total_generated_tokens", df.get("generated_tokens", 0)).mean())
    latency = float(df["e2e_latency_sec"].mean())
    passes = float(df["num_passes"].mean())

    prom_counts = df["promoted_indices"].apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 0)
    promotion_rate = float((prom_counts > 0).mean())
    mean_promotions = float(prom_counts.mean())
    gpu_peak = float(df["gpu_peak_memory_mb"].max()) if "gpu_peak_memory_mb" in df.columns else 0.0

    return {
        "mean_final_context_tokens": final_ctx,
        "mean_cumulative_prompt_tokens": cum_prompt,
        "mean_generated_tokens": gen_tokens,
        "mean_e2e_latency_sec": latency,
        "mean_num_passes": passes,
        "promotion_rate": promotion_rate,
        "mean_num_promotions": mean_promotions,
        "gpu_peak_memory_mb": gpu_peak,
    }


class GPUTracker:
    """Context manager for tracking peak GPU VRAM allocation."""

    def __init__(self, device: Union[str, int] = 0):
        self.device = device
        self.peak_mb: float = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            self.peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
