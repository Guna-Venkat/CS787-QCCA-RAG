"""Retrieval feature extraction for QCCA pre-generation statistics.

Calculates pre-generation features from ranked BM25 retrieval scores and query text:
1. Top-1 concentration (rho_1)
2. Top-1 / Top-2 normalized margin (Delta_12)
3. Normalized retrieval entropy (H_norm)
4. Query token length
5. High-score passage count (N_high)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


def compute_retrieval_features(
    scores: Sequence[float],
    query: str = "",
    epsilon: float = 1e-8,
    tau: float = 1.0,
    n_high_threshold_ratio: float = 0.8,
) -> Dict[str, float]:
    """Calculate retrieval distribution features from raw retrieval scores.

    Args:
        scores: Sequence of raw retrieval scores (expected sorted non-increasing).
        query: Query string to compute query token length.
        epsilon: Small numerical constant for division stability.
        tau: Softmax temperature parameter (if softmax entropy is used).
        n_high_threshold_ratio: Score fraction relative to top-1 for N_high.

    Returns:
        Dictionary containing extracted feature values.
    """
    if len(scores) == 0:
        return {
            "rho_1": 0.0,
            "delta_12": 0.0,
            "entropy": 0.0,
            "entropy_normalized": 0.0,
            "query_token_length": len(query.strip().split()) if query else 0,
            "n_high": 0,
        }

    raw_scores = np.array(scores, dtype=np.float64)
    # Ensure non-negative scores: s'_i = max(s_i, 0)
    s_prime = np.maximum(raw_scores, 0.0)
    s_sum = float(np.sum(s_prime))
    n_passages = len(s_prime)
    log_n = np.log(max(n_passages, 2))  # Normalized relative to number of candidates (e.g. log(10))

    # 1. Top-1 concentration: rho_1 = s'_1 / (sum_i s'_i + epsilon)
    s1 = float(s_prime[0])
    rho_1 = float(s1 / (s_sum + epsilon)) if s_sum > 0 else (1.0 / n_passages if n_passages > 0 else 0.0)

    # 2. Top-1 / Top-2 normalized margin: Delta_12 = (s'_1 - s'_2) / (s'_1 + epsilon)
    if n_passages >= 2:
        s2 = float(s_prime[1])
        delta_12 = float((s1 - s2) / (s1 + epsilon))
    else:
        delta_12 = 1.0

    # 3. Normalized retrieval entropy:
    # Handle zero/degenerate cases safely: if all scores are 0, uniform distribution has max entropy
    if s_sum <= epsilon:
        # Uniform degenerate distribution
        p = np.full(n_passages, 1.0 / n_passages, dtype=np.float64)
        entropy = float(-np.sum(p * np.log(p)))
        entropy_normalized = 1.0
    else:
        p = s_prime / (s_sum + epsilon)
        # Avoid log(0)
        p_nonzero = p[p > 0]
        entropy = float(-np.sum(p_nonzero * np.log(p_nonzero)))
        entropy_normalized = float(entropy / log_n) if log_n > 0 else 0.0

    # Clamp entropy_normalized to [0.0, 1.0] for numerical safety
    entropy_normalized = float(np.clip(entropy_normalized, 0.0, 1.0))

    # 4. Query token length
    query_token_length = len(query.strip().split()) if query else 0

    # 5. High-score passage count: N_high = count(s'_i >= ratio * s'_1)
    if s1 > epsilon:
        threshold = n_high_threshold_ratio * s1
        n_high = int(np.sum(s_prime >= threshold))
    else:
        n_high = n_passages

    return {
        "rho_1": round(rho_1, 6),
        "delta_12": round(delta_12, 6),
        "entropy": round(entropy, 6),
        "entropy_normalized": round(entropy_normalized, 6),
        "query_token_length": int(query_token_length),
        "n_high": int(n_high),
    }


def extract_features_from_retrieval_records(
    retrieval_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    epsilon: float = 1e-8,
    n_high_threshold_ratio: float = 0.8,
) -> List[Dict[str, Any]]:
    """Extract retrieval features from a dev_retrieval.jsonl file.

    Args:
        retrieval_file: Path to JSONL file containing retrieved passages per question.
        output_file: Optional path to save extracted features as JSONL.
        epsilon: Small numerical constant.
        n_high_threshold_ratio: Fraction of top-1 score for N_high.

    Returns:
        List of feature dictionaries per question.
    """
    retrieval_path = Path(retrieval_file)
    if not retrieval_path.exists():
        raise FileNotFoundError(f"Retrieval file not found: {retrieval_path}")

    records = []
    with open(retrieval_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            question_id = item["question_id"]
            paper_id = item.get("paper_id", "")
            query = item.get("question", "")
            scores = [p["score"] for p in item.get("passages", [])]

            feats = compute_retrieval_features(
                scores=scores,
                query=query,
                epsilon=epsilon,
                n_high_threshold_ratio=n_high_threshold_ratio,
            )
            record = {
                "question_id": question_id,
                "paper_id": paper_id,
                **feats,
            }
            records.append(record)

    if output_file is not None:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    return records
