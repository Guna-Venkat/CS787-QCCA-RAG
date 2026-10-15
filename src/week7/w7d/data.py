"""Data loading, split management, input length audit, and truncation for Week 7-D.

Provides:
- Strict TEST isolation guard.
- Strict zero-leakage assertions.
- Loading of paper-disjoint TRAIN-FIT and INTERNAL-VAL manifests.
- Loading of retrieval bundles (query + 10 passages with semantic & BM25 metadata).
- Input length audit across TRAIN and DEV.
- Deterministic per-document truncation (preserves all 10 passages without end-clipping).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALLOWED_BUDGETS: List[int] = [2, 4, 5, 6, 8]
ANCHOR_K: int = 4

FORBIDDEN_FEATURE_SUBSTRINGS: Set[str] = {
    "f1",
    "oracle",
    "regret",
    "answer",
    "gold",
    "target",
    "ground_truth",
}


def assert_test_disabled(split_name: str) -> None:
    """Ensure TEST split is strictly inaccessible."""
    if "test" in str(split_name).lower():
        raise RuntimeError("TEST split access is strictly prohibited in Week 7-D! TEST must remain untouched.")


def assert_zero_feature_leakage(feature_cols: Sequence[str]) -> None:
    """Ensure no downstream target columns leak into feature vectors."""
    for col in feature_cols:
        col_lower = str(col).lower()
        for f in FORBIDDEN_FEATURE_SUBSTRINGS:
            if f in col_lower:
                raise ValueError(f"LEAKAGE DETECTED: Column '{col}' contains forbidden target keyword '{f}'!")


def load_w7d_splits(
    train_fit_manifest_path: Union[str, Path] = "results/week5/content_allocator/w5b/splits/train_fit_manifest.csv",
    internal_val_manifest_path: Union[str, Path] = "results/week5/content_allocator/w5b/splits/internal_val_manifest.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load canonical paper-disjoint TRAIN-FIT and INTERNAL-VAL split manifests."""
    df_fit = pd.read_csv(train_fit_manifest_path)
    df_val = pd.read_csv(internal_val_manifest_path)

    df_fit["question_id"] = df_fit["question_id"].astype(str).str.strip()
    df_val["question_id"] = df_val["question_id"].astype(str).str.strip()
    df_fit["paper_id"] = df_fit["paper_id"].astype(str).str.strip()
    df_val["paper_id"] = df_val["paper_id"].astype(str).str.strip()

    fit_qids = set(df_fit["question_id"])
    val_qids = set(df_val["question_id"])
    overlap_qids = fit_qids.intersection(val_qids)
    if overlap_qids:
        raise ValueError(f"DATA LEAKAGE: {len(overlap_qids)} questions exist in both TRAIN-FIT and INTERNAL-VAL!")

    fit_papers = set(df_fit["paper_id"])
    val_papers = set(df_val["paper_id"])
    overlap_papers = fit_papers.intersection(val_papers)
    if overlap_papers:
        raise ValueError(f"DATA LEAKAGE: {len(overlap_papers)} papers exist in both TRAIN-FIT and INTERNAL-VAL!")

    logger.info(
        "Validated paper-disjoint split: TRAIN-FIT=%d questions (%d papers), INTERNAL-VAL=%d questions (%d papers).",
        len(df_fit), len(fit_papers), len(df_val), len(val_papers),
    )
    return df_fit, df_val


def load_retrieval_passages(
    retrieval_path: Union[str, Path],
    split_name: str = "train",
    max_passages: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Load query and retrieved passages from JSONL file.

    Returns mapping:
        question_id -> {
            "question_id": str,
            "paper_id": str,
            "question": str,
            "passages": List[Dict[str, Any]] (len <= max_passages, with rank, score, text, passage_id)
        }
    """
    assert_test_disabled(split_name)
    path = Path(retrieval_path)
    if not path.is_file():
        raise FileNotFoundError(f"Retrieval file not found at {path}")

    data: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            qid = str(item.get("question_id") or item.get("id")).strip()
            pid = str(item.get("paper_id", "")).strip()
            q_text = str(item.get("question") or item.get("query", "")).strip()

            raw_passages = item.get("passages") or item.get("contexts") or []
            passages: List[Dict[str, Any]] = []
            for rank_idx, p in enumerate(raw_passages[:max_passages]):
                p_text = p.get("text", "") if isinstance(p, dict) else str(p)
                p_id = str(p.get("passage_id", f"p_{rank_idx}")) if isinstance(p, dict) else f"p_{rank_idx}"
                bm25_score = float(p.get("score", 0.0)) if isinstance(p, dict) else 0.0
                bm25_rank = int(p.get("rank", rank_idx + 1)) if isinstance(p, dict) else rank_idx + 1

                passages.append({
                    "passage_id": p_id,
                    "bm25_rank": bm25_rank,
                    "bm25_score": bm25_score,
                    "semantic_rank": rank_idx + 1,  # Semantic ordering
                    "text": p_text,
                    "char_length": len(p_text),
                })

            data[qid] = {
                "question_id": qid,
                "paper_id": pid,
                "question": q_text,
                "passages": passages,
            }

    logger.info("Loaded retrieval passages for %s: %d queries.", split_name, len(data))
    return data


def truncate_document_deterministic(
    text: str,
    max_tokens: int = 256,
    prefix_ratio: float = 0.5,
) -> str:
    """Deterministically truncate a passage preserving both prefix and tail context.

    Takes first prefix_tokens and last suffix_tokens based on whitespace/word tokens
    to guarantee consistent length without dropping key concluding sentences.
    """
    words = text.split()
    if len(words) <= max_tokens:
        return text

    n_prefix = int(max_tokens * prefix_ratio)
    n_suffix = max_tokens - n_prefix

    prefix = " ".join(words[:n_prefix])
    suffix = " ".join(words[-n_suffix:])
    return f"{prefix} ... {suffix}"


def audit_input_lengths(
    train_data: Dict[str, Dict[str, Any]],
    dev_data: Dict[str, Dict[str, Any]],
    max_passages: int = 10,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Audit token and word length statistics for D1 + ... + D10 + q across splits.

    Ensures we know the distribution before applying sequence caps so no passages
    are accidentally wiped out.
    """
    results: List[Dict[str, Any]] = []

    for split_name, dataset in [("TRAIN", train_data), ("DEV", dev_data)]:
        q_lens: List[int] = []
        doc_lens: List[int] = []
        total_lens: List[int] = []

        for qid, item in dataset.items():
            q_words = len(item["question"].split())
            q_lens.append(q_words)

            passages = item["passages"][:max_passages]
            p_words = [len(p["text"].split()) for p in passages]
            doc_lens.extend(p_words)
            total_lens.append(q_words + sum(p_words))

        tot_arr = np.array(total_lens, dtype=float)
        p_arr = np.array(doc_lens, dtype=float)

        results.append({
            "split": split_name,
            "n_queries": len(dataset),
            "total_words_median": float(np.median(tot_arr)),
            "total_words_p75": float(np.percentile(tot_arr, 75)),
            "total_words_p90": float(np.percentile(tot_arr, 90)),
            "total_words_p95": float(np.percentile(tot_arr, 95)),
            "total_words_max": float(np.max(tot_arr)),
            "doc_words_mean": float(np.mean(p_arr)),
            "doc_words_p90": float(np.percentile(p_arr, 90)),
            "doc_words_max": float(np.max(p_arr)),
        })

    df_table = pd.DataFrame(results)
    stats_dict = {r["split"]: r for r in results}
    return stats_dict, df_table
