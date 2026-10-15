"""Data loading, split isolation, and integrity checks for Week 7-B.

Enforces:
- Budgets: K = {2, 4, 5, 6, 8}.
- Strict split isolation: TRAIN = 2,088 queries, DEV = 231 queries.
- Zero paper overlap between TRAIN-FIT and INTERNAL-VAL.
- TEST split is strictly disabled and throws an error if requested.
- Preservation of original BM25 top-10 candidate pool per query.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALLOWED_BUDGETS: Set[int] = {2, 4, 5, 6, 8}
BUDGET_LIST: List[int] = [2, 4, 5, 6, 8]


def assert_test_disabled(split_name: str) -> None:
    """Ensure TEST split is strictly inaccessible."""
    if "test" in str(split_name).lower():
        raise RuntimeError("TEST split access is strictly prohibited in Week 7-B! TEST must remain untouched.")


def load_retrieval_records(path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Load JSONL retrieval records indexed by clean string question_id."""
    records_by_qid: Dict[str, Dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Retrieval file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                qid = str(rec["question_id"]).strip()
                records_by_qid[qid] = rec
    return records_by_qid


def load_w7b_dataset_bundle(
    retrieval_path: Union[str, Path],
    split: str = "train",
) -> Dict[str, Any]:
    """Load unified dataset bundle for TRAIN or DEV.

    Args:
        retrieval_path: Path to train_retrieval.jsonl or dev_retrieval.jsonl.
        split: 'train' or 'dev'. Throws error if 'test'.

    Returns:
        Dictionary containing aligned questions, passages, gold answers, and IDs.
    """
    assert_test_disabled(split)
    ret_records = load_retrieval_records(retrieval_path)

    qids: List[str] = []
    pids: List[str] = []
    questions: List[str] = []
    passages_list: List[List[Dict[str, Any]]] = []
    gold_answers_list: List[List[str]] = []
    gold_reformatted_list: List[List[str]] = []
    bm25_scores_list: List[List[float]] = []
    raw_records_list: List[Dict[str, Any]] = []

    for qid, rec in ret_records.items():
        qids.append(qid)
        pids.append(str(rec.get("paper_id", "")).strip())
        questions.append(str(rec.get("question", "")).strip())

        passages = rec.get("passages", [])[:10]
        if len(passages) < 10:
            # Pad with empty passages if fewer than 10
            for pad_idx in range(len(passages), 10):
                passages.append({"passage_id": f"pad_{pad_idx}", "text": "", "score": 0.0})
        passages_list.append(passages)

        g_ans = rec.get("gold_answers", [])
        g_ref = rec.get("gold_answers_reformatted", g_ans)
        gold_answers_list.append(g_ans)
        gold_reformatted_list.append(g_ref)

        scores = [float(p.get("score", 0.0)) for p in passages]
        bm25_scores_list.append(scores)
        raw_records_list.append(rec)

    n_queries = len(qids)
    if split.lower() == "dev":
        assert n_queries == 231, f"Expected 231 DEV queries, got {n_queries}"
    elif split.lower() == "train":
        assert n_queries == 2088, f"Expected 2,088 TRAIN queries, got {n_queries}"

    return {
        "split": split,
        "question_ids": qids,
        "paper_ids": pids,
        "questions": questions,
        "passages": passages_list,
        "gold_answers": gold_answers_list,
        "gold_answers_reformatted": gold_reformatted_list,
        "bm25_scores": bm25_scores_list,
        "raw_records": raw_records_list,
        "n_queries": n_queries,
        "n_papers": len(set(pids)),
    }


def load_w7b_fit_val_splits(
    train_fit_manifest_path: Union[str, Path] = "results/week5/content_allocator/w5b/splits/train_fit_manifest.csv",
    internal_val_manifest_path: Union[str, Path] = "results/week5/content_allocator/w5b/splits/internal_val_manifest.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load canonical paper-disjoint TRAIN-FIT and INTERNAL-VAL split manifests.

    Asserts zero query and paper leakage.
    """
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
