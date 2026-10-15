"""Data loading, candidate-budget expansion, split alignment, and integrity checks for Week 7-C.

Week 7-C Formulates budget selection as candidate-level sufficiency prediction:
    (q, k) -> P(z=1), regret
for candidate budgets K = {2, 4, 5, 6, 8}.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALLOWED_BUDGETS: Set[int] = {2, 4, 5, 6, 8}
BUDGET_LIST: List[int] = [2, 4, 5, 6, 8]


def assert_test_disabled(split_name: str) -> None:
    """Ensure TEST split is strictly inaccessible."""
    if "test" in str(split_name).lower():
        raise RuntimeError("TEST split access is strictly prohibited in Week 7-C! TEST must remain untouched.")


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


def load_w7c_dataset_bundle(
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


def load_w7c_fit_val_splits(
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


def build_candidate_budget_dataframe(
    response_matrix_df: pd.DataFrame,
    budgets: Sequence[int] = (2, 4, 5, 6, 8),
    epsilon: float = 0.01,
    split_name: str = "train",
) -> pd.DataFrame:
    """Expand wide response matrix into candidate-budget rows (q, k).

    For every query q and budget k in budgets:
        F_1(q, k)
        F_1*(q) = max_k F_1(q, k)
        regret(q, k) = F_1*(q) - F_1(q, k)
        z(q, k) = 1[regret(q, k) <= epsilon]
        log_regret(q, k) = log(1 + regret(q, k))

    Args:
        response_matrix_df: Wide DataFrame with columns:
            question_id, paper_id, f1_k2, tokens_k2, ...
        budgets: Candidate budgets (2, 4, 5, 6, 8).
        epsilon: Sufficiency threshold (0.01).
        split_name: Name of split (e.g. 'train', 'dev').

    Returns:
        Long DataFrame with N * len(budgets) rows.
    """
    assert_test_disabled(split_name)
    rows: List[Dict[str, Any]] = []

    for _, qrow in response_matrix_df.iterrows():
        qid = str(qrow["question_id"]).strip()
        pid = str(qrow["paper_id"]).strip()

        # Extract per-budget F1 and tokens
        f1_vals = {k: float(qrow[f"f1_k{k}"]) for k in budgets}
        token_vals = {k: float(qrow[f"tokens_k{k}"]) for k in budgets}

        # Query max F1 across allowed budgets
        max_f1 = max(f1_vals.values())

        # Determine epsilon-oracle budget (smallest k achieving max_f1 - epsilon)
        eps_admissible_k = [k for k in sorted(budgets) if f1_vals[k] >= max_f1 - epsilon]
        oracle_k = eps_admissible_k[0] if eps_admissible_k else max(budgets)

        for k in budgets:
            f1_k = f1_vals[k]
            tokens_k = token_vals[k]
            regret_k = max_f1 - f1_k
            # Clip negative numerical jitter
            regret_k = max(0.0, float(regret_k))
            is_sufficient = 1 if regret_k <= epsilon else 0
            log_regret_k = float(np.log1p(regret_k))

            rows.append({
                "split": split_name,
                "question_id": qid,
                "paper_id": pid,
                "k": int(k),
                "target_f1": f1_k,
                "target_tokens": tokens_k,
                "target_max_f1": max_f1,
                "target_regret": regret_k,
                "target_log_regret": log_regret_k,
                "target_sufficient": is_sufficient,
                "target_oracle_k": oracle_k,
            })

    df_cand = pd.DataFrame(rows)
    expected_rows = len(response_matrix_df) * len(budgets)
    if len(df_cand) != expected_rows:
        raise ValueError(f"Expected {expected_rows} candidate rows, got {len(df_cand)}")

    return df_cand
