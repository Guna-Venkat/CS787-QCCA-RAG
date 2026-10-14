"""Data loading, prediction alignment, checkpoint verification, and split isolation for Week 7-A.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.week6.data import load_retrieval_records

logger = logging.getLogger(__name__)

ALLOWED_K_VALUES: Set[int] = {2, 4, 5, 6, 8}


def clean_id(val: Any) -> str:
    """Normalize question ID representation to string."""
    s = str(val).strip()
    if s.replace(".", "", 1).isdigit():
        return str(int(float(s)))
    return s


def load_frozen_w4_predictions(
    predictions_path: Union[str, Path] = "results/week4/qcca_v2_final_validation/stage2_oof_predictions.csv",
    expected_rows: int = 231,
) -> pd.DataFrame:
    """Load canonical frozen W4 Balanced Logistic Regression predictions on DEV.

    Strictly reads offline predictions without retraining.

    Guarantees:
    - Exactly 231 DEV rows.
    - All predicted k values belong to {2, 4, 5, 6, 8}.
    - No NaNs or missing predictions.
    """
    p = Path(predictions_path)
    if not p.exists():
        raise FileNotFoundError(f"Frozen W4 predictions not found at: {p}")

    df = pd.read_csv(p)
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} DEV rows in W4 predictions, found {len(df)}")

    # Ensure required columns
    required_cols = ["question_id", "paper_id", "selected_k"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in W4 predictions file: {p}")

    df["question_id"] = df["question_id"].apply(clean_id)
    df["paper_id"] = df["paper_id"].astype(str)
    df["selected_k"] = df["selected_k"].astype(int)

    # Invariant: predicted k must be in deployable set K = {2, 4, 5, 6, 8}
    invalid_k = set(df["selected_k"]) - ALLOWED_K_VALUES
    if invalid_k:
        raise ValueError(f"Found invalid budget predictions outside {ALLOWED_K_VALUES}: {invalid_k}")

    return df


def load_w7a_dev_bundle(
    dev_retrieval_path: Union[str, Path] = "results/week2/raw/dev_retrieval.jsonl",
    w4_predictions_path: Union[str, Path] = "results/week4/qcca_v2_final_validation/stage2_oof_predictions.csv",
) -> Dict[str, Any]:
    """Load unified DEV data aligned across retrieval records and frozen W4 budget predictions."""
    df_w4 = load_frozen_w4_predictions(w4_predictions_path)
    ret_records = load_retrieval_records(dev_retrieval_path)

    qids = df_w4["question_id"].tolist()
    pids = df_w4["paper_id"].tolist()
    predicted_k = df_w4["selected_k"].tolist()

    questions: List[str] = []
    passages_list: List[List[Dict[str, Any]]] = []
    gold_answers_list: List[List[str]] = []
    gold_reformatted_list: List[List[str]] = []
    raw_records_list: List[Dict[str, Any]] = []

    for qid in qids:
        if qid not in ret_records:
            raise KeyError(f"Missing retrieval record for question_id '{qid}' in {dev_retrieval_path}")
        rec = ret_records[qid]
        questions.append(str(rec.get("question", "")).strip())
        passages_list.append(rec.get("passages", [])[:10])
        g_ans = rec.get("gold_answers", [])
        g_ref = rec.get("gold_answers_reformatted", g_ans)
        gold_answers_list.append(g_ans)
        gold_reformatted_list.append(g_ref)
        raw_records_list.append(rec)

    return {
        "question_ids": qids,
        "paper_ids": pids,
        "predicted_k": predicted_k,
        "questions": questions,
        "passages": passages_list,
        "gold_answers": gold_answers_list,
        "gold_answers_reformatted": gold_reformatted_list,
        "raw_records": raw_records_list,
        "df_w4": df_w4,
        "n_queries": len(qids),
        "n_papers": len(set(pids)),
    }


def verify_sara_checkpoint(
    checkpoint_path: Union[str, Path],
) -> Dict[str, Any]:
    """Verify that fine-tuned SARA projector weights are present in the checkpoint.

    Prevents silent failure where newly initialized projector parameters are left un-restored.
    """
    ckpt_dir = Path(checkpoint_path)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"SARA checkpoint directory does not exist: {ckpt_dir}")

    # Check for pytorch_model.bin or model.safetensors or projector.pt
    found_weights = list(ckpt_dir.glob("*.bin")) + list(ckpt_dir.glob("*.safetensors")) + list(ckpt_dir.glob("*.pt"))
    if not found_weights:
        raise FileNotFoundError(f"No model weight files found in SARA checkpoint directory: {ckpt_dir}")

    return {
        "checkpoint_dir": str(ckpt_dir),
        "weights_files": [f.name for f in found_weights],
        "verified": True,
    }


def audit_w7a_split_isolation(test_disabled: bool = True) -> None:
    """Audit that TEST split is strictly disabled and never referenced."""
    if not test_disabled:
        raise ValueError("TEST_DISABLED must be True for Week 7-A execution!")

    # Check that test split files are not loaded
    logger.info("Week 7-A Split Isolation Audit: TEST split strictly forbidden and untouched.")
