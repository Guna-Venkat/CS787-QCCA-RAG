"""Data loading, split management, and integrity validation for Week 8 DAFiR.

Strict Guarantees:
- TEST split is strictly disabled and inaccessible.
- FIT (1746 queries) and INTERNAL-VAL (342 queries) are strictly paper-disjoint.
- DEV (231 queries) is in-domain held-out evaluation only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load Week 8 configuration with path resolution."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs/week8/w8_dafir.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg.get("TEST_DISABLED", True) is True, "CRITICAL ERROR: TEST split must be disabled!"
    return cfg


def assert_paper_disjoint(papers_a: Set[str], papers_b: Set[str]) -> None:
    """Assert two sets of paper_ids are disjoint."""
    overlap = set(papers_a) & set(papers_b)
    if overlap:
        raise ValueError(f"Paper overlap detected between sets: {overlap}")


def get_week8_manifest_path(split: str, cfg: Optional[Dict[str, Any]] = None) -> Path:
    """Return filesystem path to split manifest."""
    if cfg is None:
        cfg = load_config()
    if split in ("fit", "train_fit"):
        return PROJECT_ROOT / cfg["splits"]["fit_manifest"]
    elif split in ("internal_val", "val"):
        return PROJECT_ROOT / cfg["splits"]["internal_val_manifest"]
    else:
        raise ValueError(f"No manifest file for split: {split}")


def load_week8_split(
    split: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor], pd.DataFrame]:
    """Load query records, SFR embeddings, and manifest for a given split.

    Args:
        split: 'fit', 'internal_val', or 'dev'.
        cfg: Week 8 configuration dictionary.

    Returns:
        (records, sfr_embeddings_dict, manifest_df)
    """
    if cfg is None:
        cfg = load_config()

    split_lower = split.lower().strip()
    if split_lower == "test":
        raise PermissionError("TEST split is strictly forbidden in Week 8!")

    if split_lower in ("fit", "train_fit"):
        manifest_path = PROJECT_ROOT / cfg["splits"]["fit_manifest"]
        manifest_df = pd.read_csv(manifest_path)
        allowed_qids = set(manifest_df["question_id"].astype(str))

        retrieval_path = PROJECT_ROOT / cfg["splits"]["train_retrieval"]
        sfr_path = PROJECT_ROOT / cfg["splits"]["train_sfr_embeddings"]

    elif split_lower in ("internal_val", "val"):
        manifest_path = PROJECT_ROOT / cfg["splits"]["internal_val_manifest"]
        manifest_df = pd.read_csv(manifest_path)
        allowed_qids = set(manifest_df["question_id"].astype(str))

        retrieval_path = PROJECT_ROOT / cfg["splits"]["train_retrieval"]
        sfr_path = PROJECT_ROOT / cfg["splits"]["train_sfr_embeddings"]

    elif split_lower == "dev":
        manifest_df = pd.DataFrame()
        retrieval_path = PROJECT_ROOT / cfg["splits"]["dev_retrieval"]
        sfr_path = PROJECT_ROOT / cfg["splits"]["dev_sfr_embeddings"]
        allowed_qids = None
    else:
        raise ValueError(f"Unknown split: {split}. Allowed: 'fit', 'internal_val', 'dev'.")

    # Load retrieval JSONL
    records: List[Dict[str, Any]] = []
    with open(retrieval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                qid = str(item["question_id"])
                if allowed_qids is None or qid in allowed_qids:
                    records.append(item)

    # Load SFR embeddings
    sfr_raw = torch.load(sfr_path, map_location="cpu")
    sfr_dict: Dict[str, torch.Tensor] = (
        sfr_raw["embeddings"] if isinstance(sfr_raw, dict) and "embeddings" in sfr_raw else sfr_raw
    )

    logger.info("Loaded %d queries for split '%s' from %s", len(records), split, retrieval_path)
    return records, sfr_dict, manifest_df


def assert_split_disjointness(
    fit_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    dev_records: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Assert strict question-id and paper-id disjointness between FIT and INTERNAL-VAL."""
    fit_qids = {str(r["question_id"]) for r in fit_records}
    val_qids = {str(r["question_id"]) for r in val_records}

    fit_pids = {str(r.get("paper_id", "")) for r in fit_records if r.get("paper_id")}
    val_pids = {str(r.get("paper_id", "")) for r in val_records if r.get("paper_id")}

    overlap_qids = fit_qids & val_qids
    assert not overlap_qids, f"Data leakage error: {len(overlap_qids)} questions overlap between FIT and INTERNAL-VAL!"

    overlap_pids = fit_pids & val_pids
    assert not overlap_pids, f"Data leakage error: {len(overlap_pids)} papers overlap between FIT and INTERNAL-VAL!"

    if dev_records is not None:
        dev_qids = {str(r["question_id"]) for r in dev_records}
        assert not (fit_qids & dev_qids), "Data leakage: FIT overlaps with DEV!"
        assert not (val_qids & dev_qids), "Data leakage: INTERNAL-VAL overlaps with DEV!"
