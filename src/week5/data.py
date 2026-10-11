"""Data loading, joins, feature normalization, and dataset construction for Week 5 Content-Aware Allocation.

Implements:
- Strict split isolation (TRAIN=2,088, DEV=231, TEST never touched)
- Joins across supervision, retrieval, response matrix, and W4 pre-generation features
- TRAIN-only feature scaler fitting and transform application to DEV
- Bijections for 5-class allocation budget mapping: k in {2, 4, 5, 6, 8} <-> class index 0..4
- Multi-output regret target vector construction [regret_k2, ..., regret_k8]
- Pairwise query-passage tokenization with strict query preservation (truncation='only_second')
- Truncation diagnostics and reporting
- PyTorch Dataset and DataLoader constructors for conditions I1, I2, I3, I4
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

K_CLASSES: List[int] = [2, 4, 5, 6, 8]
K_TO_CLASS: Dict[int, int] = {2: 0, 4: 1, 5: 2, 6: 3, 8: 4}
CLASS_TO_K: Dict[int, int] = {0: 2, 1: 4, 2: 5, 3: 6, 4: 8}
NUM_CLASSES: int = len(K_CLASSES)

FEATURE_COLS: List[str] = [
    "rho_1",
    "delta_12",
    "entropy",
    "entropy_normalized",
    "query_token_length",
    "n_high",
]


def extract_w4_features_from_retrieval(record: Dict[str, Any]) -> Dict[str, float]:
    """Extract canonical 6 pre-generation retrieval features from a retrieval record."""
    passages = record.get("passages", [])
    scores = [float(p.get("score", 0.0)) for p in passages]
    if len(scores) < 10:
        scores = scores + [0.0] * (10 - len(scores))
    scores = sorted(scores[:10], reverse=True)
    s_arr = np.array(scores, dtype=float)
    s_sum = float(s_arr.sum())
    s_sum_safe = s_sum if s_sum > 0 else 1.0

    s1, s2 = s_arr[0], s_arr[1]
    rho_1 = float(s1 / s_sum_safe)
    delta_12 = float((s1 - s2) / s1) if s1 > 0 else 0.0

    # Softmax entropy
    exps = np.exp(s_arr - np.max(s_arr))
    probs = exps / np.sum(exps)
    ent = -float(np.sum(probs * np.log(probs + 1e-12)))
    ent_norm = float(ent / np.log(len(s_arr)))

    query_text = str(record.get("question", "")).strip()
    query_len = len(query_text.split())
    n_high = int(np.sum(s_arr >= 0.5 * s1)) if s1 > 0 else 1

    return {
        "rho_1": rho_1,
        "delta_12": delta_12,
        "entropy": ent,
        "entropy_normalized": ent_norm,
        "query_token_length": float(query_len),
        "n_high": float(n_high),
    }


def load_retrieval_records(path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Load JSONL retrieval records indexed by string question_id."""
    records_by_qid: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                qid = str(rec["question_id"])
                records_by_qid[qid] = rec
    return records_by_qid


def audit_splits_and_leakage(train_qids: List[str], dev_qids: List[str]) -> None:
    """Verify split sizes and zero overlap across splits."""
    set_train = set(train_qids)
    set_dev = set(dev_qids)
    overlap = set_train.intersection(set_dev)
    if len(overlap) > 0:
        raise ValueError(f"DATA LEAKAGE: {len(overlap)} question_ids exist in both TRAIN and DEV!")
    assert len(train_qids) == 2088, f"Expected 2,088 TRAIN items, got {len(train_qids)}"
    assert len(dev_qids) == 231, f"Expected 231 DEV items, got {len(dev_qids)}"
    logger.info("Split isolation audit PASSED (TRAIN=2,088, DEV=231, overlap=0).")


def build_w5_dataset_bundle(
    train_sup_path: Union[str, Path] = "results/week5/supervision/train_allocation_supervision.csv",
    dev_sup_path: Union[str, Path] = "results/week5/supervision/dev_allocation_supervision.csv",
    train_ret_path: Union[str, Path] = "results/week5/train_response_surface/train_retrieval.jsonl",
    dev_ret_path: Union[str, Path] = "results/week2/raw/dev_retrieval.jsonl",
    train_mat_path: Union[str, Path] = "results/week5/train_response_surface/train_fixed_k_matrix.csv",
    dev_mat_path: Union[str, Path] = "results/week2/raw/dev_fixed_k_matrix.jsonl",
    dev_feat_path: Optional[Union[str, Path]] = "results/week2/processed/dev_retrieval_features.jsonl",
) -> Tuple[Dict[str, Any], Dict[str, Any], StandardScaler]:
    """Load, join, normalize features, and produce complete TRAIN and DEV dataset bundles.

    Returns:
        (train_bundle, dev_bundle, fitted_scaler)
    """
    df_train_sup = pd.read_csv(train_sup_path)
    df_dev_sup = pd.read_csv(dev_sup_path)

    # Standardize string question IDs
    df_train_sup["question_id"] = df_train_sup["question_id"].astype(str)
    df_dev_sup["question_id"] = df_dev_sup["question_id"].astype(str)

    train_ret = load_retrieval_records(train_ret_path)
    dev_ret = load_retrieval_records(dev_ret_path)

    audit_splits_and_leakage(
        df_train_sup["question_id"].tolist(),
        df_dev_sup["question_id"].tolist(),
    )

    # 1. Extract W4 Features
    # For TRAIN: extract deterministically from train retrieval
    train_feats = []
    for qid in df_train_sup["question_id"]:
        assert qid in train_ret, f"Missing TRAIN retrieval record for {qid}"
        f_dict = extract_w4_features_from_retrieval(train_ret[qid])
        f_vec = [f_dict[c] for c in FEATURE_COLS]
        train_feats.append(f_vec)
    train_feats_arr = np.array(train_feats, dtype=np.float32)

    # For DEV: load from frozen dev_retrieval_features if available, else extract
    if dev_feat_path and Path(dev_feat_path).exists():
        df_dev_feats = pd.read_json(dev_feat_path, lines=True)
        df_dev_feats["question_id"] = df_dev_feats["question_id"].astype(str)
        dev_feats_dict = {
            row["question_id"]: [float(row[c]) for c in FEATURE_COLS]
            for _, row in df_dev_feats.iterrows()
        }
        dev_feats = [dev_feats_dict[qid] for qid in df_dev_sup["question_id"]]
        dev_feats_arr = np.array(dev_feats, dtype=np.float32)
    else:
        dev_feats = []
        for qid in df_dev_sup["question_id"]:
            f_dict = extract_w4_features_from_retrieval(dev_ret[qid])
            f_vec = [f_dict[c] for c in FEATURE_COLS]
            dev_feats.append(f_vec)
        dev_feats_arr = np.array(dev_feats, dtype=np.float32)

    # 2. Strict TRAIN-only Scaler Fitting
    scaler = StandardScaler()
    train_feats_scaled = scaler.fit_transform(train_feats_arr)
    dev_feats_scaled = scaler.transform(dev_feats_arr)

    # 3. Targets: T1 (Class index) & T2 (Regret vector)
    train_t1 = np.array([K_TO_CLASS[int(k)] for k in df_train_sup["oracle_k_eps001"]], dtype=np.int64)
    dev_t1 = np.array([K_TO_CLASS[int(k)] for k in df_dev_sup["oracle_k_eps001"]], dtype=np.int64)

    regret_cols = [f"regret_k{k}" for k in K_CLASSES]
    train_t2 = df_train_sup[regret_cols].values.astype(np.float32)
    dev_t2 = df_dev_sup[regret_cols].values.astype(np.float32)

    # 4. Context token counts and routed F1 matrices for downstream policy lookup
    train_mat = pd.read_csv(train_mat_path)
    train_mat["question_id"] = train_mat["question_id"].astype(str)
    train_mat_map = {row["question_id"]: row for _, row in train_mat.iterrows()}

    # DEV matrix
    dev_tokens_map: Dict[Tuple[str, int], int] = {}
    dev_f1_map: Dict[Tuple[str, int], float] = {}
    with open(dev_mat_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                qid = str(d["question_id"])
                kval = int(d["k"])
                dev_tokens_map[(qid, kval)] = int(d.get("context_tokens", 0))
                dev_f1_map[(qid, kval)] = float(d.get("token_f1", 0.0))

    # Assemble Bundles
    train_bundle = {
        "split": "train",
        "question_ids": df_train_sup["question_id"].tolist(),
        "paper_ids": df_train_sup["paper_id"].astype(str).tolist(),
        "questions": [train_ret[qid]["question"] for qid in df_train_sup["question_id"]],
        "passages": [train_ret[qid]["passages"][:10] for qid in df_train_sup["question_id"]],
        "features": train_feats_scaled,
        "features_raw": train_feats_arr,
        "t1_labels": train_t1,
        "t2_regrets": train_t2,
        "supervision_df": df_train_sup,
        "train_mat_map": train_mat_map,
    }

    dev_bundle = {
        "split": "dev",
        "question_ids": df_dev_sup["question_id"].tolist(),
        "paper_ids": df_dev_sup["paper_id"].astype(str).tolist(),
        "questions": [dev_ret[qid]["question"] for qid in df_dev_sup["question_id"]],
        "passages": [dev_ret[qid]["passages"][:10] for qid in df_dev_sup["question_id"]],
        "features": dev_feats_scaled,
        "features_raw": dev_feats_arr,
        "t1_labels": dev_t1,
        "t2_regrets": dev_t2,
        "supervision_df": df_dev_sup,
        "dev_tokens_map": dev_tokens_map,
        "dev_f1_map": dev_f1_map,
    }

    return train_bundle, dev_bundle, scaler


class ContentAllocatorDataset(Dataset):
    """PyTorch Dataset supporting conditions I1, I2, I3, I4 with target T1 or T2."""

    def __init__(
        self,
        bundle: Dict[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        condition: str = "I3",  # I1, I2, I3, I4
        max_seq_length: int = 256,
    ):
        assert condition in ["I1", "I2", "I3", "I4"], f"Invalid condition {condition}"
        self.bundle = bundle
        self.tokenizer = tokenizer
        self.condition = condition
        self.max_seq_length = max_seq_length
        self.n_items = len(bundle["question_ids"])

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        qid = self.bundle["question_ids"][idx]
        pid = self.bundle["paper_ids"][idx]
        q_text = self.bundle["questions"][idx]
        passages = self.bundle["passages"][idx]
        feat = self.bundle["features"][idx]
        t1_label = self.bundle["t1_labels"][idx]
        t2_regret = self.bundle["t2_regrets"][idx]

        item: Dict[str, Any] = {
            "question_id": qid,
            "paper_id": pid,
            "t1_label": torch.tensor(t1_label, dtype=torch.long),
            "t2_regret": torch.tensor(t2_regret, dtype=torch.float32),
            "condition": self.condition,
        }

        # Tabular features (for I2, I4)
        if self.condition in ["I2", "I4"]:
            item["features"] = torch.tensor(feat, dtype=torch.float32)
        else:
            item["features"] = torch.zeros(len(FEATURE_COLS), dtype=torch.float32)

        # Tokenization
        if self.condition in ["I1", "I2"]:
            # Query-only tokenization
            q_enc = self.tokenizer(
                q_text,
                max_length=self.max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            item["query_input_ids"] = q_enc["input_ids"].squeeze(0)
            item["query_attention_mask"] = q_enc["attention_mask"].squeeze(0)

        if self.condition in ["I3", "I4"]:
            # Query-passage pairs for all 10 passages
            # Preserve query completely; truncate only passage ('only_second')
            passage_texts = [p.get("text", "") for p in passages]
            while len(passage_texts) < 10:
                passage_texts.append("")
            passage_texts = passage_texts[:10]

            queries_repeated = [q_text] * 10
            pairs_enc = self.tokenizer(
                queries_repeated,
                passage_texts,
                max_length=self.max_seq_length,
                truncation="only_second",
                padding="max_length",
                return_tensors="pt",
            )
            item["passage_input_ids"] = pairs_enc["input_ids"]  # (10, max_seq_length)
            item["passage_attention_mask"] = pairs_enc["attention_mask"]  # (10, max_seq_length)

            # Passage existence mask: True if passage has non-empty text
            p_mask = torch.tensor([len(t.strip()) > 0 for t in passage_texts], dtype=torch.bool)
            if not p_mask.any():
                p_mask[0] = True  # Avoid all-zero padding mask
            item["passage_mask"] = p_mask  # (10,)

        return item


def compute_truncation_statistics(
    bundle: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 256,
) -> Dict[str, float]:
    """Compute exact truncation diagnostics across all query-passage pairs."""
    total_pairs = 0
    truncated_pairs = 0
    original_lengths: List[int] = []
    retained_lengths: List[int] = []

    for q_text, passages in zip(bundle["questions"], bundle["passages"]):
        for p in passages[:10]:
            p_text = p.get("text", "")
            if not p_text.strip():
                continue
            total_pairs += 1

            # Length without truncation
            q_toks = len(tokenizer.tokenize(q_text))
            p_toks = len(tokenizer.tokenize(p_text))
            raw_len = q_toks + p_toks + 3  # [CLS], [SEP], [SEP]
            original_lengths.append(raw_len)

            if raw_len > max_seq_length:
                truncated_pairs += 1
                retained_lengths.append(max_seq_length)
            else:
                retained_lengths.append(raw_len)

    pct_truncated = (truncated_pairs / total_pairs * 100.0) if total_pairs > 0 else 0.0
    mean_orig = float(np.mean(original_lengths)) if original_lengths else 0.0
    mean_retained = float(np.mean(retained_lengths)) if retained_lengths else 0.0

    return {
        "total_query_passage_pairs": total_pairs,
        "truncated_pairs_count": truncated_pairs,
        "pct_truncated": round(pct_truncated, 2),
        "mean_original_tokens": round(mean_orig, 1),
        "mean_retained_tokens": round(mean_retained, 1),
        "max_seq_length": max_seq_length,
    }


def compute_train_class_weights(train_t1_labels: np.ndarray) -> torch.Tensor:
    """Compute balanced class weights strictly from TRAIN distribution."""
    counts = np.bincount(train_t1_labels, minlength=NUM_CLASSES)
    total = len(train_t1_labels)
    # Inverse class frequency weighting: total / (num_classes * count)
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    for i, c in enumerate(counts):
        weights[i] = total / (NUM_CLASSES * c) if c > 0 else 1.0
    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)
