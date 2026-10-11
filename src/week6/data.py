"""Data processing, evidence mapping, split isolation, and PyTorch dataset construction for Week 6 Passage Routing.

Responsibilities:
- Strict split isolation: TRAIN-FIT (1,746 questions), INTERNAL-VAL (342 questions), DEV (231 questions).
- TEST split is strictly forbidden and never accessed.
- Deterministic QASPER gold evidence matching against BM25 top-10 retrieved passages.
- Query-preserving tokenization (truncation='only_second') with sentence-transformers/all-MiniLM-L6-v2.
- PassageRoutingDataset and collation for pair-level ranking and classification.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.week0.dataset_adapter import QASPERAdapter
from src.week0.dataset_analyzer import extract_content_words

logger = logging.getLogger(__name__)


def match_passages_to_gold_evidence(
    passages: List[Dict[str, Any]],
    gold_evidence_items: List[Dict[str, Any]],
    min_hl_length: int = 15,
    min_overlap_ev: float = 0.3,
    min_overlap_p: float = 0.5,
    min_shared_words: int = 10,
) -> Tuple[np.ndarray, List[str]]:
    """Deterministically match top-10 retrieved passages against QASPER gold evidence annotations.

    Matching hierarchy:
    1. Highlight Substring: Any non-trivial annotated highlighted span (len > min_hl_length)
       appears verbatim (case-insensitive) within the passage text.
    2. Evidence Recall: The content-word overlap covers >= min_overlap_ev of the gold paragraph.
    3. Passage Precision: The content-word overlap covers >= min_overlap_p of the passage with
       at least min_shared_words shared tokens.

    Returns:
        (binary_labels, match_details):
            binary_labels: shape (len(passages),) float32 array with 1.0 for match, 0.0 otherwise.
            match_details: list of descriptive strings for diagnostic reporting.
    """
    flat_ev: List[str] = []
    flat_hl: List[str] = []
    for item in gold_evidence_items:
        flat_ev.extend(item.get("evidence_paragraphs", []))
        flat_hl.extend(item.get("highlighted_spans", []))

    unique_ev = list(dict.fromkeys(flat_ev))
    unique_hl = [h.strip().lower() for h in dict.fromkeys(flat_hl) if len(h.strip()) > min_hl_length]

    labels = np.zeros(len(passages), dtype=np.float32)
    details: List[str] = []

    for idx, p in enumerate(passages):
        p_text = p.get("text", "")
        p_lower = p_text.lower()
        p_words = set(extract_content_words(p_text))

        is_match = False
        match_reason = "none"

        # 1. Highlight substring match
        for hl in unique_hl:
            if hl in p_lower:
                is_match = True
                match_reason = "highlight_substring"
                break

        # 2. Paragraph content-word overlap
        if not is_match and p_words:
            for g_para in unique_ev:
                g_words = set(extract_content_words(g_para))
                if not g_words:
                    continue
                shared = p_words & g_words
                overlap_ev = len(shared) / len(g_words)
                overlap_p = len(shared) / len(p_words)

                if overlap_ev >= min_overlap_ev:
                    is_match = True
                    match_reason = f"evidence_recall_{overlap_ev:.2f}"
                    break
                elif overlap_p >= min_overlap_p and len(shared) >= min_shared_words:
                    is_match = True
                    match_reason = f"passage_precision_{overlap_p:.2f}"
                    break

        if is_match:
            labels[idx] = 1.0
            details.append(f"Rank {idx + 1}: MATCH ({match_reason})")
        else:
            details.append(f"Rank {idx + 1}: NO_MATCH")

    return labels, details


def load_retrieval_records(path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Load JSONL retrieval records indexed by question_id string."""
    records_by_qid: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                qid = str(rec["question_id"])
                records_by_qid[qid] = rec
    return records_by_qid


def audit_w6_split_isolation(
    train_fit_qids: List[str],
    internal_val_qids: List[str],
    dev_qids: List[str],
    train_fit_pids: List[str],
    internal_val_pids: List[str],
    dev_pids: List[str],
) -> None:
    """Verify strictly paper-disjoint splits with zero question or paper leakage."""
    set_fit_q = set(train_fit_qids)
    set_val_q = set(internal_val_qids)
    set_dev_q = set(dev_qids)

    set_fit_p = set(train_fit_pids)
    set_val_p = set(internal_val_pids)
    set_dev_p = set(dev_pids)

    # Question disjointness
    if not set_fit_q.isdisjoint(set_val_q):
        overlap = set_fit_q & set_val_q
        raise ValueError(f"LEAKAGE: {len(overlap)} questions overlap between TRAIN-FIT and INTERNAL-VAL!")
    if not set_fit_q.isdisjoint(set_dev_q):
        overlap = set_fit_q & set_dev_q
        raise ValueError(f"LEAKAGE: {len(overlap)} questions overlap between TRAIN-FIT and DEV!")
    if not set_val_q.isdisjoint(set_dev_q):
        overlap = set_val_q & set_dev_q
        raise ValueError(f"LEAKAGE: {len(overlap)} questions overlap between INTERNAL-VAL and DEV!")

    # Paper disjointness between TRAIN-FIT and INTERNAL-VAL
    if not set_fit_p.isdisjoint(set_val_p):
        overlap_p = set_fit_p & set_val_p
        raise ValueError(f"LEAKAGE: {len(overlap_p)} papers overlap between TRAIN-FIT and INTERNAL-VAL!")

    logger.info(
        f"Split isolation audit passed: "
        f"TRAIN-FIT={len(set_fit_q)}q/{len(set_fit_p)}p, "
        f"INTERNAL-VAL={len(set_val_q)}q/{len(set_val_p)}p, "
        f"DEV={len(set_dev_q)}q/{len(set_dev_p)}p. Overlaps=0."
    )


class PassageRoutingDataset(Dataset):
    """PyTorch Dataset for passage routing.

    For each question record, prepares tokenized pairs (query, passage_i) for i=1..10,
    with rank features (0..9) and deterministic binary gold evidence labels.
    """

    def __init__(
        self,
        question_ids: List[str],
        paper_ids: List[str],
        questions: List[str],
        passages_list: List[List[Dict[str, Any]]],
        labels_list: List[np.ndarray],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 256,
    ):
        self.question_ids = question_ids
        self.paper_ids = paper_ids
        self.questions = questions
        self.passages_list = passages_list
        self.labels_list = labels_list
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        assert len(question_ids) == len(paper_ids) == len(questions) == len(passages_list) == len(labels_list)

    def __len__(self) -> int:
        return len(self.question_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        qid = self.question_ids[idx]
        pid = self.paper_ids[idx]
        q_text = self.questions[idx]
        passages = self.passages_list[idx][:10]
        labels = self.labels_list[idx][:10]

        # Ensure exactly 10 passages
        p_texts = [p.get("text", "").strip() for p in passages]
        while len(p_texts) < 10:
            p_texts.append("")
        while len(labels) < 10:
            labels = np.pad(labels, (0, 10 - len(labels)))

        # Tokenize 10 pairs with strict query preservation (truncation='only_second')
        queries_repeated = [q_text] * 10
        encoded = self.tokenizer(
            queries_repeated,
            p_texts,
            padding="max_length",
            truncation="only_second",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        rank_indices = torch.arange(10, dtype=torch.long)

        return {
            "input_ids": encoded["input_ids"],  # shape: (10, max_seq_length)
            "attention_mask": encoded["attention_mask"],  # shape: (10, max_seq_length)
            "rank_indices": rank_indices,  # shape: (10,)
            "labels": torch.tensor(labels, dtype=torch.float32),  # shape: (10,)
            "question_id": qid,
            "paper_id": pid,
            "question": q_text,
            "passages": passages,
        }


def routing_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate batch of passage routing examples into batched tensors."""
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)  # (B, 10, S)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)  # (B, 10, S)
    rank_indices = torch.stack([item["rank_indices"] for item in batch], dim=0)  # (B, 10)
    labels = torch.stack([item["labels"] for item in batch], dim=0)  # (B, 10)

    question_ids = [item["question_id"] for item in batch]
    paper_ids = [item["paper_id"] for item in batch]
    questions = [item["question"] for item in batch]
    passages_batch = [item["passages"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "rank_indices": rank_indices,
        "labels": labels,
        "question_ids": question_ids,
        "paper_ids": paper_ids,
        "questions": questions,
        "passages": passages_batch,
    }


def build_w6_dataset_bundle(
    train_fit_manifest_path: Union[str, Path] = "results/week5/content_allocator/w5b/splits/train_fit_manifest.csv",
    internal_val_manifest_path: Union[str, Path] = "results/week5/content_allocator/w5b/splits/internal_val_manifest.csv",
    train_retrieval_path: Union[str, Path] = "results/week5/train_response_surface/train_retrieval.jsonl",
    dev_retrieval_path: Union[str, Path] = "results/week2/raw/dev_retrieval.jsonl",
    dev_supervision_path: Union[str, Path] = "results/week5/supervision/dev_allocation_supervision.csv",
    qasper_train_split: Union[str, Path] = "Baselines/SARA-main/data/reformatted/QASPER_train_split.jsonl",
    qasper_dev_split: Union[str, Path] = "Baselines/SARA-main/data/reformatted/QASPER_dev.jsonl",
    raw_arrow_train: Optional[Union[str, Path]] = None,
    raw_arrow_val: Optional[Union[str, Path]] = None,
    qasper_manifest: Optional[Union[str, Path]] = None,
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_seq_length: int = 256,
    evidence_match_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Build unified dataset bundles for TRAIN-FIT, INTERNAL-VAL, and DEV.

    Returns:
        (train_bundle, val_bundle, dev_bundle)
    """
    if evidence_match_kwargs is None:
        evidence_match_kwargs = {
            "min_hl_length": 15,
            "min_overlap_ev": 0.3,
            "min_overlap_p": 0.5,
            "min_shared_words": 10,
        }

    # 1. Load split manifests
    df_fit = pd.read_csv(train_fit_manifest_path)
    df_val = pd.read_csv(internal_val_manifest_path)
    df_dev = pd.read_csv(dev_supervision_path)

    def clean_id(s: Any) -> str:
        v = str(s).strip()
        return str(int(float(v))) if v.replace(".", "", 1).isdigit() else v

    df_fit["question_id"] = df_fit["question_id"].apply(clean_id)
    df_fit["paper_id"] = df_fit["paper_id"].astype(str)
    df_val["question_id"] = df_val["question_id"].apply(clean_id)
    df_val["paper_id"] = df_val["paper_id"].astype(str)
    df_dev["question_id"] = df_dev["question_id"].apply(clean_id)
    df_dev["paper_id"] = df_dev["paper_id"].astype(str)

    # Audit isolation
    audit_w6_split_isolation(
        train_fit_qids=df_fit["question_id"].tolist(),
        internal_val_qids=df_val["question_id"].tolist(),
        dev_qids=df_dev["question_id"].tolist(),
        train_fit_pids=df_fit["paper_id"].tolist(),
        internal_val_pids=df_val["paper_id"].tolist(),
        dev_pids=df_dev["paper_id"].tolist(),
    )

    # 2. Load retrieval records
    train_ret = load_retrieval_records(train_retrieval_path)
    dev_ret = load_retrieval_records(dev_retrieval_path)

    # 3. Load QASPER adapter for gold evidence annotations
    adapter = QASPERAdapter(
        train_split_path=qasper_train_split,
        dev_split_path=qasper_dev_split,
        raw_arrow_train_path=raw_arrow_train,
        raw_arrow_val_path=raw_arrow_val,
        manifest_path=qasper_manifest,
        dev_retrieval_path=dev_retrieval_path,
    )
    train_raw_recs = adapter.load_split("train")
    dev_raw_recs = adapter.load_split("dev")

    train_raw_by_qid = {str(r.get("id", r.get("question_id"))): r for r in train_raw_recs}
    dev_raw_by_qid = {str(r.get("id", r.get("question_id"))): r for r in dev_raw_recs}

    # Helper to construct split bundle
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def construct_bundle(
        df_split: pd.DataFrame,
        ret_dict: Dict[str, Dict[str, Any]],
        raw_dict: Dict[str, Dict[str, Any]],
        split_name: str,
    ) -> Dict[str, Any]:
        qids = df_split["question_id"].tolist()
        pids = df_split["paper_id"].tolist()

        questions: List[str] = []
        passages_list: List[List[Dict[str, Any]]] = []
        labels_list: List[np.ndarray] = []
        match_details_list: List[List[str]] = []

        for qid in qids:
            assert qid in ret_dict, f"Missing retrieval record for {qid} in {split_name}"
            ret_item = ret_dict[qid]
            q_text = str(ret_item.get("question", "")).strip()
            passages = ret_item.get("passages", [])[:10]

            # Get gold evidence
            raw_rec = raw_dict.get(qid)
            if raw_rec is not None:
                ev_items = adapter.get_gold_evidence(raw_rec)
            else:
                ev_items = []

            labels, details = match_passages_to_gold_evidence(
                passages=passages,
                gold_evidence_items=ev_items,
                **evidence_match_kwargs,
            )

            questions.append(q_text)
            passages_list.append(passages)
            labels_list.append(labels)
            match_details_list.append(details)

        dataset = PassageRoutingDataset(
            question_ids=qids,
            paper_ids=pids,
            questions=questions,
            passages_list=passages_list,
            labels_list=labels_list,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )

        labels_arr = np.stack(labels_list, axis=0)  # (N, 10)
        pos_rate = float(np.mean(labels_arr))

        return {
            "split_name": split_name,
            "manifest_df": df_split,
            "question_ids": qids,
            "paper_ids": pids,
            "questions": questions,
            "passages": passages_list,
            "labels": labels_arr,
            "match_details": match_details_list,
            "dataset": dataset,
            "positive_rate": pos_rate,
        }

    train_bundle = construct_bundle(df_fit, train_ret, train_raw_by_qid, "train_fit")
    val_bundle = construct_bundle(df_val, train_ret, train_raw_by_qid, "internal_val")
    dev_bundle = construct_bundle(df_dev, dev_ret, dev_raw_by_qid, "dev")

    return train_bundle, val_bundle, dev_bundle
