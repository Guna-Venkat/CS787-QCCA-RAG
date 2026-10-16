"""MEDIUM extractive passage compressor and resumable caching for W7-E.

Implements query-focused sentence extraction:
1. Deterministic sentence splitting.
2. Relevance scoring against the query.
3. Greedy sentence selection up to token cap (M64, M96, M128).
4. Deterministic restoration of original sentence order.
5. Deterministic truncation for sentences exceeding cap.
6. Fallback handling for empty or malformed passages.
7. Parquet-based resumable caching.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.week7.w7e.sentence_scorer import SentenceScorer
from src.week7.w7e.sentence_splitter import split_into_sentences

logger = logging.getLogger(__name__)


def count_tokens(text: str, tokenizer: Any) -> int:
    """Accurately count tokens using SARA generator tokenizer or whitespace fallback."""
    if not text.strip():
        return 0
    if tokenizer is not None:
        if hasattr(tokenizer, "tokenize"):
            return len(tokenizer.tokenize(text))
        elif hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text, add_special_tokens=False))
    return len(text.split())


def truncate_tokens_deterministic(text: str, token_cap: int, tokenizer: Any) -> str:
    """Truncate text deterministically to exactly token_cap tokens."""
    if not text.strip():
        return ""
    if tokenizer is not None and hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= token_cap:
            return text
        truncated_tokens = tokens[:token_cap]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True).strip()

    # Whitespace fallback
    words = text.split()
    if len(words) <= token_cap:
        return text
    return " ".join(words[:token_cap]).strip()


def extract_medium_passage(
    query: str,
    passage_text: str,
    token_cap: int,
    scorer: SentenceScorer,
    tokenizer: Any = None,
    sentence_scores: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Extract query-focused MEDIUM extractive text from passage.

    Algorithm:
    1. Split passage into sentences s_1 ... s_m.
    2. Score sentences against query (or reuse precomputed sentence_scores).
    3. Rank descending by score.
    4. Greedily collect sentences while cumulative tokens <= token_cap.
    5. If top sentence exceeds cap: deterministically truncate it.
    6. Re-sort selected sentences by original document index.
    7. Join into natural-language text.
    """
    clean_p = passage_text.strip()
    if not clean_p:
        return {
            "selected_text": "",
            "selected_sentence_ids": [],
            "sentence_scores": [],
            "all_sentence_scores": [],
            "original_tokens": 0,
            "medium_tokens": 0,
            "compression_ratio": 0.0,
            "token_cap": token_cap,
            "is_fallback": True,
        }

    orig_tokens = count_tokens(clean_p, tokenizer)

    # 1. Split sentences
    sentences = split_into_sentences(clean_p)
    if not sentences:
        fallback_text = truncate_tokens_deterministic(clean_p, token_cap, tokenizer)
        m_tokens = count_tokens(fallback_text, tokenizer)
        return {
            "selected_text": fallback_text,
            "selected_sentence_ids": [0],
            "sentence_scores": [0.0],
            "all_sentence_scores": [0.0],
            "original_tokens": orig_tokens,
            "medium_tokens": m_tokens,
            "compression_ratio": float(m_tokens / max(1, orig_tokens)),
            "token_cap": token_cap,
            "is_fallback": True,
        }

    # 2. Score sentences
    if sentence_scores is not None:
        scores = list(sentence_scores)
    else:
        scores = scorer.score_sentences(query, sentences)

    # 3. Measure token length of each sentence
    sentence_infos: List[Dict[str, Any]] = []
    for idx, (sent, sc) in enumerate(zip(sentences, scores)):
        t_count = count_tokens(sent, tokenizer)
        sentence_infos.append({
            "orig_idx": idx,
            "text": sent,
            "score": float(sc),
            "tokens": t_count,
        })

    # 4. Rank descending by relevance score
    ranked_by_score = sorted(sentence_infos, key=lambda x: x["score"], reverse=True)

    selected: List[Dict[str, Any]] = []
    accum_tokens = 0

    for item in ranked_by_score:
        if accum_tokens + item["tokens"] <= token_cap:
            selected.append(item)
            accum_tokens += item["tokens"]

    # 5. Handle case where top sentence alone exceeds cap
    if not selected:
        top_item = ranked_by_score[0]
        truncated_text = truncate_tokens_deterministic(top_item["text"], token_cap, tokenizer)
        selected = [{
            "orig_idx": top_item["orig_idx"],
            "text": truncated_text,
            "score": top_item["score"],
            "tokens": count_tokens(truncated_text, tokenizer),
        }]

    # 6. Re-sort selected sentences by original document order
    selected_sorted_order = sorted(selected, key=lambda x: x["orig_idx"])

    # 7. Join sentences
    medium_text = " ".join([s["text"] for s in selected_sorted_order]).strip()
    actual_m_tokens = count_tokens(medium_text, tokenizer)
    comp_ratio = float(actual_m_tokens / max(1, orig_tokens))

    return {
        "selected_text": medium_text,
        "selected_sentence_ids": [int(s["orig_idx"]) for s in selected_sorted_order],
        "sentence_scores": [float(s["score"]) for s in selected_sorted_order],
        "all_sentence_scores": [float(s) for s in scores],
        "original_tokens": orig_tokens,
        "medium_tokens": actual_m_tokens,
        "compression_ratio": comp_ratio,
        "token_cap": token_cap,
        "is_fallback": False,
    }


class MediumExtractionCache:
    """Resumable Parquet cache storing precomputed MEDIUM extracts."""

    def __init__(self, cache_file_path: Union[str, Path]) -> None:
        p = Path(cache_file_path)
        if p.is_dir() or p.suffix == "":
            p.mkdir(parents=True, exist_ok=True)
            self.cache_file_path = p / "w7e_medium_cache.parquet"
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file_path = p

        self.df_cache: Optional[pd.DataFrame] = None
        self.cached_keys: Set[Tuple[str, int, int]] = set()  # (qid, semantic_rank, token_cap)
        self._buffered_records: List[Dict[str, Any]] = []
        self._load()

    def __len__(self) -> int:
        n_cached = len(self.df_cache) if self.df_cache is not None else 0
        return n_cached + len(self._buffered_records)

    def _load(self) -> None:
        if self.cache_file_path.is_file():
            try:
                self.df_cache = pd.read_parquet(self.cache_file_path)
                for _, r in self.df_cache.iterrows():
                    key = (str(r["question_id"]).strip(), int(r["semantic_rank"]), int(r["token_cap"]))
                    self.cached_keys.add(key)
                logger.info("Loaded %d MEDIUM extracts from cache: %s", len(self.df_cache), self.cache_file_path)
            except Exception as e:
                logger.warning("Could not read medium cache %s: %s. Starting fresh.", self.cache_file_path, e)
                self.df_cache = None
                self.cached_keys = set()

    def has_entry(self, question_id: str, semantic_rank: int, token_cap: int) -> bool:
        return (str(question_id).strip(), int(semantic_rank), int(token_cap)) in self.cached_keys

    def get_entry(self, question_id: str, semantic_rank: int, token_cap: int) -> Optional[Dict[str, Any]]:
        # 1. Check buffered records in memory first
        for rec in reversed(self._buffered_records):
            if (
                str(rec["question_id"]).strip() == str(question_id).strip()
                and int(rec["semantic_rank"]) == int(semantic_rank)
                and int(rec["token_cap"]) == int(token_cap)
            ):
                return rec

        # 2. Check loaded dataframe
        if self.df_cache is not None and not self.df_cache.empty:
            sub = self.df_cache[
                (self.df_cache["question_id"].astype(str) == str(question_id).strip())
                & (self.df_cache["semantic_rank"].astype(int) == int(semantic_rank))
                & (self.df_cache["token_cap"].astype(int) == int(token_cap))
            ]
            if not sub.empty:
                return sub.iloc[0].to_dict()

        return None

    def get(
        self,
        split: str,
        question_id: str,
        semantic_rank: int,
        token_cap: int,
        scorer_type: str = "",
    ) -> Optional[Dict[str, Any]]:
        return self.get_entry(question_id, semantic_rank, token_cap)

    def put(
        self,
        split: str,
        question_id: str,
        paper_id: str,
        passage_id: str,
        semantic_rank: int,
        original_text: str,
        selected_sentence_ids: List[int],
        selected_text: str,
        sentence_scores: List[float],
        original_tokens: int,
        medium_tokens: int,
        token_cap: int,
        scorer_type: str = "",
        is_fallback: bool = False,
    ) -> None:
        rec = {
            "split": split,
            "question_id": str(question_id).strip(),
            "paper_id": str(paper_id).strip(),
            "passage_id": str(passage_id).strip(),
            "semantic_rank": int(semantic_rank),
            "original_text": original_text,
            "selected_sentence_ids": selected_sentence_ids,
            "selected_text": selected_text,
            "sentence_scores": sentence_scores,
            "original_tokens": int(original_tokens),
            "medium_tokens": int(medium_tokens),
            "compression_ratio": float(medium_tokens / max(1, original_tokens)),
            "token_cap": int(token_cap),
            "scorer_type": scorer_type,
            "is_fallback": is_fallback,
        }
        self._buffered_records.append(rec)
        key = (str(question_id).strip(), int(semantic_rank), int(token_cap))
        self.cached_keys.add(key)
        # Auto-flush every 500 records
        if len(self._buffered_records) >= 500:
            self.flush()

    def flush(self) -> None:
        if not self._buffered_records:
            return
        self.append_and_save(self._buffered_records)
        self._buffered_records = []

    def append_and_save(self, new_records: List[Dict[str, Any]]) -> None:
        """Append newly extracted records and persist to Parquet immediately."""
        if not new_records:
            return
        df_new = pd.DataFrame(new_records)
        for _, r in df_new.iterrows():
            key = (str(r["question_id"]).strip(), int(r["semantic_rank"]), int(r["token_cap"]))
            self.cached_keys.add(key)

        if self.df_cache is not None and not self.df_cache.empty:
            self.df_cache = pd.concat([self.df_cache, df_new], ignore_index=True)
        else:
            self.df_cache = df_new

        self.df_cache.to_parquet(self.cache_file_path, index=False)
        logger.info("Saved %d total records to medium cache: %s", len(self.df_cache), self.cache_file_path)

    def get_dataframe(self) -> pd.DataFrame:
        self.flush()
        return self.df_cache if self.df_cache is not None else pd.DataFrame()
