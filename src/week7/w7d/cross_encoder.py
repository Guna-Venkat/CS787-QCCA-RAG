"""Pretrained Cross-Encoder wrapper and resumable scoring cache for Week 7-D.

Provides:
- Resumable pair scoring for (q, D_i) over top-10 retrieved passages.
- Parquet score cache with canonical schema:
    split, question_id, paper_id, semantic_rank, bm25_rank, passage_id,
    cross_encoder_score, semantic_score, bm25_score, passage_length.
- Batch prediction with CPU/GPU device selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderScorer:
    """Computes and caches joint (q, D_i) relevance scores using a pretrained cross-encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model = None

    @property
    def model(self):
        """Lazy load model on first inference."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading CrossEncoder model: %s on %s...", self.model_name, self.device)
            self._model = CrossEncoder(self.model_name, device=self.device)
            for param in self._model.model.parameters():
                param.requires_grad = False
            self._model.model.eval()
        return self._model

    def score_pairs(self, query_passage_pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute cross-encoder scores for list of (query, passage_text) tuples."""
        if not query_passage_pairs:
            return []
        scores = self.model.predict(
            query_passage_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]

    def compute_and_cache_scores(
        self,
        retrieval_data: Dict[str, Dict[str, Any]],
        cache_path: Union[str, Path],
        split_name: str = "train",
        max_passages: int = 10,
    ) -> pd.DataFrame:
        """Compute cross-encoder scores for all queries and save to resumable parquet cache.

        If cache_path exists, loads existing cached rows and only processes missing question_ids.
        """
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        existing_df: Optional[pd.DataFrame] = None
        cached_qids: Set[str] = set()

        if cache_file.is_file():
            try:
                existing_df = pd.read_parquet(cache_file)
                cached_qids = set(existing_df["question_id"].astype(str).str.strip())
                logger.info("Loaded %d existing cached query scores from %s.", len(cached_qids), cache_file)
            except Exception as e:
                logger.warning("Could not read existing cache %s: %s. Recomputing.", cache_file, e)
                existing_df = None

        missing_qids = [qid for qid in retrieval_data if qid not in cached_qids]
        logger.info("%d queries to compute scores for in %s.", len(missing_qids), split_name)

        if not missing_qids and existing_df is not None:
            return existing_df

        new_records: List[Dict[str, Any]] = []

        # Batch questions to score efficiently
        all_pairs: List[Tuple[str, str]] = []
        pair_metadata: List[Dict[str, Any]] = []

        for qid in missing_qids:
            q_info = retrieval_data[qid]
            q_text = q_info["question"]
            pid = q_info["paper_id"]
            passages = q_info["passages"][:max_passages]

            for rank_idx, p in enumerate(passages):
                all_pairs.append((q_text, p["text"]))
                pair_metadata.append({
                    "split": split_name,
                    "question_id": qid,
                    "paper_id": pid,
                    "semantic_rank": rank_idx + 1,
                    "bm25_rank": p.get("bm25_rank", rank_idx + 1),
                    "passage_id": p.get("passage_id", f"p_{rank_idx}"),
                    "semantic_score": 1.0 / (rank_idx + 1),  # Default reciprocal rank if raw sim not present
                    "bm25_score": float(p.get("bm25_score", 0.0)),
                    "passage_length": len(p["text"].split()),
                })

        if all_pairs:
            logger.info("Computing cross-encoder scores for %d (query, passage) pairs...", len(all_pairs))
            scores = self.score_pairs(all_pairs)
            for meta, score in zip(pair_metadata, scores):
                meta["cross_encoder_score"] = float(score)
                new_records.append(meta)

        df_new = pd.DataFrame(new_records)

        if existing_df is not None and not df_new.empty:
            df_combined = pd.concat([existing_df, df_new], ignore_index=True)
        elif not df_new.empty:
            df_combined = df_new
        else:
            df_combined = existing_df if existing_df is not None else pd.DataFrame()

        # Sort and save
        if not df_combined.empty:
            df_combined = df_combined.sort_values(by=["question_id", "semantic_rank"]).reset_index(drop=True)
            df_combined.to_parquet(cache_file, index=False)
            logger.info("Saved %d total score rows to cache: %s.", len(df_combined), cache_file)

        return df_combined
