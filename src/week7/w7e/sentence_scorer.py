"""Sentence relevance scoring against query for W7-E MEDIUM compression.

Supports:
- Cross-Encoder scoring: computes joint (q, sentence) scores using CrossEncoder.
- Semantic Bi-Encoder scoring: computes cosine similarity between question embedding and sentence embeddings.
- Deterministic batching and CPU/GPU support.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_SEMANTIC_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceScorer:
    """Computes relevance scores between a query and a list of candidate sentences."""

    def __init__(
        self,
        scorer_type: str = "cross_encoder",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        self.scorer_type = scorer_type.lower()
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.scorer_type == "cross_encoder":
            self.model_name = model_name or DEFAULT_CROSS_ENCODER
        elif self.scorer_type == "semantic":
            self.model_name = model_name or DEFAULT_SEMANTIC_ENCODER
        else:
            raise ValueError(f"Unknown scorer_type: {scorer_type}. Must be 'cross_encoder' or 'semantic'.")

        self._model = None

    @property
    def model(self):
        """Lazy loader for encoder model."""
        if self._model is None:
            if self.scorer_type == "cross_encoder":
                from sentence_transformers import CrossEncoder
                logger.info("Loading CrossEncoder for sentence scoring: %s on %s...", self.model_name, self.device)
                self._model = CrossEncoder(self.model_name, device=self.device)
                for p in self._model.model.parameters():
                    p.requires_grad = False
                self._model.model.eval()
            else:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading SentenceTransformer for sentence scoring: %s on %s...", self.model_name, self.device)
                self._model = SentenceTransformer(self.model_name, device=self.device)
                for p in self._model.parameters():
                    p.requires_grad = False
                self._model.eval()
        return self._model

    def score_sentences(self, query: str, sentences: List[str]) -> List[float]:
        """Compute query-sentence relevance scores.

        Args:
            query: Question text.
            sentences: List of extracted sentence strings.

        Returns:
            List of float scores corresponding 1:1 with sentences.
        """
        if not sentences:
            return []

        if self.scorer_type == "cross_encoder":
            pairs = [(query, s) for s in sentences]
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            return [float(s) for s in scores]
        else:
            # Semantic cosine similarity
            q_emb = self.model.encode(query, convert_to_tensor=True, device=self.device)  # (d,)
            s_embs = self.model.encode(sentences, convert_to_tensor=True, device=self.device)  # (M, d)

            # Normalize and compute cosine sim
            q_norm = torch.nn.functional.normalize(q_emb.unsqueeze(0), p=2, dim=1)
            s_norm = torch.nn.functional.normalize(s_embs, p=2, dim=1)
            sims = torch.mm(s_norm, q_norm.t()).squeeze(1)  # (M,)
            return [float(x) for x in sims.cpu().numpy()]
