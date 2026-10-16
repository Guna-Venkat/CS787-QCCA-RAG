"""Semantic similarity scorer for Week 8 using SentenceTransformer."""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SemanticScorer:
    """Computes cosine similarity between queries and candidate passages."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self._encoder = None

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            logger.info("Loading SentenceTransformer (%s) on %s...", self.model_name, self.device)
            self._encoder = SentenceTransformer(self.model_name, device=self.device)
            self._encoder.eval()
        return self._encoder

    def score_query_passages(self, query: str, passages: List[str]) -> torch.Tensor:
        """Compute cosine similarity scores between query and list of passages."""
        if not passages:
            return torch.empty(0, device=self.device)
        q_emb = self.encoder.encode([query], convert_to_tensor=True, device=self.device)
        p_embs = self.encoder.encode(passages, convert_to_tensor=True, device=self.device)

        q_norm = q_emb / (q_emb.norm(dim=-1, keepdim=True) + 1e-12)
        p_norm = p_embs / (p_embs.norm(dim=-1, keepdim=True) + 1e-12)

        sims = torch.mm(q_norm, p_norm.t()).squeeze(0)
        return sims
