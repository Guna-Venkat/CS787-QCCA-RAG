"""RAG agent with embedding-based semantic retrieval.

Indexes history at document granularity and retrieves via cosine similarity.
"""

from __future__ import annotations

import time

import numpy as np

from sane.core.config import Config, load_config
from sane.core.interfaces import BaseAgent
from sane.embedding.base import EmbeddingModel
from sane.llm.openai_client import OpenAIClient
from sane.agents.tool_loop import run_tool_loop


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer the user's questions based on "
    "the history records provided."
)

# Standard k values for recall@k evaluation.
RECALL_K_VALUES: list[int] = [5, 10, 20, 30]


# Prompt template for generation.
_ANSWER_TEMPLATE = (
    "I will give you several history records. "
    "Please answer the question based on the relevant records.\n\n"
    "History Records:\n"
    "{history}\n\n"
    "Current Date: {question_date}\n"
    "Question: {question}\n"
    "Answer:"
)

_DOC_HEADER = "### Doc {idx}:\nDate: {date}\nContent:\n"


class RAGAgent(BaseAgent):
    """RAG agent aligned with the official LongMemEval pipeline.

    Docs are indexed at natural boundaries — one entry per doc, no
    fixed-token chunking.  Retrieval uses cosine similarity over embeddings.

    Parameters
    ----------
    config : Config
        Agent configuration.  Relevant keys under ``method``:
        ``k``, ``rag_user_only``.
    embedding_model : EmbeddingModel, optional
        Pre-loaded embedding model.
    """

    def __init__(
        self,
        config: Config | None = None,
        extra_tools: list[dict] | None = None,
        extra_tool_handlers: dict | None = None,
        system_prompt: str | None = None,
        embedding_model: EmbeddingModel | None = None,
    ):
        self.config = config or load_config()
        mc = self.config.method
        self._method = mc.method
        self._top_k = mc.k
        self._user_only = mc.rag_user_only

        self._llm = OpenAIClient(
            base_url=self.config.llm.base_url,
            api_key=self.config.llm.api_key,
            model=self.config.llm.model,
            max_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature,
            max_retries=self.config.llm.max_retries,
            retry_base_s=self.config.llm.retry_base_s,
            retry_max_s=self.config.llm.retry_max_s,
            request_timeout=120.0,
            chat_template_kwargs=self.config.llm.chat_template_kwargs,
        )

        self._extra_tools = extra_tools or []
        self._extra_tool_handlers = extra_tool_handlers or {}
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        # ── Index state ──────────────────────────────────────────
        # _index_texts: the text used for retrieval (user turns concatenated)
        # _docs: full turn data for generation (user + assistant)
        # _dates: timestamps for sorting and display
        # _doc_ids: document identifiers
        self._index_texts: list[str] = []
        self._docs: list[list[dict]] = []
        self._dates: list[str] = []
        self._doc_ids: list[str] = []
        self._doc_labels: list[str] = []  # per-benchmark display label for synopsis

        # ── Deferred batched ingest ────────────────────────────────
        # When True, embeddings are built lazily on first retrieval
        # to enable single-batch embedding per instance.
        self._pending_ingest: bool = False
        self._pending_index_texts: list[str] = []

        # Embedding state
        if embedding_model is None:
            from sane.embedding import create_embedding_model
            self._embedding_model = create_embedding_model(self.config.embedding)
        else:
            self._embedding_model = embedding_model
        self._embeddings: list[list[float]] = []

        # Ground-truth answer doc IDs for retrieval recall (set by runner)
        self._answer_doc_ids: set[str] = set()

        # Per-send retrieval recall snapshot (populated by send())
        self._last_retrieval_recall: dict[str, float] = {}

        # Relevance-ranked indices from last retrieval (before date-sorting),
        # for proper recall@k computation.
        self._last_retrieved_ranked: list[int] = []

        # Timing
        self._last_send_timing: dict[str, float] = {}
        self._cumulative_timing: dict[str, float] = {}
        self._send_count: int = 0
        self._ingest_count: int = 0

    # ── Public API ───────────────────────────────────────────────

    def ingest_doc(
        self,
        turns: list[dict],
        doc_id: str = "",
        date: str = "",
        label: str | None = None,
    ):
        """Index a document as a single retrieval unit.

        Index text = space-joined user turns.
        Full conversation data (including assistant turns) is preserved for
        generation.

        Parameters
        ----------
        turns : list[dict]
            Turns with ``role`` and ``content`` keys.
        doc_id : str
            Unique document identifier.
        date : str
            Timestamp for sorting and display.
        """
        if not turns:
            return

        t_start = time.perf_counter()

        # Index text: user turns only (matching official)
        if self._user_only:
            index_text = " ".join(
                t["content"] for t in turns if t.get("role") == "user"
            )
        else:
            index_text = " ".join(
                t["content"] for t in turns
                if t.get("role") in ("user", "assistant")
            )

        if not index_text.strip():
            return

        # Truncate if the index text exceeds the embedding model's token limit.
        if self._embedding_model is not None:
            max_tok = self._embedding_model.max_input_tokens()
            if max_tok and max_tok > 0:
                limit = int(max_tok * 0.9)
                try:
                    import tiktoken
                    enc = tiktoken.get_encoding("cl100k_base")
                    tokens = enc.encode(index_text, disallowed_special=())
                    if len(tokens) > limit:
                        index_text = enc.decode(tokens[:limit])
                except Exception:
                    pass

        self._index_texts.append(index_text)
        self._docs.append(turns)
        self._dates.append(date)
        self._doc_ids.append(doc_id)
        # Only store explicitly-provided labels; skip auto-generated defaults.
        # Default "Doc {N} ({date})" is redundant with the date the
        # synopsis already shows and can confuse LM positional selection.
        self._doc_labels.append(label if label else "")

        # Defer: accumulate for batch embedding
        self._pending_ingest = True
        self._pending_index_texts.append(index_text)

        self._cumulative_timing["ingest"] = (
            self._cumulative_timing.get("ingest", 0.0)
            + time.perf_counter() - t_start
        )
        self._ingest_count += 1

    def ingest_text(
        self,
        text: str,
        doc_id: str = "",
        date: str = "",
        label: str | None = None,
    ):
        """Index raw text as a retrieval unit without role filtering.

        For document / chunk benchmarks where the data has no conversational
        structure.  The text is used directly as the index text.
        Internally stored as a single-user-turn session
        so synopsis and evidence extraction work without modification.

        Parameters
        ----------
        text : str
            Raw text to index.
        doc_id : str
            Unique identifier for this retrieval unit.
        date : str
            Timestamp for sorting and display (empty for document chunks).
        """
        if not text or not text.strip():
            return

        t_start = time.perf_counter()

        index_text = text.strip()

        # Truncate if the index text exceeds the embedding model's token limit
        if self._embedding_model is not None:
            max_tok = self._embedding_model.max_input_tokens()
            if max_tok and max_tok > 0:
                limit = int(max_tok * 0.9)
                try:
                    import tiktoken
                    enc = tiktoken.get_encoding("cl100k_base")
                    tokens = enc.encode(index_text, disallowed_special=())
                    if len(tokens) > limit:
                        index_text = enc.decode(tokens[:limit])
                except Exception:
                    pass

        self._index_texts.append(index_text)
        # Store as a single-user-turn session for synopsis / evidence code
        self._docs.append([{"role": "user", "content": text}])
        self._dates.append(date)
        self._doc_ids.append(doc_id)
        self._doc_labels.append(label if label else "")

        self._pending_ingest = True
        self._pending_index_texts.append(index_text)

        self._cumulative_timing["ingest"] = (
            self._cumulative_timing.get("ingest", 0.0)
            + time.perf_counter() - t_start
        )
        self._ingest_count += 1

    def send(self, message: str) -> str:
        """Retrieve relevant sessions and generate an answer."""
        if not message.strip():
            return ""

        t_send_start = time.perf_counter()
        timing: dict[str, float] = {}

        # Retrieve top-k sessions
        t0 = time.perf_counter()
        retrieved_indices = self._retrieve(message)
        timing["retrieve"] = time.perf_counter() - t0

        # Compute retrieval recall against answer sessions (if set by runner).
        # Use relevance-ranked indices for proper recall@k, not date-sorted.
        ranked = self._last_retrieved_ranked or retrieved_indices
        self._last_retrieval_recall = self._compute_recall(
            ranked, label="retrieval",
            k_values=RECALL_K_VALUES)

        # Build prompt
        t0 = time.perf_counter()
        context = self._build_context(retrieved_indices, message)
        timing["build_context"] = time.perf_counter() - t0

        # Generate
        t0 = time.perf_counter()
        response = run_tool_loop(
            llm_client=self._llm,
            messages=context,
            tools=self._extra_tools if self._extra_tools else None,
            tool_handlers=self._extra_tool_handlers,
        )
        timing["llm_loop"] = time.perf_counter() - t0

        for stage, elapsed in timing.items():
            self._cumulative_timing[stage] = (
                self._cumulative_timing.get(stage, 0.0) + elapsed
            )
        timing["_total"] = time.perf_counter() - t_send_start
        self._last_send_timing = timing
        self._send_count += 1

        return response

    def set_answer_doc_ids(self, ids: list[str]) -> None:
        """Set ground-truth answer session IDs for retrieval recall computation.

        Called by the benchmark runner before each ``send()`` so the agent can
        report how many answer sessions were captured by retrieval.
        """
        self._answer_doc_ids = set(ids)

    def _compute_recall(
        self, indices: list[int], label: str = "retrieval",
        k_values: list[int] | None = None,
    ) -> dict[str, float]:
        """Compute recall metrics for *indices* against answer docs.

        Args:
            indices: Retrieved/selected indices into ``self._doc_ids``.
            label: Prefix for the metric keys (e.g. "retrieval", "wide", "selected").
            k_values: Optional list of k cutoffs for recall@k computation.
                If provided, additional ``{label}_recall@{k}`` keys are included.

        Returns:
            Dict with ``{label}_recall`` (fraction of answer docs found) and
            ``{label}_hit`` (at least one answer doc found), plus optional
            ``{label}_recall@{k}`` entries.
        """
        k_values = k_values or []
        if not self._answer_doc_ids:
            return {}
        if not indices or not self._doc_ids:
            result: dict[str, float] = {
                f"{label}_recall": 0.0,
                f"{label}_hit": 0.0,
            }
            for k in k_values:
                result[f"{label}_recall@{k}"] = 0.0
            return result

        def _ids_at(sub_indices: list[int]) -> set[str]:
            return {
                self._doc_ids[idx]
                for idx in sub_indices
                if 0 <= idx < len(self._doc_ids)
            }

        # Full recall (all retrieved indices)
        rid_full = _ids_at(indices)
        overlap = rid_full & self._answer_doc_ids
        recall = len(overlap) / len(self._answer_doc_ids)
        hit = 1.0 if overlap else 0.0
        result = {
            f"{label}_recall": recall,
            f"{label}_hit": hit,
        }

        # Recall@k: evaluate recall at each cutoff on the prefix of indices
        for k in k_values:
            k_indices = indices[:k]
            k_overlap = _ids_at(k_indices) & self._answer_doc_ids
            k_recall = len(k_overlap) / len(self._answer_doc_ids)
            result[f"{label}_recall@{k}"] = k_recall

        return result

    def reset(self):
        self._index_texts.clear()
        self._docs.clear()
        self._dates.clear()
        self._doc_ids.clear()
        self._doc_labels.clear()
        self._embeddings.clear()

        self._pending_ingest = False
        self._pending_index_texts.clear()
        self._answer_doc_ids.clear()
        self._last_retrieval_recall.clear()
        self._last_retrieved_ranked.clear()
        self._cumulative_timing.clear()
        self._last_send_timing.clear()
        self._send_count = 0
        self._ingest_count = 0
        if hasattr(self._llm, "reset_metrics"):
            self._llm.reset_metrics()

    def get_timing_summary(self) -> dict:
        total = sum(
            v for k, v in self._cumulative_timing.items()
            if not k.startswith("_")
        )
        result = {
            "send_count": self._send_count,
            "ingest_count": self._ingest_count,
            "total_s": total,
            "stages": dict(self._cumulative_timing),
            "last_send": dict(self._last_send_timing),
            "llm": {
                "main": self._llm.get_usage_summary()
                if hasattr(self._llm, "get_usage_summary") else {},
            },
        }
        # Always include retrieval recall if available
        if self._last_retrieval_recall:
            result.setdefault("ie", {})
            result["ie"].update(self._last_retrieval_recall)
        return result

    # ── Retrieval ────────────────────────────────────────────────

    def _flush_ingest(self):
        """Batch-embed all pending texts in one shot.

        Called lazily on first retrieval so that all session texts accumulated
        during ingest are embedded in a single batch.
        """
        if not self._pending_ingest or not self._pending_index_texts:
            return
        n = len(self._pending_index_texts)
        # Batch embed — split large batches to stay under API token limits
        # (OpenAI embedding: max ~300K tokens/request; ~100 chunks × 2K tokens
        #  = ~200K tokens per batch, safe with margin).
        _batch_size = 100
        for _start in range(0, n, _batch_size):
            _batch = self._pending_index_texts[_start:_start + _batch_size]
            embs = self._embedding_model.embed(_batch)
            self._embeddings.extend(embs)
        self._pending_index_texts.clear()
        self._pending_ingest = False

    def _retrieve(self, query: str) -> list[int]:
        """Return indices of top-k entries for *query*, sorted by date.

        Matching the official pipeline, retrieved entries are re-sorted by
        session date so the LLM sees them in chronological order.
        """
        if not self._index_texts or self._top_k <= 0:
            return []

        # Lazy batch ingest: embed all pending texts
        self._flush_ingest()

        scored = self._retrieve_semantic(query)

        # Store relevance-ranked indices before date-sorting (for recall@k)
        self._last_retrieved_ranked = [idx for idx, _ in scored]

        # Sort by date (chronological) — matches official
        scored.sort(key=lambda x: self._dates[x[0]])

        return [idx for idx, _ in scored]

    def _retrieve_scored(
        self, query: str, top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Return relevance-ranked (index, score) tuples for *query*.

        Like :meth:`_retrieve` but returns scored tuples WITHOUT date-sorting.
        Used by wide-retrieval stages (e.g. synopsis candidate pools) that need
        raw relevance scores.
        """
        k = top_k or self._top_k
        if not self._index_texts or k <= 0:
            return []

        self._flush_ingest()
        return self._retrieve_semantic(query, top_k=k)

    def _retrieve_semantic(self, query: str,
                           restrict_to: set[int] | None = None,
                           top_k: int | None = None) -> list[tuple[int, float]]:
        query_emb = self._embedding_model.embed([query])[0]
        if not self._embeddings:
            return []

        chunk_matrix = np.array(self._embeddings, dtype=np.float32)
        query_vec = np.array(query_emb, dtype=np.float32)

        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        chunk_norms = chunk_matrix / (
            np.linalg.norm(chunk_matrix, axis=1, keepdims=True) + 1e-10
        )
        similarities = np.dot(chunk_norms, query_norm)

        k = min(top_k or self._top_k, len(self._index_texts))

        if restrict_to is not None:
            # Score only within restricted indices; sort by similarity
            valid = [(int(i), float(similarities[i]))
                     for i in restrict_to if 0 <= i < len(similarities) and similarities[i] > 0]
            valid.sort(key=lambda x: x[1], reverse=True)
            return valid[:k]

        top = np.argsort(similarities)[::-1][:k]
        return [(int(i), float(similarities[i])) for i in top if similarities[i] > 0]


    # ── Context assembly ─────────────────────────────────────────

    def _format_history_for_evidence(self, retrieved_indices: list[int]) -> str:
        """Format retrieved sessions as raw text for evidence extraction.

        Returns the formatted history string without wrapping it in an
        answer template, so it can be used as input to a post-retrieval
        LM call (e.g., evidence extraction).
        """
        history_parts = []
        for display_idx, sess_idx in enumerate(retrieved_indices, start=1):
            turns = self._docs[sess_idx]
            date = self._dates[sess_idx]
            header = _DOC_HEADER.format(idx=display_idx, date=date)
            turn_lines = []
            for turn in turns:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                turn_lines.append(f"{role}: {content}")
            history_parts.append(header + "\n".join(turn_lines))

        return "\n\n".join(history_parts) if history_parts else "(no relevant history)"

    def _build_context(
        self, retrieved_indices: list[int], question: str
    ) -> list[dict]:
        """Build LLM messages: system + formatted history + question.

        Format matches the official ``prepare_prompt()`` output:
        system prompt is the first message, history and question are
        combined into a single user message.
        """
        history_str = self._format_history_for_evidence(retrieved_indices)

        # Build user prompt matching official template
        user_prompt = _ANSWER_TEMPLATE.format(
            history=history_str,
            question_date="unknown",
            question=question,
        )

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]
