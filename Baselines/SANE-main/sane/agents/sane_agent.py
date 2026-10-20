"""SANEAgent — Select-And-Extract RAG agent.

Pipeline: wide retrieval → synopsis → LM selection → evidence extraction → answer.
Synopsis-based selection and evidence extraction are independently toggleable.

Configuration lives in ``SANESettings``, nested under ``method.sane`` in YAML.
"""

from __future__ import annotations

import time

from sane.core.config import Config, SANESettings, load_config
from sane.embedding.base import EmbeddingModel
from sane.llm.openai_client import OpenAIClient
from sane.agents.rag_agent import (
    RAGAgent, RECALL_K_VALUES,
)
from sane.agents.sane_prompts import (
    _ANSWER_TEMPLATE_EVIDENCE,
    DOC_SELECTION_PROMPT_FULL_TEXT,
    EVIDENCE_EXTRACTION_PROMPT,
    get_suite_prompt,
)
from sane.agents.sane_helpers import (
    extract_evidence,
    generate_doc_synopses,
    select_docs,
)
from sane.agents.tool_loop import run_tool_loop


class SANEAgent(RAGAgent):
    """Select-And-Extract RAG agent.

    Extends the doc-level semantic RAG pipeline with two optional stages:

    - Synopsis selection: wide retrieval → condensed synopses → LM selects
      the most promising sessions.
    - Evidence extraction: LM extracts minimum necessary evidence from
      selected sessions as a self-discovered markdown table.

    Both stages are independently toggleable via ``SANESettings``.

    Method key: ``sane``.
    """

    def __init__(
        self,
        config: Config | None = None,
        extra_tools: list[dict] | None = None,
        extra_tool_handlers: dict | None = None,
        system_prompt: str | None = None,
        embedding_model: EmbeddingModel | None = None,
    ):
        self_config = config or load_config()
        mc = self_config.method
        self._real_method = mc.method

        # ── SANE settings (from nested sane: config section) ──────
        st = getattr(mc, "sane", None)
        if st is None:
            st = SANESettings()

        # Evidence extraction
        self._extraction_enabled = st.extraction_enabled
        self._extraction_model = st.extraction_model
        self._extraction_base_url = st.extraction_base_url
        self._extraction_api_key = st.extraction_api_key
        self._extraction_max_tokens = st.extraction_max_tokens
        self._extraction_prompt = st.extraction_prompt or None

        # Synopsis-based wide retrieval (two-pass)
        self._selection_enabled = st.selection_enabled
        self._synopsis_wide_k = getattr(mc, "w", 20)
        self._synopsis_max_select = mc.k
        self._synopsis_keyword_n = st.synopsis_keyword_top_n
        self._synopsis_show_keywords = st.synopsis_show_keywords
        self._synopsis_show_labels = st.synopsis_show_labels
        self._synopsis_include_excerpt = st.synopsis_include_excerpt
        self._synopsis_full_text = st.synopsis_full_text
        self._schema_suite = st.schema_suite

        # Resolve effective extraction prompt template
        if self._extraction_prompt:
            self._effective_extraction_prompt = self._extraction_prompt
        else:
            self._effective_extraction_prompt = (
                get_suite_prompt(self._schema_suite) or EVIDENCE_EXTRACTION_PROMPT
            )

        # Force semantic doc-level retrieval for evidence agent.
        mc.method = "rag"
        try:
            super().__init__(
                config=self_config,
                extra_tools=extra_tools,
                extra_tool_handlers=extra_tool_handlers,
                system_prompt=system_prompt,
                embedding_model=embedding_model,
            )
        finally:
            mc.method = self._real_method

        # ── Extraction artifact capture (for faithfulness evaluation) ─
        self._last_evidence: str = ""
        self._last_extraction_sources: str = ""

        # ── Evidence extraction LLM client ───────────────────────
        if self._extraction_enabled:
            ext_model = self._extraction_model or self.config.llm.model
            ext_base_url = self._extraction_base_url or self.config.llm.base_url
            ext_api_key = self._extraction_api_key or self.config.llm.api_key
            self._extraction_llm = OpenAIClient(
                base_url=ext_base_url,
                api_key=ext_api_key,
                model=ext_model,
                max_tokens=self._extraction_max_tokens,
                temperature=0.0,
                max_retries=5,
                retry_base_s=1.0,
                retry_max_s=10.0,
                request_timeout=120.0,
                chat_template_kwargs=None,
            )
        else:
            self._extraction_llm = None

        # ── Synopsis mode: selection LLM ─────────────────────────
        self._sel_llm = None
        if self._selection_enabled:
            sel_model = self._extraction_model or self.config.llm.model
            sel_base_url = self._extraction_base_url or self.config.llm.base_url
            sel_api_key = self._extraction_api_key or self.config.llm.api_key
            self._sel_llm = OpenAIClient(
                base_url=sel_base_url,
                api_key=sel_api_key,
                model=sel_model,
                max_tokens=256,                 # sufficient for JSON list response
                temperature=0.0,
                max_retries=5,
                retry_base_s=1.0,
                retry_max_s=10.0,
                request_timeout=120.0,
                chat_template_kwargs=None,
            )

    # ── Pipeline stages ──────────────────────────────────────────

    def _retrieval_stage(
        self, query: str, timing: dict
    ) -> list[int]:
        """Retrieve session indices for *query* and compute top-K recall."""
        t0 = time.perf_counter()
        indices = self._retrieve(query)
        timing["retrieve"] = time.perf_counter() - t0

        ranked = self._last_retrieved_ranked or indices
        self._last_retrieval_recall = self._compute_recall(
            ranked, label="topk", k_values=RECALL_K_VALUES)
        return indices

    def _post_retrieval_stage(
        self,
        query: str,
        question: str,
        retrieved_indices: list[int],
        timing: dict,
    ) -> list[dict]:
        """Selection → extraction → context.

        Two independent, sequential stages:

        1. **Selection**: synopsis-based curation (if enabled), otherwise
           pass through the embedding-retrieved indices.
        2. **Extraction**: LLM evidence extraction (if enabled) on top of
           whatever indices were selected in stage 1.
        """
        t0 = time.perf_counter()

        # ── Stage 1: Selection ──────────────────────────────────
        if self._selection_enabled:
            _topk_recall = dict(self._last_retrieval_recall)
            indices = self._select_docs_via_synopsis(query, timing)
            self._last_retrieval_recall.update(_topk_recall)
        else:
            indices = retrieved_indices

        # ── Stage 2: Evidence extraction ────────────────────────
        if self._extraction_enabled and self._extraction_llm is not None:
            prompt = self._effective_extraction_prompt
            raw_history = self._format_history_for_evidence(indices)
            evidence = extract_evidence(
                self._extraction_llm, query, raw_history,
                prompt_template=prompt,
                max_tokens=self._extraction_max_tokens,
            )
            self._last_evidence = evidence
            self._last_extraction_sources = raw_history
            timing["extraction"] = time.perf_counter() - t0
            return self._build_context_with_evidence(evidence, question)
        else:
            timing["build_context"] = time.perf_counter() - t0
            return self._build_context(indices, question)

    # ── Public API ───────────────────────────────────────────────

    def send(self, message: str) -> str:
        """Select-And-Extract pipeline.

        1. Retrieval: semantic top-k
        2. Post-retrieval: synopsis selection → evidence extraction → context
        3. Generate: LLM answer with tool loop
        """
        if not message.strip():
            return ""

        t_send_start = time.perf_counter()
        timing: dict[str, float] = {}

        # Stage 1: Retrieval
        retrieved_indices = self._retrieval_stage(message, timing)

        if not retrieved_indices:
            context = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": f"Question: {message}\nAnswer:"},
            ]
            t0 = time.perf_counter()
            response = run_tool_loop(
                llm_client=self._llm,
                messages=context,
                tools=self._extra_tools if self._extra_tools else None,
                tool_handlers=self._extra_tool_handlers,
            )
            timing["llm_loop"] = time.perf_counter() - t0
            timing["_total"] = time.perf_counter() - t_send_start
            self._last_send_timing = timing
            self._send_count += 1
            return response

        # Stage 2: Post-retrieval (selection → extraction → context)
        context = self._post_retrieval_stage(
            message, message, retrieved_indices, timing)

        # Stage 3: Generate
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

    def reset(self):
        self._last_evidence = ""
        self._last_extraction_sources = ""
        super().reset()
        if self._sel_llm and hasattr(self._sel_llm, "reset_metrics"):
            self._sel_llm.reset_metrics()
        if self._extraction_llm and hasattr(self._extraction_llm, "reset_metrics"):
            self._extraction_llm.reset_metrics()

    def get_timing_summary(self) -> dict:
        result = super().get_timing_summary()
        llm_info: dict = result.setdefault("llm", {})
        if self._sel_llm and hasattr(self._sel_llm, "get_usage_summary"):
            llm_info["synopsis_select"] = self._sel_llm.get_usage_summary()
        if self._extraction_llm and hasattr(self._extraction_llm, "get_usage_summary"):
            llm_info["extraction"] = self._extraction_llm.get_usage_summary()
        return result

    # ── Synopsis-based session selection ──────────────────────────

    def _select_docs_via_synopsis(
        self, query: str, timing: dict,
    ) -> list[int]:
        """Wide retrieval → synopses → LM selection → padding.

        Returns selected session indices. The caller decides whether to
        extract evidence or use the raw history directly.
        """
        # 1. Wide retrieval (top-S)
        t0 = time.perf_counter()
        all_candidates = self._retrieve_scored(
            query, top_k=self._synopsis_wide_k)
        wide_indices = [idx for idx, _ in all_candidates]
        timing["synopsis_retrieve"] = time.perf_counter() - t0

        if not wide_indices:
            return []

        # 2. Generate synopses or format full text (rule-based, no LM)
        t0 = time.perf_counter()
        if self._synopsis_full_text:
            # Format full candidate text — bypass synopsis compression
            parts: list[str] = []
            for i, idx in enumerate(wide_indices):
                doc_text = self._format_history_for_evidence([idx])
                parts.append(f"## Record {i + 1}\n{doc_text}")
            synopses_text = "\n\n".join(parts)
            idx_map = list(wide_indices)
            sel_prompt = DOC_SELECTION_PROMPT_FULL_TEXT
        else:
            synopses_text, idx_map = generate_doc_synopses(
                self._docs, self._dates, wide_indices,
                self._synopsis_keyword_n,
                show_keywords=self._synopsis_show_keywords,
                show_labels=self._synopsis_show_labels,
                labels=self._doc_labels if self._doc_labels else None,
                query=query,
                include_excerpt=self._synopsis_include_excerpt)
            sel_prompt = None  # use default DOC_SELECTION_PROMPT
        timing["synopsis_generate"] = time.perf_counter() - t0

        # 3. LM selects promising sessions from synopses / full text
        t0 = time.perf_counter()
        selected = select_docs(
            self._sel_llm, query, synopses_text, idx_map,
            max_select=self._synopsis_max_select,
            prompt_template=sel_prompt)
        timing["synopsis_select"] = time.perf_counter() - t0

        # 4. Pad with top embedding sessions to reach max_select
        if not selected:
            timing["synopsis_fallback"] = 1
        else:
            timing["synopsis_fallback"] = 0
        selected_set = set(selected)
        for idx in wide_indices:
            if len(selected) >= self._synopsis_max_select:
                break
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
        timing["synopsis_selected_count"] = len(selected)

        # 5. Sort chronologically for display
        if selected:
            indexed = [(idx, self._dates[idx]) for idx in selected]
            indexed.sort(key=lambda x: x[1])
            selected = [idx for idx, _ in indexed]

        # 6. Compute recall for wide retrieval and final selection
        self._last_retrieval_recall = self._compute_recall(
            wide_indices, label="wide",
            k_values=RECALL_K_VALUES)
        self._last_retrieval_recall.update(
            self._compute_recall(selected, label="selected"))

        return selected

    # ── Evidence-based context assembly ───────────────────────────

    def _build_context_with_evidence(
        self, evidence: str, question: str
    ) -> list[dict]:
        """Build LLM messages using extracted evidence instead of raw history."""
        user_prompt = _ANSWER_TEMPLATE_EVIDENCE.format(
            evidence=evidence,
            question_date="unknown",
            question=question,
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]
