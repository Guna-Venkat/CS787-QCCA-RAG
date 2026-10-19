# Week 9 reference audit

BRANCH = gagan. Local references are the source of truth. No external implementations were substituted.

## Sources inspected

- `Baselines/papers/SemRAG.pdf`, arXiv:2507.21110v1, especially §3.2.1–3.2.3, Algorithm 1 and equations (1)–(2).
- `Baselines/papers/SANE.pdf`, arXiv:2608.00658v1, Method, Algorithm 1, Table 1, cost analysis and faithfulness discussion.
- `Baselines/SANE-main/README.md`, `config/default.yaml`, `config/hotpotqa.yaml`.
- `Baselines/SANE-main/sane/agents/sane_agent.py`: `SANEAgent._retrieval_stage`, `_post_retrieval_stage`, `_select_docs_via_synopsis`, `_build_context_with_evidence`, `send`.
- `Baselines/SANE-main/sane/agents/sane_helpers.py`: `generate_doc_synopses`, `_compute_idf`, `_extract_keywords_tf`, `_extract_best_sentence`, `select_docs`, `extract_evidence`.
- `Baselines/SANE-main/sane/agents/sane_prompts.py`: `DOC_SELECTION_PROMPT`, `BLUEPRINT_GUIDED_PROMPT`, `EVIDENCE_EXTRACTION_PROMPT`, `get_suite_prompt`.
- `Baselines/SANE-main/sane/core/config.py`, `core/interfaces.py`, `llm/base.py`, `llm/openai_client.py`, `embedding/base.py`, `agents/rag_agent.py`, `benchmarks/metrics.py`, `benchmarks/hotpotqa/runner.py`.
- `src/week6/{data,evaluation,routing}.py`; `src/week7/w7b/response_surface.py`; `src/week7/w7e/{prompt_builder,generation}.py`.
- `src/week8/{data,semantic,promotion_surface,generator_features,utility_training,selective_controller,closeout,qcla_training,evidence_enrichment}.py`.
- `scripts/week8/backfill_exact_draft_diagnostics.py`, Week-8 notebook/readiness inventory and tests; `configs/week8/w8_dafir.yaml`.
- `results/week8/surfaces/promotion_surface_fit.jsonl`, Week-7 canonical train results (schema/first records only), Week-8 controller/QCLA artifact inventory. TEST files were not opened.
- `src/week2/precompute_dev_embeddings.py`, `Baselines/SARA-main/scripts/build_qasper_dataset.py`, `Baselines/SARA-main/src/utils/data_utils.py:get_full_text_qasper,rank_documents`.

## SemRAG chunking

Algorithm 1 splits sentences, merges neighboring context using buffer b, embeds buffered sentences, computes adjacent distances `1-cos(z_i,z_{i+1})`, and breaks at threshold theta. Equations constrain token lengths to 1024 with 128-token overlap for oversized groups. The experimental embedding is Nomic-Embed-Text through Ollama. Buffer, threshold and token cap are corpus-dependent; the paper reports different buffer optima on its two datasets. A percentile splitter is not specified.

The paper's prose also describes comparison to the current chunk; Algorithm 1 instead explicitly uses adjacent embeddings. Its loop omits a final flush and does not specify ownership of overlapping buffered sentences. **SemRAG-buffered-project-v1** implements adjacent distance from Algorithm 1, uses symmetric buffers for embedding only, assigns each original sentence once, flushes the last group and splits oversized groups with explicit token overlap. It does not duplicate buffered text in output. MiniLM replaces Nomic for the controlled primary comparison. Threshold is an absolute distance, not a percentile. All deviations are explicit. No knowledge graph, entity extraction, Leiden communities, graph summaries or graph retrieval is implemented.

## SANE

Retrieve a wide dense pool w. Each synopsis combines metadata, TF-IDF keywords using IDF over that candidate pool, and a query-aware lexical-overlap sentence excerpt. **Synopses are query-dependent and computed at query time**, rule-based, with zero LM calls. The LM selects 1-based synopsis positions; mapping back to original identities is essential. `_select_docs_via_synopsis` pads short/empty selections with top dense candidates. History sorting is chronological; QASPER adaptation uses source order.

One joint extraction call processes ALL selected candidates, rather than one call per chunk. Blueprints are soft in-context schema demonstrations: ATOMIC (`fact,evidence_quote,source,source_date`), COMPARATIVE (`entity,dimension,value,unit,evidence_quote,source`), TEMPORAL (`event,position,resolved_date,date_precision,evidence_quote,source`). They are not a fixed classifier. Extraction outputs Markdown tables and generation consumes those tables. Reference pipeline uses three query-time calls (select, extract, answer). In the draft-aware project integration the initial SARA draft is an additional call. Paper models are DeepSeek-V4-Flash/Qwen3.5-9B and Qwen3-Embedding-0.6B; local SARA/Mistral and MiniLM are **adapted baselines**, not exact reproduction.

Reference synopsis/selection/extraction helper bodies and prompt strings are loaded through an isolated AST adapter without importing the API client or modifying the reference. Source/config/prompt hashes participate in cache identities. Selection padding reproduces the reference. Strict source-ID and whitespace-only span validation are project additions. Reference tables permit elisions and inferred calculations: those are retained for audit but fail the project's strict contiguous support check. A valid quote alone does not prove a paraphrased fact is entailed; unsupported-fact rate is null unless a fact is directly copied or separately judged. The paper itself acknowledges unsupported extracted rows.

## Canonical task and extensions

SARA `rank_documents` ranks the complete question-paper context; Week-2 retrieval builds and caches a retriever per paper. Week 9 never performs cross-paper search. Raw Arrow paper sections are restored in source order (not question-ranked context order); original passages are the complete saved reformatted context, aligned back by whitespace-normalized text with offsets. Missing/ambiguous lineage fails rather than silently approximating it. Gold is read separately after retrieval/generation for evaluation only, using the raw paper's exact question annotations (not the W8 question-string-only map).

The canonical k4 allocation and answers are loaded from W7's frozen semantic surface and checked against W8 overlapping promotion rows. Original prompts and SFR ordering are reused unchanged. W9 adds evidence to a new second-stage prompt; it does not rewrite W8 prompts or regenerate y0. k_F=4 is frozen; W and K_E are distinct.

D2 vs D3 share backend, decoding, format and source pool; only initial-draft exposure/objective changes. E0/E1 are clearly labeled project formats; E2 retains the actual SANE blueprint prompt. F0/F1 use identical frozen extracted evidence. Sequential candidate-specific verification in G is conditional on the preregistered FIT headroom gate. FIT selects; INTERNAL-VAL confirms; DEV is inspected; TEST requires an explicit locked final-run interface.

## Final implementation clarifications

Long MiniLM inputs are pooled over explicit encoder windows rather than silently truncated. Original-passage alignment uses the complete original partition to disambiguate short strings that recur inside longer passages. Project SANE selection fixes a ranking budget of four and exposes prefixes; the separately labelled final SANE baseline requests exact K and uses a dense pool. Its concise scientific-QA answer instruction is adapted from the reference evidence-consumption interface. Global G headroom is checked on verified outer FIT OOF F outcomes; outer-fold G training choices are evaluated on that fold's held-out papers.
