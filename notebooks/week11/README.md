# Week 11 — multi-granular selective-fidelity laboratory

Implementation framework only. No Week-11 scientific result or winner is claimed from smoke fixtures. Full runs are manual and opt-in.

## Start here

1. Open `notebooks/week11/week11_MASTER.ipynb` for status. It never launches generation.
2. Open `week11_0_setup_and_data.ipynb` and Run All with its defaults. This is fabricated smoke only.
3. Development data has now been prepared at the user’s request. `sara_env/bin/python scripts/week11/prepare_datasets.py --download` verifies and reuses locked files. If the original Hotpot host is unavailable, the downloader explicitly uses the pinned `hotpotqa/hotpot_qa` distractor/validation source, restores the native JSON field layout without altering source strings, and records its revision, conversion and hashes in `data/week11/source_lock.json`. No experimental phase is started by this command.
4. Inspect `configs/week11/`: primary reader Qwen; SciFact verification; sample50 IDs per dataset; beta5; gamma1; hybrid0.5; budgets256/512/1024/FULL. Change design choices before the first real setup/freeze, not after outcomes.
5. For a manual real phase, change only `RUN_FULL=True` and `RUN_SMOKE=False` in its first FLAGS cell, then Run All. Derived phase flags require no other changes. Validated existing rows and identical-text reader outputs are reused.
6. After real results finish, set `REVIEW_PHASE`, `REVIEWED_CONFIG` (dictionary), and `REVIEW_REASON` in FLAGS. Rerun the review cell, or Run All to reuse existing rows. The next dependent full phase refuses a missing/changed review.
7. If optional LLMLingua conditions remain pending, review can explicitly acknowledge `REVIEWED_CONFIG={"waive_pending":["LLMLingua-2"]}`. This does not claim those conditions were executed. Otherwise qualify the token compressor before the first real freeze.
8. Before F, copy actual reviewed action IDs and their configuration dictionaries into `week11_runs.yaml`. Empty/nonexistent finalist IDs refuse execution. Later finalist/analysis fields are scoped so they do not invalidate unrelated earlier freezes.
9. H/I consume saved results only. No generation occurs in them. Default generation notebooks prefer saved real results when their status receipt exists, otherwise they run fabricated smoke. Both smoke and full disabled loads saved views; no work starts in generation notebooks.

For F, the dictionary keyed by a copied action ID uses input fields such as `reader: Qwen`, `granularity: SENTENCE`, `selector: semantic`, `budget: 512`, `compressor: C0`, `pool_width: 20`. Do not paste a whole output summary row with metric fields; unknown settings are rejected. These are manually selected candidate settings, not automatically discovered winners.

## Files created / modified

All implementation changes are Week11-scoped. Historical Week0–10 files are read-only. Source/config/notebook entry points:

- `src/week11/__init__.py`
- `src/week11/bootstrap.py`
- `src/week11/budgets.py`
- `src/week11/cache.py`
- `src/week11/catalog.py`
- `src/week11/compressors.py`
- `src/week11/dashboard.py`
- `src/week11/datasets.py`
- `src/week11/embeddings.py`
- `src/week11/evaluation.py`
- `src/week11/factorial.py`
- `src/week11/fixtures.py`
- `src/week11/generation.py`
- `src/week11/hierarchical.py`
- `src/week11/hierarchy.py`
- `src/week11/oracle.py`
- `src/week11/planning.py`
- `src/week11/plotting.py`
- `src/week11/prompts.py`
- `src/week11/propositions.py`
- `src/week11/protocol.py`
- `src/week11/provenance.py`
- `src/week11/qasper_adapter.py`
- `src/week11/readers.py`
- `src/week11/reporting.py`
- `src/week11/retrievers.py`
- `src/week11/runner.py`
- `src/week11/selectors.py`
- `scripts/week11/build_notebooks.py`
- `scripts/week11/check_token_compressor.py`
- `scripts/week11/prepare_datasets.py`
- `scripts/week11/validate_analysis_smoke.py`
- `scripts/week11/validate_notebooks.py`
- `scripts/week11/write_guide.py`
- `configs/week11/week11_base.yaml`
- `configs/week11/week11_compressors.yaml`
- `configs/week11/week11_datasets.yaml`
- `configs/week11/week11_granularity.yaml`
- `configs/week11/week11_readers.yaml`
- `configs/week11/week11_retrievers.yaml`
- `configs/week11/week11_runs.yaml`
- `tests/week11/test_data_preparation.py`
- `tests/week11/test_framework.py`
- `notebooks/week11/week11_0_setup_and_data.ipynb`
- `notebooks/week11/week11_A_retrieval_granularity.ipynb`
- `notebooks/week11/week11_B_passage_representation.ipynb`
- `notebooks/week11/week11_C_granularity_budget.ipynb`
- `notebooks/week11/week11_D_reader_granularity.ipynb`
- `notebooks/week11/week11_E_compressor_reader.ipynb`
- `notebooks/week11/week11_F_retriever_robustness.ipynb`
- `notebooks/week11/week11_G_hierarchical_fidelity.ipynb`
- `notebooks/week11/week11_H_oracle_and_interactions.ipynb`
- `notebooks/week11/week11_I_final_report.ipynb`
- `notebooks/week11/week11_MASTER.ipynb`

Additional new artifacts: Week11-scoped cache/raw-data/output ignore files; `setup/historical_sha256.json`, protocol/resource inventories, fabricated smoke tables/figures, and this guide.

## Dataset protocol

| Dataset | Split | Candidate corpus | Evaluation |
|---|---|---|---|
| QASPER | Existing project DEV, drawn from raw QASPER TRAIN | Full corresponding scientific paper | Existing project token F1/EM; answer-type breakdown |
| HotpotQA | Official dev, distractor setting | Supplied ten paragraphs, including titles | Official normalization/yes-no answer F1 and EM; selected-source evidence coverage |
| SciFact | Official claims_dev + corpus | Entire abstract corpus including titles; not oracle cited documents | Strict normalized label accuracy, fixed-three-class macro-F1, per-class metrics, invalid-label rate |

SciFact SUPPORT/CONTRADICT map to SUPPORT/REFUTE. Empty-evidence claims retain NOT_ENOUGH_INFO. Conflicting labels fail explicitly. Corpus titles are source text and native title units; original sentence IDs are retained for gold mapping. Gold is evaluation-only, and no labels enter inference dictionaries. No QASPER TEST file is read, selected, generated against, or evaluated. These are characterization splits, not blind confirmation.

**Default study size:** deterministic SHA256 sample of50 queries per dataset (150 total), seed42, independent of labels/outcomes. Full available counts and exact selected IDs are saved in `cohort.json`. Change this predefined size before freezing if desired.

External dataset byte hashes are locked before outcomes. The official SciFact release URL is mutable; `source_lock.json` freezes the actual first-downloaded bytes. Following implementation, the user authorized development-data downloads; HotpotQA DEV and SciFact DEV/corpus are now present, with no experimental phase started.

Primary schema references: [HotpotQA official site](https://hotpotqa.github.io/), [official answer evaluator](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py), [SciFact schema](https://github.com/allenai/scifact/blob/master/doc/data.md).

## Evidence/retrieval protocol

- Canonical hierarchy: Document → Passage → Sentence → PROPOSITION_RULE_V1. Every unit has exact offsets, source hash, parent identity, document order, and canonical Qwen token count.
- Native Hotpot paragraphs and SciFact abstracts are passages. QASPER uses frozen exact-word-boundary passages under256 actual Qwen tokens, zero overlap. Single lexical units exceeding the ceiling fail, not silently truncate.
- Academic sentence splitter is reused read-only from E3. PROPOSITION_RULE_V1 conservatively splits semicolon/dash clauses. These are exact source fragments, not Dense X propositions or generated atomic facts.
- BM25 is the project Okapi k1=1.5/b=.75 implementation; MiniLM and SFR use exact W10 revisions. SFR query instruction and explicit long-input window pooling are reused read-only.
- P0 direct normalized passage vectors; P1 mean normalized sentence vectors then renormalize; P2 maximum query/sentence cosine; P3 softmax(beta5 × cosine) weighted sentence vector then renormalize. Empty sentence sets use direct passage vectors.
- Selectors: lexical, SFR semantic, deterministic min-max hybrid0.5/0.5; constant score columns normalize to zero. Density is raw semantic relevance/token_count^1, used by H5. No outcome-dependent tuning.
- C/D share a query-retrieved SFR top20 passage pool; E uses top10 for matched SARA source exposure. FULL means this frozen available candidate pool, capped once at6144 evidence tokens using the maximum actual count across all three tokenizers. This is not unlimited document input.
- Ranking selects WHAT; presentation restores original document/source order. Unit additions use strict prefix-under-budget; an oversized next unit stops selection. Parent/child overlaps cannot be double-exposed.
- Shared evidence must satisfy ALL pinned reader tokenizers. Per-reader actual evidence/prompt/generated token counts remain separate. This conservative shared-budget convention guarantees identical textual evidence in D; it is not independently optimized token filling for each reader.
- Retrieval gold metrics use full non-whitespace annotated-character union coverage, repeated QASPER occurrences as alternatives, and strict all-annotation coverage. Whitespace gaps created at exact clause/sentence boundaries do not turn otherwise complete content into a recall miss. Unmapped evidence remains missed. No-gold recall is null with an explicit denominator. This is not official SciFact rationale scoring or Hotpot joint answer/support F1.

## Frozen models

| Name | Model | Revision |
|---|---|---|
| Mistral | mistralai/Mistral-7B-Instruct-v0.2 | `63a8b081895390a26e140280378bc85ec8bce07a` |
| MiniLM | sentence-transformers/all-MiniLM-L6-v2 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| SFR | Salesforce/SFR-Embedding-Mistral | `33a60fe7b5e00b9414981280c571e8a4f9f42f1b` |
| Qwen | Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` |
| Llama | meta-llama/Llama-3.1-8B-Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` |

Readers use FP16, no quantization, efficient SDPA, greedy decoding, seed42, max128 new tokens, max8192 context. Qwen thinking is disabled. The same user instruction/evidence is wrapped in each native chat template. Context overflow fails without truncation. Peak VRAM is recorded only when measured.

## Compressors and hierarchy

| Condition | State | Meaning |
|---|---|---|
| C0 | Implemented; synthetic smoke | Selected source text, no additional transformation |
| C1 / TOKEN | Adapter + mock interface tested; full pending | LLMLingua-2 token deletion, not cosine-ranked wordpieces |
| C2 | Implemented; synthetic smoke | Query-only SFR-ranked top8 exact sentences, budget-constrained and source-order restored; E3-inspired mechanism on W11 pools, not reuse/reinterpretation of W9 outcomes |
| C3 | Existing runtime integrated; mocked/synthetic restriction tests | MISTRAL_ONLY_LATENT_BASELINE, QASPER local SARA checkpoint, k4 text/n10 total; no new projector training |
| C4 | Future work | Faithful SeleCom not implemented; no approximation is labelled SeleCom |

LLMLingua is not installed in this environment, and no compressor model was downloaded. Probe: `sara_env/bin/python scripts/week11/check_token_compressor.py`. After user installation, a separate explicit `--qualify --revision <immutable-revision> --download --write-config` performs one fabricated-input qualification and pins the configuration. Do this before real setup/freeze, or retain the explicit pending conditions. No token-compression performance is invented. [Official LLMLingua-2 example](https://github.com/microsoft/LLMLingua/blob/main/examples/LLMLingua2.ipynb).

H0 all passages; H1 full head plus sentence remainder; H2 full head plus clause remainder; H3 sentence head/middle plus clause tail; H4 full head + sentence middle + LLMLingua token tail (pending); H5 density-ranked mixed units with overlap suppression. Head count, middle region and tail fraction are configured. These are deterministic prototypes, not learned controllers.

H4 compresses tail passages separately, records their parent IDs/hashes, and restores all full/sentence/compressed components to source order. Token output is not falsely assigned exact contiguous source spans. C3 uses an adapted local checkpoint with its own adapted no-evidence baseline for HELP/HURT, so it is a distinct specialist reader variant: portable reader/compressor comparisons and Pareto frontiers exclude it. A C3-versus-public-Mistral difference does not isolate compression alone.

## Cache, resume, safety

All caches are new under `cache/week11/v1/`. Normalized queries reference shared documents rather than copying the SciFact corpus into every query shard. Hierarchy is cached per document. Text vectors share a deduplicated SQLite store; passage/sentence/proposition index manifests reference that store rather than duplicating large SFR vectors. BM25 inverted postings and query rankings persist separately.

Selection keys include dataset/split/query, source identity, retriever, granularity, selector, budget, chunking config, encoder revision, tokenizer identity and strategy. Actual encoder/compressor weight hashes identify the selection cache namespace. D is reuse-only and refuses missing C evidence instead of recomputing rankings per reader. Generation reuse keys include question, exact evidence, prompt/task, reader weights, decoding and any latent vectors/checkpoint.

Before resume: config, relevant source/dependency code, dataset bytes, pinned model bytes and evidence-selection hashes are checked. Existing rows are checksummed and immutable. A mismatch requires a new version; overwrite is unsupported. Completed rows skip model generation. Embedding preprocessing and cached-original timing are distinguished; no false zero conceptual cost or unmeasured end-to-end latency is reported. Candidate text in memory is not counted as LM prompt tokens.

Sparse output shards/cache databases and downloaded raw data have Week11-scoped ignore files; summaries, protocols and plots remain inspectable. No historical .gitignore, source, notebook, result or model file was edited.

## Notebook inventory and exact order

| Order | Notebook | Purpose/output | Flags and action |
|---|---|---|
| 0 | `week11_0_setup_and_data.ipynb` | `setup/` | Default RUN_SMOKE=True/RUN_FULL=False; opt in manually after prerequisite review |
| A | `week11_A_retrieval_granularity.ipynb` | `retrieval/` | Default RUN_SMOKE=True/RUN_FULL=False; opt in manually after prerequisite review |
| B | `week11_B_passage_representation.ipynb` | `passage_representation/` | Default RUN_SMOKE=True/RUN_FULL=False; opt in manually after prerequisite review |
| C | `week11_C_granularity_budget.ipynb` | `granularity_budget/` | Default RUN_SMOKE=True/RUN_FULL=False; opt in manually after prerequisite review |
| D | `week11_D_reader_granularity.ipynb` | `reader_granularity/` | Default RUN_SMOKE=True/RUN_FULL=False; opt in manually after prerequisite review |
| E | `week11_E_compressor_reader.ipynb` | `compressor_reader/` | Default RUN_SMOKE=True/RUN_FULL=False; opt in manually after prerequisite review |
| F | `week11_F_retriever_robustness.ipynb` | `retriever_robustness/` | Default RUN_SMOKE=True/RUN_FULL=False; opt in manually after prerequisite review |
| G | `week11_G_hierarchical_fidelity.ipynb` | `hierarchical/` | Default RUN_SMOKE=True/RUN_FULL=False; opt in manually after prerequisite review |
| H | `week11_H_oracle_and_interactions.ipynb` | `oracle/` | Saved-result analysis; no generation |
| I | `week11_I_final_report.ipynb` | `final_report/` | Saved-result analysis; no generation |

`week11_MASTER.ipynb` is an additional read-only dashboard. It displays status, expected/completed counts, configuration, resources, setup/cache tables and current completed figures. It never executes the study.

## DataFrame and plot inventory

### 0 — setup

Tables: `dataset_summary_df`, `answer_type_df`, `document_length_df`, `hierarchy_stats_df`, `model_registry_df`, `cache_status_df`.

Plots: document_length, answer_types, unit_lengths, units_per_document.

### A — retrieval

Tables: `retrieval_summary_df`, `recall_at_k_df`, `evidence_coverage_df`, `retrieval_cost_df`, `retrieval_pairwise_df`.

Plots: recall_curves, any_gold, coverage_tokens, retriever_granularity, first_rank, redundancy.

### B — passage_representation

Tables: `passage_representation_summary_df`, `ranking_agreement_df`, `recall_comparison_df`, `topk_overlap_df`, `per_query_representation_df`.

Plots: pooling_recall, pooling_mrr, pooling_overlap, weighted_minus_direct, length_pooling_gain.

### C — granularity_budget

Tables: `c_summary_df`, `c_quality_df`, `c_efficiency_df`, `c_help_hurt_df`, `c_pareto_df`, `c_pairwise_bootstrap_df`, `c_per_query_df`.

Plots: quality_budget, quality_evidence, quality_prompt, granularity_budget, granularity_selector, help_hurt, pareto, quality_per_1000, actual_requested, dataset_quality.

### D — reader_granularity

Tables: `reader_granularity_summary_df`, `reader_quality_df`, `reader_help_hurt_df`, `reader_efficiency_df`, `reader_pairwise_df`, `reader_interaction_df`, `reader_per_query_df`.

Plots: reader_granularity, reader_budget, granularity_budget, reader_help_hurt, reader_efficiency, reader_ranking, reader_dataset_rank, difference_from_mistral.

### E — compressor_reader

Tables: `compressor_reader_summary_df`, `compressor_quality_df`, `compressor_efficiency_df`, `compressor_pairwise_df`, `compressor_interaction_df`.

Plots: compressor_reader, compression_quality, compression_prompt, compression_hurt, reader_compressor_rank, compressor_pareto.

### F — retriever_robustness

Tables: `retriever_robustness_summary_df`, `retriever_interaction_df`, `retriever_pairwise_df`.

Plots: retriever_finalist, recall_downstream, retriever_pareto, relative_bm25.

### G — hierarchical

Tables: `hierarchy_summary_df`, `hierarchy_budget_allocation_df`, `hierarchy_quality_df`, `hierarchy_help_hurt_df`, `hierarchy_pairwise_df`, `hierarchy_oracle_df`.

Plots: hierarchy_fixed, hierarchy_allocation, hierarchy_context, hierarchy_help_hurt, hierarchy_pareto, hierarchy_dataset.

### H — oracle

Tables: `oracle_summary_df`, `oracle_action_distribution_df`, `oracle_gain_concentration_df`, `per_query_oracle_df`, `interaction_summary_df`, `factorial_effects_df`.

Plots: oracle_fixed, oracle_actions, gain_concentration, dataset_oracle, reader_oracle, dataset_granularity, reader_granularity, factorial_effects.

### I — final_report

Tables: `master_summary_df`, `best_by_dataset_df`, `best_by_reader_df`, `best_by_budget_df`, `best_by_granularity_df`, `pareto_front_df`, `statistical_comparisons_df`.

Plots: fig01_retrieval, fig02_budget, fig03_reader_granularity, fig04_dataset_granularity_rank, fig05_pareto, fig06_help_hurt, fig07_retriever_granularity, fig08_compressor_reader, fig09_oracle_actions, fig10_oracle_concentration, fig11_hierarchy_fixed, fig12_project_trajectory.

Every notebook also displays configuration and run status. Each major figure displays its underlying DataFrame, saves source CSV, PNG and PDF, and labels synthetic smoke. Recall curves are faceted by granularity; pooled descriptive means name the averaged factors in the title. Final reader/compressor plots use their matched experimental blocks rather than averaging incompatible phase settings. Unavailable data is left pending, never drawn as fabricated zeros. Figure12 supports reviewed historical W3/W6/W8/W9/W10/W11 records in separate protocol panels; populate `history_trajectory_records` explicitly rather than invent or pool old metrics.

## Analysis protocol

QA HELP/HURT uses paired task-score differences >+0.01 / <−0.01 versus the same-reader no-evidence control. Verification uses wrong→correct / correct→wrong. Undefined metrics are null. Verification reports strict label-format errors, accuracy, macro-F1 and class metrics; QA reports token F1 and EM. No combined raw QA-F1/SciFact-accuracy “overall F1” is produced.

Paired bootstrap:10,000 resamples, seed42; QASPER paper clusters, Hotpot query and SciFact claim clusters. Comparisons are labelled primary/secondary/exploratory. Main factorial effects and interaction cells are descriptive within datasets; dataset summaries use normalized baseline gains/ranks, not pooled raw task scores. Explicit four-cell difference-in-differences contrasts can be supplied in config, with paired bootstrap. No unstable mixed-effects/ANOVA assumptions are forced.

H requires paired eligible surfaces for exact/epsilon oracle (QA epsilon0.01, verification exact ties). Cost breaks ties deterministically. It saves action distributions, top1/5/10/20% gain concentration, cumulative curves and OCR when its denominator is nontrivial. Global within-dataset and within-reader oracles are separate. Incomplete pairing is reported, never silently complete-case filtered. Oracles are retrospective, not deployable methods. W10 seed diagnostics are read-only hooks with a distinct protocol label.

## Resources: configurations and answer-row upper bounds

These are work counts, not runtime predictions. No full W11 rate has been measured. Identical-text response reuse can substantially reduce physical calls.

| phase   | dataset   |   configured_queries |   planned_configurations |   executable_configurations |   pending_configurations |   answer_rows_upper_bound |   executable_answer_rows_upper_bound | new_lm_calls                                                                          | actual_time_estimate                        |
|:--------|:----------|---------------------:|-------------------------:|----------------------------:|-------------------------:|--------------------------:|-------------------------------------:|:--------------------------------------------------------------------------------------|:--------------------------------------------|
| 0       | QASPER    |                   50 |                        0 |                           0 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| 0       | HotpotQA  |                   50 |                        0 |                           0 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| 0       | SciFact   |                   50 |                        0 |                           0 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| A       | QASPER    |                   50 |                        9 |                           9 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| A       | HotpotQA  |                   50 |                        9 |                           9 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| A       | SciFact   |                   50 |                        9 |                           9 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| B       | QASPER    |                   50 |                        4 |                           4 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| B       | HotpotQA  |                   50 |                        4 |                           4 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| B       | SciFact   |                   50 |                        4 |                           4 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| C       | QASPER    |                   50 |                       41 |                          37 |                        4 |                      2050 |                                 1850 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| C       | HotpotQA  |                   50 |                       41 |                          37 |                        4 |                      2050 |                                 1850 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| C       | SciFact   |                   50 |                       41 |                          37 |                        4 |                      2050 |                                 1850 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| D       | QASPER    |                   50 |                      123 |                         111 |                       12 |                      6150 |                                 5550 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| D       | HotpotQA  |                   50 |                      123 |                         111 |                       12 |                      6150 |                                 5550 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| D       | SciFact   |                   50 |                      123 |                         111 |                       12 |                      6150 |                                 5550 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| E       | QASPER    |                   50 |                       41 |                          29 |                       12 |                      2050 |                                 1450 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| E       | HotpotQA  |                   50 |                       39 |                          27 |                       12 |                      1950 |                                 1350 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| E       | SciFact   |                   50 |                       39 |                          27 |                       12 |                      1950 |                                 1350 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| F       | QASPER    |                   50 |                        0 |                           0 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| F       | HotpotQA  |                   50 |                        0 |                           0 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| F       | SciFact   |                   50 |                        0 |                           0 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| G       | QASPER    |                   50 |                       25 |                          21 |                        4 |                      1250 |                                 1050 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| G       | HotpotQA  |                   50 |                       25 |                          21 |                        4 |                      1250 |                                 1050 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| G       | SciFact   |                   50 |                       25 |                          21 |                        4 |                      1250 |                                 1050 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| H       | QASPER    |                   50 |                        0 |                           0 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| H       | HotpotQA  |                   50 |                        0 |                           0 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| H       | SciFact   |                   50 |                        0 |                           0 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| I       | QASPER    |                   50 |                        0 |                           0 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| I       | HotpotQA  |                   50 |                        0 |                           0 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |
| I       | SciFact   |                   50 |                        0 |                           0 |                        0 |                         0 |                                    0 | <= answer rows; identical evidence + reader outputs reused; D reuses primary C reader | NOT ESTIMATED; no full W11 timings measured |

A/B use retrieval/embedding only. Setup and H/I do not generate answers. F has no configurations until manually supplied. Pending C1/H4 conditions are separately counted. Default150 queries give C6150 configured answer rows (5550 executable without token), D18450 (16650 executable; Qwen rows reusable from C), E5950 (4150 executable, including the specialist own-baseline control), and G3750 (3150 executable). Baseline/duplicate evidence results can be reused; these are upper bounds, not promised physical call counts.

## Validation status

Safe-default notebook checks: 0=PASS, A=PASS, B=PASS, C=PASS, D=PASS, E=PASS, F=PASS, G=PASS, H=PASS, I=PASS, MASTER=PASS.

{
  "analysis_plot_validation": [
    {
      "all_figures_present": true,
      "figure_types": 8,
      "figures": 12,
      "full": false,
      "model_calls": 0,
      "phase": "H",
      "synthetic": true,
      "tables": 14
    },
    {
      "all_figures_present": true,
      "figure_types": 12,
      "figures": 26,
      "full": false,
      "model_calls": 0,
      "phase": "I",
      "synthetic": true,
      "tables": 25
    }
  ],
  "branch": "gagan",
  "default_notebooks_prefer_saved_real_results": true,
  "default_queries_per_dataset": 50,
  "external_datasets_downloaded": false,
  "full_started": false,
  "historical_bytes_unchanged": true,
  "historical_files_checked": 36549,
  "manual_order": [
    "0",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I"
  ],
  "qasper_test_used": false,
  "real_model_generation": false,
  "real_model_loading_tested": false,
  "safe_notebooks_passed": 11,
  "source_identity": "fe7a9e38c2cefefb948455c04ccf0658e2cf4bc39a03f0605c894a11aaaf3298",
  "timestamp": "2026-10-27T16:30:08.493811+00:00",
  "token_compressor": "PENDING_DEPENDENCY_AND_QUALIFICATION",
  "unit_test_seconds": 84.88,
  "unit_tests_passed": 84,
  "visual_checks": [
    "hierarchy_vs_fixed",
    "six_panel_trajectory"
  ]
}

## Remaining limits

- No real Week11 model generation or full dataset run was executed by the agent.
- LLMLingua-2 model qualification and full TOKEN/H4 support remain pending; faithful learned compression remains future work.
- External development files are prepared and byte-locked before outcomes. The failed CMU Hotpot host was replaced by a recorded, pinned HotpotQA-maintained Hugging Face source; this changes the file serialization, not the selected dataset setting.
- Exact source grounding is not semantic faithfulness or answer correctness.
- Strict-prefix budgets can leave unused capacity, especially when a passage exceeds the budget; this is a declared condition, not a bug or silent closest-fit substitution.
- Default50-query samples characterize these splits; they do not establish fresh confirmation or universal superiority.
- No winner, significance, controller success or research conclusion can be inferred from fabricated smoke data.
