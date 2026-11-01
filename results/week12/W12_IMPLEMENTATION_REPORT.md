# Week 12 implementation and smoke report

**The framework is ready for manual development runs. No full experiment, full training, confirmation evaluation or QASPER TEST evaluation was started.** Week13 confirmation remains blocked pending provenance certification and a final method freeze.

## Branch, files and preservation

Branch: `gagan`. Implementation checkout when this report was built: `53a51357c009d2f7ee447b22fee5d1ed4665b98e`. Runtime code hash: `37e44de2c40ca58e102ac775cfe6a0e14e1fd0e034c9bf42a0462fa512a4296c`. Model/source/package identities are recorded in qualification receipts and immutable result stores.

- `configs/week12/README.md`
- `configs/week12/protocol.yaml`
- `notebooks/week12/README.md`
- `notebooks/week12/week12_0_protocol_and_split_audit.ipynb`
- `notebooks/week12/week12_A_qbgf_algorithm.ipynb`
- `notebooks/week12/week12_B_baseline_qualification.ipynb`
- `notebooks/week12/week12_C_fixed_policy_development.ipynb`
- `notebooks/week12/week12_D_qbgf_ablations.ipynb`
- `notebooks/week12/week12_E_llmlingua2.ipynb`
- `notebooks/week12/week12_F_lora_training.ipynb`
- `notebooks/week12/week12_G_lora_ablation.ipynb`
- `notebooks/week12/week12_H_analysis.ipynb`
- `notebooks/week12/week12_I_week13_freeze.ipynb`
- `operations/week12/build_report.py`
- `operations/week12/verify_smokes.py`
- `scripts/week12/build_notebooks.py`
- `scripts/week12/prepare_training_data.py`
- `scripts/week12/qualify_lora.py`
- `scripts/week12/run.py`
- `scripts/week12/validate_notebooks.py`
- `src/week12/__init__.py`
- `src/week12/algorithm.py`
- `src/week12/analysis.py`
- `src/week12/canonical.py`
- `src/week12/data.py`
- `src/week12/laboratory.py`
- `src/week12/lora.py`
- `src/week12/protocol.py`
- `src/week12/qualification.py`
- `src/week12/reader.py`
- `src/week12/runner.py`
- `src/week12/statistics.py`
- `tests/week12/test_framework.py`

Historical byte check: **2,359 tracked W0–W11 files unchanged**, including the user's pre-existing notebook outputs/flags. Historical frozen gates still pass. New caches/results are under Week12 only; historical SFR embeddings are reused through a read-only SQLite connection. No dependency was installed into the historical environment and no historical model cache was deleted.

## QBGF definition and agreed corrections

N=10, tau_p=tau_s=5, budgets 256/1024, pinned SFR and pinned Qwen3-8B. Passage softmax weights multiply query-ranked sentence relevance mass. DROP/SENT1/2/4/8/FULL form exact multiple-choice knapsack options. DP optimizes this relevance surrogate, **not true F1 or causal support**. Utility ties at1e-12 prefer fewer tokens, then lexicographic option indices. Oversized FULL is merely infeasible. Exact source sentences are selected by relevance and exposed in original source order. Overlapping candidate source regions fail before optimization; equivalent k options are deduplicated.

The implemented objective is `R_i = softmax(5 r^P)_i`, `a_ij = softmax(5 r_i^S)_j`, and `U_i(SENT_k) = R_i * sum(a_ij for j in top-k)`. FULL has utility `R_i`; DROP has zero. The DP chooses one option per passage to maximize total utility subject to `sum(option_token_cost) <= B`.

Costs include source separators in additive Qwen token-ID blocks. The reader consumes the exact assembled IDs; it never retokenizes the joined evidence and silently changes the budget. Each QBGF result checks sum(option costs)=actual evidence-ID count<=B. Prospective matched controls share this serialization. RAW_STRICT separately reproduces historical whole-text/max-three-tokenizer strict-prefix allocation on the W12 pool; RAW_SKIP uses skip-to-fit. Old W11 numeric scores are never reused as matched W12 outcomes.

W12-C is **development/validation**, not confirmation. Method selection occurs on development only, before a future Week13 locked evaluation. No Week12 entrypoint evaluates reserved confirmation examples.

## Baseline inventory

| id                         | category              | reader               | status                                        | checkpoint                                                                                                          | budget                                                | qualification_receipt                                                                                             |   k |   n |   max_new_tokens |
|:---------------------------|:----------------------|:---------------------|:----------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------|----:|----:|-----------------:|
| NONE                       | canonical_control     | Qwen                 | IMPLEMENTED                                   |                                                                                                                     |                                                       |                                                                                                                   |     |     |                  |
| CANONICAL_RAG_BM25         | external              | adapted Mistral RAG  | REAL_SMOKE_PASSED                             | /home/gunavenkat/Downloads/CS787-RAG-Project/Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42    | natural original top10; separate specialist reference | /home/gunavenkat/Downloads/CS787-RAG-Project/results/week12/qbgf/v1/qualification/CANONICAL_RAG_BM25.json         |     |     |                  |
| SARA_CANONICAL_BM25_K5_N10 | external              | adapted Mistral SARA | REAL_SMOKE_PASSED                             | /home/gunavenkat/Downloads/CS787-RAG-Project/Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42 |                                                       | /home/gunavenkat/Downloads/CS787-RAG-Project/results/week12/qbgf/v1/qualification/SARA_CANONICAL_BM25_K5_N10.json |   5 |  10 |               64 |
| STRONG_SFR_RAG             | external_control      | Qwen                 | IMPLEMENTED_NATURAL_TOP10_REFERENCE           |                                                                                                                     |                                                       |                                                                                                                   |     |     |                  |
| LLMLINGUA2                 | external              | Qwen                 | QUALIFICATION_REQUIRED                        |                                                                                                                     |                                                       |                                                                                                                   |     |     |                  |
| SARA_SFR_K5                | historical_specialist |                      | NOT_QUALIFIED_HISTORICAL_POOL_PARITY_REQUIRED |                                                                                                                     |                                                       |                                                                                                                   |     |     |                  |
| SARA_SFR_K6_HISTORICAL     | historical_specialist |                      | NOT_QUALIFIED_HISTORICAL_POOL_PARITY_REQUIRED |                                                                                                                     |                                                       |                                                                                                                   |     |     |                  |
| FIXED_SENTENCE             | project_baseline      |                      | IMPLEMENTED_NEW_PROTOCOL_PORT                 |                                                                                                                     |                                                       |                                                                                                                   |     |     |                  |
| C2                         | project_baseline      |                      | IMPLEMENTED_NEW_PROTOCOL_PORT                 |                                                                                                                     |                                                       |                                                                                                                   |     |     |                  |
| H3                         | project_baseline      |                      | IMPLEMENTED_NEW_PROTOCOL_PORT                 |                                                                                                                     |                                                       |                                                                                                                   |     |     |                  |
| RAW_STRICT                 | project_baseline      |                      | IMPLEMENTED_NEW_PROTOCOL_PORT                 |                                                                                                                     |                                                       |                                                                                                                   |     |     |                  |
| RAW_SKIP                   | project_baseline      |                      | IMPLEMENTED_NEW_PROTOCOL_PORT                 |                                                                                                                     |                                                       |                                                                                                                   |     |     |                  |
| QBGF_NO_QUERY              | QBGF_ablation         |                      | IMPLEMENTED                                   |                                                                                                                     |                                                       |                                                                                                                   |     |     |                  |
| QBGF_SENTENCE_ONLY         | QBGF_ablation         |                      | IMPLEMENTED                                   |                                                                                                                     |                                                       |                                                                                                                   |     |     |                  |
| QBGF_PASSAGE_ONLY          | QBGF_ablation         |                      | IMPLEMENTED                                   |                                                                                                                     |                                                       |                                                                                                                   |     |     |                  |
| QBGF_NO_MASS               | QBGF_ablation         |                      | IMPLEMENTED                                   |                                                                                                                     |                                                       |                                                                                                                   |     |     |                  |
| QBGF_GREEDY                | QBGF_ablation         |                      | IMPLEMENTED                                   |                                                                                                                     |                                                       |                                                                                                                   |     |     |                  |
| QBGF_BM25                  | QBGF_ablation         |                      | IMPLEMENTED                                   |                                                                                                                     |                                                       |                                                                                                                   |     |     |                  |

Canonical SARA executes original BM25 top-10/chunk-256, k5 full text plus five SFR vectors, trained Mistral checkpoint, historical BF16 prompt/interface and 64-token greedy answer cap. The original full-text RAG reference uses its own r16 adapted checkpoint and all ten textual passages. They remain specialist references, separate from public Qwen; their reader/checkpoint/prompt/cap assumptions do not justify a bare “QBGF beats SARA” statement. Canonical source/input/checkpoint hashes and execution smoke receipts are saved.

SARA_SFR_K5 and SARA_SFR_K6_HISTORICAL are explicitly unqualified until historical pool/prompt parity is established. They cannot be silently relabelled as original SARA. Natural SFR/BM25 Qwen references and budgeted controls have distinct names.

## True ablations

- NO_QUERY: uniform passage and sentence weights, source-order sentence choices, **within the same query-retrieved pool**. Tests verify score mutation does not affect these options. It tests allocation conditioning, not removal of retrieval conditioning.
- SENTENCE_ONLY: no FULL choice; PASSAGE_ONLY: DROP/FULL only.
- NO_MASS: uniform min(k,m)/m coverage, retaining query-ranked sentence selection. It does not collapse every non-DROP option to the same utility.
- GREEDY: deterministic feasible marginal utility/token upgrades; toy tests distinguish it from exact DP.
- BM25: secondary retrieval-plus-relevance sensitivity, not an isolated granularity ablation.

## Split audit and remaining provenance work

Development: 50 questions each from the explicit historical W11 cohort. This is reused development data, never fresh confirmation.

Reserved candidate confirmation counts: {'HotpotQA': 200, 'QASPER': 100, 'SciFact': 100}. They exclude known pilot question hashes, and SciFact reservation excludes connected cited-paper components touched by pilot claims. **These reservations are not certified fresh:** W0–W10 historical exposure still requires exhaustive review. No fresh QASPER confirmation is claimed, and QASPER TEST stays permanently closed. The complete lists/hashes are in `protocol/cohorts.json`, `split_audit.json` and `W12_CONFIRMATION_FREEZE.json`; the concise human audit is `results/week12/W12_SPLIT_AUDIT.md`.

TRAIN readiness: {'HotpotQA': False, 'QASPER': True, 'SciFact': False}. HotpotQA/SciFact TRAIN must be prepared explicitly. Full F creates the actual filtered membership manifest, excludes development/reserved question hashes, excludes historical QASPER DEV papers, and uses connected SciFact cited-paper groups. Normalized duplicate HotpotQA questions share a train/validation group even when their IDs differ. Train/validation leakage is checked before model loading. Historical TRAIN reuse is allowed; historical validation examples are not treated as unseen data. Token length caps fail rather than silently truncating training targets.

## LoRA readiness and decomposition

The default F smoke trains a random miniature Qwen adapter for two CPU steps, checking frozen base weights and changed adapter weights. Separately the pinned **real Qwen3-8B NF4** completed one optimizer step on a synthetic string: adapter changed, frozen base-parameter samples stayed unchanged, and no adapter was saved. Peak allocated VRAM was approximately 9.14 GiB. This qualifies pipeline compatibility, not model quality or full-training resource feasibility.

Manual full training uses dataset-balanced sampling, r8/alpha16/dropout0.05, q/v projections, one epoch and no hyperparameter search. Optimizer checkpoints resume. Corpus preparation resumes per query/mode/budget shard. BASE_NF4 and LORA_NF4 use identical source evidence; G verifies evidence hashes against C and verifies the adapter checksum. The FP16 main surface is not used as the causal QLoRA baseline.

The mandatory real 2×2 is RAW_SKIP/QBGF × BASE_NF4/LORA_NF4. Analysis reports selection, RAW adaptation, QBGF adaptation and combined effects, plus the paired interaction `(LoRA_QBGF−Base_QBGF)−(LoRA_RAW−Base_RAW)` with bootstrap CI. Full training is still disabled and unexecuted.

## LLMLingua-2

Status: **NOT_QUALIFIED**. Reason: `MISSING_DEPENDENCY; no automatic installation`. The official package is absent, the model revision is not pinned, and downloads are disabled by default. No token-compression result was fabricated, no model substituted, and no full LLMLingua experiment ran. Resource inventory records disk/HF cache/RAM/VRAM/package versions and estimated checkpoint footprint before any attempted installation/download. The adapter follows the official Microsoft API: [package](https://github.com/microsoft/LLMLingua), [checkpoint](https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank).

To add it later, install/qualify separately, pin the checkpoint revision before outcomes, and use a new configured output version if the scientific configuration changes. Its own tokenizer target is not accepted as a Qwen budget guarantee; the adapter validates actual Qwen tokens and fails without silent truncation if its predefined retries cannot fit. Token deletion is not called exact contiguous-span grounding or native query conditioning.

## Tests and smoke evidence

- **86 unit/integration tests passed.** Includes DP versus brute force, greedy distinction, grounding, source order, no-query score independence, strict/skip behavior, macro-F1 resampling, Holm, split guards, train/val leakage refusal before model load, source identity invalidation and the synthetic 2×2 schema.
- **10/10 notebooks executed successfully with their original safe defaults**; copies stay in ignored smoke outputs.
- Real model smoke uses **three unique development queries total**, reused across qualification stages. No confirmation query or TEST record was used. Repeated smoke checks after fixes are not independent research trials.

| dataset   | query_id                 |   evidence_tokens |   budget |   unused |   source_support |   prompt_tokens | generation_cap_hit   | format_compliant   |
|:----------|:-------------------------|------------------:|---------:|---------:|-----------------:|----------------:|:---------------------|:-------------------|
| HotpotQA  | 5abd01205542992ac4f3819b |               249 |      256 |        7 |                1 |             306 | False                |                    |
| QASPER    | 2102                     |               255 |      256 |        1 |                1 |             296 | False                |                    |
| SciFact   | 3                        |               254 |      256 |        2 |                1 |             326 | False                | True               |

All three selections passed exact source validation and restored source ordering; budget checks use actual input IDs. Smoke answer scores are not a quality result and did not tune the algorithm. One sampled QASPER question has an existing annotation-mapping warning, preserved as such.

Canonical SARA and canonical RAG each completed one QASPER execution smoke on that same QASPER query. Real QLoRA qualification used synthetic data only. Resume verification reran the real QBGF smoke and confirmed completed result shards were byte-identical; no completed answers were regenerated.

## Metrics, statistics and figures

QA F1/EM remain separate from SciFact macro-F1/accuracy. SciFact trims whitespace and accepts exact normalized-adapter labels only; invalid explanations/case changes remain invalid. Macro-F1 is recomputed from predictions/labels in every cluster bootstrap. QASPER clusters by paper; SciFact uses connected cited-document components; Hotpot uses normalized-question clusters. Classification HELP/HURT uses correctness transitions, not per-query macro-F1.

The 39-member primary family is fixed before outcomes across dataset/budget/contrast. Paired bootstrap: 10,000, seed 42. Raw two-sided cluster label-swap randomization p-values state the exchangeability assumption. Holm includes reserved optional hypotheses conservatively at p=1 internally; missing tests remain pending and never acquire invented observed p-values. LoRA precision groups remain separate in Pareto comparisons. No mixed raw dataset score is produced.

Saved analysis includes paired primary tests/CI, per-class SciFact metrics, evidence mapping metrics, answer-type breakdowns, HELP/HURT, allocation, unused-budget/compression tables, Pareto and LoRA decomposition. Plot scaffolding covers the twelve requested views and a paired-CI panel, with PNG/PDF and underlying CSV. Missing LLMLingua or adapted outcomes are not fabricated. End-to-end latency remains null unless measured; no latency superiority is claimed.

## Manual notebook order and workload

See `notebooks/week12/README.md`. Order: 0 → A → B → C → D → optional E/F/G → H → I.

- **Run All defaults remain smoke-only**. For C/D/G change `RUN_FULL=True`; the next cell derives effective_smoke=False. Leave RESUME=True, OVERWRITE=False.
- E requires qualification before `RUN_LLMLINGUA_FULL=True`; download stays opt-in.
- F requires explicit TRAIN preparation before `RUN_FULL_LORA_TRAIN=True`. It displays actual dataset/mode/length/group counts before training.
- H/I: `RUN_SMOKE=False` reads real saved development results only. I needs explicit FINAL_METHOD/FINAL_REASON; there is no automatic winner selection or Week13 evaluation.

Estimated logical development rows: C 2,400 + 100 separate canonical QASPER references; D 2,100 (300 overlap C); optional E 300; optional G 1,200. These are not physical-call counts or independent questions. Training cap is 500 source questions per dataset before filtering; the actual corpus must be inspected in F. Full execution time is not estimated from tiny smoke timings.

## Go/no-go and remaining risks

**GO for manual base-Qwen development C/D after reviewing notebook0/B.** Actual counts, source validation, pinned model loading, canonical smoke, default execution and resume checks pass.

**Conditional:** F/G await official TRAIN files and full user-run training; E is unqualified; SFR-SARA variants await faithful historical adapters. **NO-GO for Week13 confirmation now:** certify historical exposure/group disjointness, review the final development comparison and freeze the final method first. Available disk is about 20 GiB at qualification, so full training/checkpoint storage still needs a separate capacity check. A natural full-text reference that exceeds context fails loudly; it is never silently truncated. A 50-query development pilot may not support the conservative noninferiority gates; retaining fixed sentences is an allowed outcome.

No full runs were started. No claim that QBGF, LoRA or fusion improves quality follows from smoke tests.
