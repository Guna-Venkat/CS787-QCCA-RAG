# Week 11 — completed multigranular evidence pilot

**All available phases 0–I are complete, audited and frozen on branch gagan. No further full run is active.** Optional LLMLingua-2/token conditions remain pending; this is completion of the available text study, not completion of those missing methods.

## Study scope and integrity

The frozen pilot uses 50 deterministic development queries per dataset (150 total, seed 42): QASPER project DEV, HotpotQA distractor DEV, SciFact DEV. It does not use the full datasets, and Week 11 does not read QASPER TEST. QASPER has 40 paper clusters among 50 queries; other datasets have 50 query/claim clusters. Readers are the pinned Mistral-7B-Instruct-v0.2, Qwen3-8B and Llama-3.1-8B-Instruct. Every reader completed its available assigned surface; no Llama placeholders remain.

There are 31,900 saved logical generation rows across C–G. Cross-phase reuse deduplicates to 25,300 unique dataset/query/action observations, 506 complete dataset/condition summaries, each with 50 outcomes. Duplicate predictions, utility, evidence, model and source identities agree exactly. These are many repeated conditions on 150 questions, not 25,300 independent questions. Retrieval rows A/B are separate from answer-generation rows.

## What each phase tested and observed

| Part | Work completed | Result |
|---|---|---|
| 0 | Dataset, hierarchy, reader and source qualification | Frozen 50-per-dataset cohort and pinned interfaces |
| A | BM25/MiniLM/SFR retrieval across passage/sentence/proposition units; 1,350 rows | SFR/P0 retained as controlled primary; no universal retriever winner |
| B | P0 passage embedding vs three sentence-pooling variants; 600 rows | Pooling gains inconclusive on this pilot; retain P0 |
| C | Qwen granularity × selector × evidence budget; 5,550 available rows | No universal optimal unit; compact units avoid some tight-budget failures |
| D | Same C evidence under all three readers; 16,650 rows | Reader-specific point patterns; SciFact format compliance strongly confounds two readers |
| E | Raw C0 vs exact top-eight sentences C2, plus QASPER SARA specialist; 4,150 rows | Some C2 tight-budget gains are confounded by raw passage fit; SARA remains a separate specialist |
| F | Five manually reviewed configurations × three retrievers, Qwen; 2,400 rows | Small QASPER variation, some SciFact sensitivity, Hotpot provided-pool ceiling |
| G | Five deterministic hierarchy exposures × four budgets, Qwen/SFR; 3,150 rows | H3 beats raw at 256; advantage over simple sentence selection unconfirmed |
| H | Saved-result oracles and 12 named paired interactions | Budget-dependent representation signal; oracle headroom is retrospective |
| I | Deduplicated master tables, figures, specialist/public separation and final synthesis | Available study complete; no universal winner or trained controller established |

## Best observed public-reader points

Each row below is selected retrospectively over that reader’s available action surface on the same pilot. These are descriptive peaks, not a fair identical-configuration reader leaderboard. QA uses F1; SciFact uses strict accuracy. Budget FULL is capped context, not unlimited whole-corpus input.

| dataset   | reader   | retriever   | evidence_granularity   | selector   | budget   |   task_score |   prompt_tokens |
|:----------|:---------|:------------|:-----------------------|:-----------|:---------|-------------:|----------------:|
| HotpotQA  | Llama    | SFR         | PROPOSITION            | hybrid     | FULL     |     0.443019 |     1370.640000 |
| HotpotQA  | Mistral  | SFR         | PASSAGE                | semantic   | 512      |     0.324281 |      493.060000 |
| HotpotQA  | Qwen     | SFR         | HIERARCHICAL           | semantic   | 1024     |     0.487169 |      917.060000 |
| QASPER    | Llama    | SFR         | SENTENCE               | lexical    | FULL     |     0.289146 |     4144.400000 |
| QASPER    | Mistral  | SFR         | SENTENCE               | lexical    | 1024     |     0.200750 |     1044.020000 |
| QASPER    | Qwen     | BM25        | PASSAGE                | semantic   | FULL     |     0.249211 |     4186.440000 |
| SciFact   | Llama    | SFR         | PASSAGE                | semantic   | FULL     |     0.040000 |     4900.320000 |
| SciFact   | Mistral  | SFR         | PASSAGE                | lexical    | 256      |     0.000000 |       83.220000 |
| SciFact   | Qwen     | SFR         | PASSAGE                | hybrid     | 1024     |     0.780000 |      689.020000 |

The QASPER adapted SARA specialist is separate:

| dataset   | reader   | retriever   | evidence_granularity   | selector   | budget   |   task_score |   prompt_tokens | reader_variant               |
|:----------|:---------|:------------|:-----------------------|:-----------|:---------|-------------:|----------------:|:-----------------------------|
| QASPER    | Mistral  | SFR         | PASSAGE                | semantic   | FULL     |     0.403749 |     1215.140000 | MISTRAL_ONLY_LATENT_BASELINE |

Its own adapted no-evidence baseline is 0.190914 F1; intervention reaches 0.403749, paired paper-cluster delta +0.212834, nominal CI [0.118354, 0.304464]. This is a valid within-specialist policy gain. It does not identify a latent-only or compression-only causal gain, because adaptation, prompt and interface differ from public readers.

## Claim boundaries and experimental-design confounds

The table below states the permitted interpretation explicitly. Some cautions were already present in the detailed review; this revision makes the summary equally precise. None changes the frozen scores or retrospective comparisons.

| Claim to avoid | Supported wording and limitation |
|---|---|
| Sentences/propositions are generally better than passages | The preferred exposed unit depends on task, reader, requested budget and allocation policy. No universal granularity winner is established. |
| H3 is better than sentence selection | H3/256 improves over raw H0 under the frozen strict-prefix allocator. Its matched fixed-sentence comparisons have zero-containing intervals; added value from hierarchy is unconfirmed. |
| Reader × granularity interaction is established | Reader-specific point patterns are suggestive. The explicitly matched interaction intervals include zero. |
| SFR is universally best | SFR is a frozen primary control. BM25 remains competitive; HotpotQA provided-context saturation prevents a general retriever-robustness conclusion. |
| C2 proves compression improves quality | C2/256 improves some policy outcomes, partly because coarse passages fail to fit. Strict-prefix allocation is a major experimental confound; suppression, granularity and realized evidence length are not isolated. |
| Mistral/Llama fail SciFact reasoning | Their strict scores are dominated by output-format violations and cap hits. These results cannot cleanly compare reasoning ability. |
| SARA proves latent vectors are superior | Its 0.403749 F1 is a distinct adapted specialist result; checkpoint/adaptation, projector, prompt and interface differ. The latent contribution is not isolated. |
| Audits prove no implementation bug exists | Checksum, pairing and provenance audits support integrity of the checked data flow. No direct implementation bug is demonstrated by these reports; audits do not exhaustively rule out bugs. Most identified limitations concern experimental design and interpretation. |

QASPER annotation-mapping gaps, the 50-query-per-dataset development pilot, many unadjusted exploratory comparisons, C2's top-eight saturation, and unexecuted LLMLingua-2 conditions remain explicit flags. A zero-containing interval is not evidence of equivalence. Exact source matching does not establish factual support or answer correctness.


## Supported findings and unresolved claims

**1. Budget fit matters.** Qwen C2 at 256 beats same-budget raw C0 on QASPER by +0.109351 F1 (CI [0.046333, 0.171307]) and SciFact by +0.32 accuracy ([0.14, 0.50]). HotpotQA +0.100422 remains inconclusive ([−0.005202, 0.210173]). Some raw prefixes are empty because the first passage does not fit; these are not length-matched experiments. With larger budgets C2 is not consistently better, and top-eight sentence saturation explains identical 1024/FULL exposures.

**2. Hierarchy is not yet necessary.** H3/256 versus H0 gains have positive nominal intervals on all three datasets: QASPER +0.085380, HotpotQA +0.099869, SciFact +0.32. Against fixed semantic sentences at 256, every interval includes zero. The HotpotQA H3/1024 peak 0.487169 exceeds raw 0.480272 by only +0.006897, CI [−0.051911, 0.068590]. Compact source exposure is promising; the added value of mixing granularities is unproven.

**3. A reader × granularity interaction is not established.** Llama’s best QASPER point is 0.289146 and Qwen’s best HotpotQA point is 0.487169. The explicitly matched reader-by-granularity interactions have zero-containing intervals. Different point-estimate winners or significance in one group alone do not prove an interaction.

**4. Retriever robustness is constrained by the evaluation setting.** In F, HotpotQA all retrievers include all 10 supplied paragraphs under width 20, producing identical downstream evidence/predictions. QASPER retriever differences are small. SciFact BM25 passage/1024 is 0.76 vs SFR 0.68, delta CI [0.00, 0.18]; this does not establish a positive lower bound. The one strict nonzero F interval is SFR versus MiniLM at compact hybrid sentences, one exploratory finding among 45 contrasts.

**5. Retrospective routing headroom exists; a controller result does not.** Within Qwen, best fixed vs exact oracle is HotpotQA 0.487169→0.612285, QASPER 0.249211→0.373358, SciFact 0.78→0.96. Oracle choices use true outcomes after generation. They do not show that inference-time features can recover that gain, nor justify automatically training an elaborate controller.

## Flags that must accompany the results

| phase   |   rows |   physical_calls |   empty_evidence |   generation_cap |   invalid_labels |   mapping_rows |
|:--------|-------:|-----------------:|-----------------:|-----------------:|-----------------:|---------------:|
| C       |   5550 |             3506 |              308 |                0 |                0 |            259 |
| D       |  16650 |             7012 |              924 |              987 |             3675 |            777 |
| E       |   4150 |              850 |              297 |              214 |              900 |            203 |
| F       |   2400 |              519 |                0 |                0 |                0 |            112 |
| G       |   3150 |             1863 |              297 |                0 |                0 |            147 |

Flag counts are logical condition rows and overlap across phases/shared cached observations. They must not be added as independent failed questions. The same seven QASPER queries have incomplete evidence-annotation mapping; two other sampled QASPER queries have no gold annotations. SciFact evidence-recall denominators exclude the 25 unannotated/no-evidence claims. Main answer utilities retain all 50 queries.

The largest problem is strict SciFact format compliance: D has 3,675 invalid-label rows; E has 900. Mistral and Llama frequently emit explanatory prose instead of exactly SUPPORT/REFUTE/NOT_ENOUGH_INFO, and often hit the generation cap. Their near-zero strict accuracy cannot be interpreted as a clean reasoning comparison. Separate anchored leading-label diagnostics are explicitly post-hoc; primary scores were never silently reparsed or replaced. Qwen F/G has no invalid labels or cap hits.

Strict allocation stops when the next ranked unit cannot fit, rather than skipping it. This explains many empty raw/H1/H2 evidence blocks. Preserve these as part of v1; any skip-to-fit or alternative prompt/parser experiment belongs in a newly versioned protocol. Source-grounded spans are exact substrings, not proof of factual support or answer correctness.

## Cost and statistical limits

13,750 physical scientific generation calls served the 31,900 logical generation rows; operational qualification calls are recorded separately. Cached policy evaluations retain conceptual LM cost. Embedding preprocessing/ranking is nonzero work, and source text in memory is not counted as LM prompt input. End-to-end serving latency was not measured; no speed claim follows from partial cached timing fields.

Paired comparisons use 10,000 bootstrap resamples, seed 42, clustering by QASPER paper and other dataset query/claim. Intervals are nominal, exploratory and uncorrected for the many comparisons. The same small development cohort supports selection and analysis, so best points and oracles are optimistic. Unequal block factorial means are descriptive; QA F1 and verification accuracy are never averaged together.

## Relationship to prior research

These papers motivate hypotheses and baselines; they do not independently validate the Week 11 findings or our proposed method.

- [Dense X Retrieval](https://aclanthology.org/2024.emnlp-main.845/) studies indexing granularity and reports benefits from fine-grained retrieval under its tested budgets. Week 11 includes downstream exposure of smaller units inside a passage pool; its deterministic exact-clause “propositions” are not the paper's self-contained factoid units. This is related evidence, not a faithful reproduction or a universal superiority result.
- [ACC-RAG](https://aclanthology.org/2025.findings-emnlp.1307/) studies input-dependent compression rates. It motivates adaptation but does not establish that our small action router will outperform fixed policies.
- [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/) selects retrieval strategies according to question complexity. It supports considering query-dependent decisions; its strategy space differs from our proposed representation/budget actions.
- [SeleCom](https://arxiv.org/abs/2602.15856) motivates query-conditioned selection for soft compression. Our proposed source-text fusion is a separate mechanism; it should not be labelled a SeleCom reproduction.
- [LLMLingua-2](https://aclanthology.org/2024.findings-acl.57/) provides a task-agnostic token-classification compressor and reports cross-LLM generalization in its experiments. It remains unqualified/unexecuted in frozen Week 11. A future query-conditioned system may select source regions before this compressor; do not assume its task-agnostic scoring itself supplies query conditioning.


## Proposed two-week direction — unexecuted research plan

Investigate one **query-conditioned, budget-aware multigranular fusion policy**, with SFR retrieval and Qwen held fixed initially. This is a research hypothesis, not a validated deployment recommendation or a universal SFR/Qwen ranking. Do not replace it with another unrestricted global-k predictor.

`query + available token budget → fixed SFR candidate pool → query-conditioned feasible exposure action → Qwen`

Small budgets favor testing selected sentences and, if qualified, token compression; medium budgets test sentence/passage fusion; larger budgets test selected full passages. These are candidate actions to evaluate, not hard-coded conclusions from v1. Budget thresholds must be frozen using training/validation only. The router must never choose an action exceeding the supplied budget.

Use a small preregistered action set: query-selected sentences, LLMLingua-2 compression, selected passages, and a mixed sentence/passage exposure. Labels such as sentence-256, LLMLingua-256 and passage-1024 describe particular feasible operating points, not interchangeable options under the same 256-token budget. Retain no-intervention, raw-passage, fixed-sentence, C2 and H3 controls; compare the learned router with a simple fixed budget-based rule.

### Week 1: controls, qualification and action surface

1. Preserve v1. Create a newly versioned protocol with fixed source pools, prompts, decoding, source order and budgets. Separate representation from allocation with a preregistered skip-to-fit control and matched realized-token comparisons where feasible. Report requested versus actual lengths, empty contexts, HELP/HURT, task quality and actual costs.
2. Qualify LLMLingua-2 if disk, dependencies and GPU memory permit: pin the model/revision and package versions, check actual tokenizer budgets, deterministic behavior and output alignment on a tiny sample. Record query use explicitly. If unavailable, mark the arm pending and exclude it from router training; never substitute another method silently. No installation, qualification, download or run is performed by this report revision.
3. Keep the query central. Permitted inference-time features include query embeddings, retrieval-score gaps/dispersion, explicitly defined query-to-passage/sentence MaxSim, redundancy, budget, and complexity estimates trained on FIT. Dataset/task identity is allowed only if known at deployment. Answer-type features must be known or predicted without gold labels; gold answer types, gold evidence, F1, HELP/HURT and oracle choices are not inference-time inputs. Outcome labels can supervise FIT-side research only.
4. Audit split provenance before training. Use dataset-balanced TRAIN/FIT sampling, with paper/document/query-disjoint validation as appropriate, and exclude all previously inspected pilot or validation examples and overlapping groups from training. Audit historical W0–W10 usage before calling any existing TEST split untouched. Reserve an unseen confirmation set if official TEST data have already informed decisions; do not relabel previously evaluated data as pristine TEST.

### Week 2: routing, separate LoRA ablation and frozen evaluation

5. Fit a lightweight router over the small feasible action set using training data only. Freeze model, features, action definitions, budgets and selection rule on validation. Oracle ceilings remain retrospective diagnostics; validate the actual router against cheap fixed policies on disjoint data.
6. Treat Qwen LoRA as an optional, separately ablated evidence-utilization and concise-output intervention. Use the same exact evidence, query, prompt and decoding for base-Qwen versus LoRA, with dataset-balanced TRAIN/FIT and disjoint validation. Include a 2×2 comparison: base/fixed evidence, base/fusion evidence, LoRA/fixed evidence, LoRA/fusion evidence. Freeze the fusion evidence across readers in this ablation so reader adaptation and exposure changes can be separated. Keep SARA as a separate specialist reference.
7. Fix and qualify SciFact's output contract in the new protocol before reader conclusions. Prespecify prompt/parser rules, retain raw outputs and report strict compliance plus task utility. Do not retroactively reparse the v1 primary scores.
8. Once the complete protocol is frozen, run one multi-dataset confirmation evaluation on the genuinely reserved data. Use paired paper/query-cluster intervals, report all preregistered actions and failures, and account separately for compressor/embedding preprocessing, query-time work, training and measured end-to-end cost. Add open-corpus HotpotQA only under an explicit separate protocol if claiming retriever robustness.

No dynamic fusion policy, LoRA enhancement or LLMLingua-2 result has been established by Week 11. Historical cross-week scores are not directly comparable; no synthetic comparable trajectory is fabricated. This plan does not start any experiments.


## Files for review

- A/B: ../retrieval/INTERPRETATION.md and ../passage_representation/INTERPRETATION.md
- C: ../granularity_budget/INTERPRETATION.md
- D: ../reader_granularity/RESEARCH_REVIEW.md
- E: ../compressor_reader/RESEARCH_REVIEW.md
- F: ../retriever_robustness/RESEARCH_REVIEW.md
- G: ../hierarchical/RESEARCH_REVIEW.md
- H: ../oracle/RESEARCH_REVIEW.md
- I: master_summary_df.csv, best_by_reader_df.csv, oracle_summary_df.csv, aggregate_duplicate_audit.json and figures.json

All available notebooks completed through I. Existing notebook edits and historical artifacts were preserved. Results, compact derived tables, audits and reports are committed on gagan; raw caches and executed notebook copies remain local for resume/review.
