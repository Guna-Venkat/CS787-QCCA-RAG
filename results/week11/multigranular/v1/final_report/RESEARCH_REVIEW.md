# Week 11 — completed multigranular evidence pilot

**All available phases 0–I are complete, audited and frozen on branch gagan. No further full run is active.** Optional LLMLingua-2/token conditions remain pending; this is completion of the available text study, not completion of those missing methods.

## Study scope and integrity

The frozen pilot uses 50 deterministic development queries per dataset (150 total, seed 42): QASPER project DEV, HotpotQA distractor DEV, SciFact DEV. It does not use the full datasets, and Week 11 does not read QASPER TEST. QASPER has 40 paper clusters among 50 queries; other datasets have 50 query/claim clusters. Readers are the pinned Mistral-7B-Instruct-v0.2, Qwen3-8B and Llama-3.1-8B-Instruct. Every reader completed its available assigned surface; no Llama placeholders remain.

There are 31,900 saved logical generation rows across C–G. Cross-phase reuse deduplicates to 25,300 unique dataset/query/action observations, 506 complete dataset/condition summaries, each with 50 outcomes. Duplicate predictions, utility, evidence, model and source identities agree exactly. These are many repeated conditions on 150 questions, not 25,300 independent questions. Retrieval rows A/B are separate from answer-generation rows.

## What each phase established

| Part | Work completed | Result |
|---|---|---|
| 0 | Dataset, hierarchy, reader and source qualification | Frozen 50-per-dataset cohort and pinned interfaces |
| A | BM25/MiniLM/SFR retrieval across passage/sentence/proposition units; 1,350 rows | SFR/P0 retained as controlled primary; no universal retriever winner |
| B | P0 passage embedding vs three sentence-pooling variants; 600 rows | Pooling gains inconclusive on this pilot; retain P0 |
| C | Qwen granularity × selector × evidence budget; 5,550 available rows | No universal optimal unit; compact units avoid some tight-budget failures |
| D | Same C evidence under all three readers; 16,650 rows | Reader-specific point patterns; SciFact format compliance strongly confounds two readers |
| E | Raw C0 vs exact top-eight sentences C2, plus QASPER SARA specialist; 4,150 rows | C2 helps tight budgets, not universally; adapted SARA is a strong separate specialist reference |
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

## Supported findings and unresolved claims

**1. Budget fit matters.** Qwen C2 at 256 beats same-budget raw C0 on QASPER by +0.109351 F1 (CI [0.046333, 0.171307]) and SciFact by +0.32 accuracy ([0.14, 0.50]). HotpotQA +0.100422 remains inconclusive ([−0.005202, 0.210173]). Some raw prefixes are empty because the first passage does not fit; these are not length-matched experiments. With larger budgets C2 is not consistently better, and top-eight sentence saturation explains identical 1024/FULL exposures.

**2. Hierarchy is not yet necessary.** H3/256 versus H0 gains have positive nominal intervals on all three datasets: QASPER +0.085380, HotpotQA +0.099869, SciFact +0.32. Against fixed semantic sentences at 256, every interval includes zero. The HotpotQA H3/1024 peak 0.487169 exceeds raw 0.480272 by only +0.006897, CI [−0.051911, 0.068590]. Compact source exposure is promising; the added value of mixing granularities is unproven.

**3. A universal reader/granularity interaction is unconfirmed.** Llama’s best QASPER point is 0.289146 and Qwen’s best HotpotQA point is 0.487169. The explicitly matched reader-by-granularity interactions have zero-containing intervals. Different point-estimate winners or significance in one group alone do not prove an interaction.

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

## Recommended next study

1. Preserve v1 and preregister a small v2 confirmation panel: no evidence, raw passages, fixed semantic sentences, C2 top-eight sentences and H3. Use fixed readers/retrievers and budgets 256/1024; do not select another winner on the same pilot.

2. Fix the SciFact output contract in v2 and requalify all readers before interpreting reader effects. Choose the prompt/parser rule in advance, retain raw outputs and strict compliance metrics, and report reasoning utility separately from formatting.

3. Separate allocation from representation: include a matched realized-token or preregistered skip-to-fit control, so empty strict-prefix raw contexts do not carry the whole compression contrast.

4. Expand to a larger prospectively specified confirmation cohort with paper/query separation. Add open-corpus retrieval if claiming retriever robustness; the Hotpot provided-context ceiling cannot answer that question.

5. Defer learned adaptive stopping until fixed-policy confirmation. If pursued, use paper-disjoint FIT training, inference-time features only, a held-out confirmation set, and compare against cheap fixed sentences. Keep SARA as a distinct adapted specialist; test matched adapted text/latent ablations before attributing its gain to latents.

LLMLingua-2 remains an explicit missing-method limitation, not an executed negative result. Historical cross-week trajectory records are unspecified; protocols differ and no comparable score series is fabricated.

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
