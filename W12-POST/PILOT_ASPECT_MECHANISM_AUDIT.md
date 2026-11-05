# Saved-pilot audit: aspect decomposition, stopping and FULL escalation

**Scope:** existing 150-query VAL pilot only; 900 saved answer rows, 150 saved problem caches, and saved selections/traces. Comparisons use both FIXED_SENTENCE and QBGF_FULL at each budget. No model generation, embedding computation, parameter changes, tuning, expansion, or confirmation access.

## Findings

1. **The aspect concern is confirmed descriptively:** 49/50 QASPER queries have exactly one extracted aspect. Hotpot has 26/50 single-aspect queries; SciFact has 40/50. The means are 1.64 / 1.02 / 1.20. In this pilot, all extracted aspects happened to meet the represented-aspect cosine threshold, so the two reported means coincide. They remain different definitions.
2. **Single-aspect coverage is not zero or constant:** QASPER still gets later coverage rewards in every query. It is improving the best similarity to one query fragment, rather than covering distinct decomposed information needs. At 256, mean total weighted coverage reward is 0.3607, with 0.1264 after the first action; at 1024, 0.3640 total and 0.1298 later. Thus replacing the interpretation with exactly `Rel − lambda Red` would be inaccurate.
3. **The aspect heuristic is lexical, not semantic decomposition.** It splits punctuation/conjunctions, filters fragments shorter than two words, caps at six, and falls back to the whole query. A single fragment can contain multiple information needs; several fragments can be incomplete phrases rather than independent sub-questions. The examples below demonstrate both issues. No decomposition-quality gold labels exist here.
4. **QASPER losses are real but not explained by aspect count alone.** BEA loses 0.0799 / 0.0812 answer F1 against FIXED at 256 / 1024. There is only **one** multi-aspect QASPER query, so there is no adequate comparison group to estimate a decomposition effect. At 256, BEA already uses 96.0% of budget; large unused context cannot be the sole explanation for that cell.
5. **Correct the FULL denominator:** Hotpot has **14 upgrade events in 13/50 queries** at 256 and **25 events in 19/50 queries** at 1024. QASPER has none; SciFact has 0/50 and 2/50. The hypothesis that escalation is mostly a Hotpot phenomenon is supported in this pilot.
6. **Lower redundancy does not establish downstream utility:** every QASPER query has lower measured redundancy than FIXED, yet mean F1 is lower. At QASPER 1024, unused tokens correlate negatively with relative F1 (Spearman −0.308 vs FIXED, −0.259 vs QBGF), which is consistent with possible under-allocation but does not prove that adding tokens would help.

The existing preregistered gate remains **PROMISING**: it allows up to two clearly degraded cells, and both QASPER cells are flagged. This audit does not rewrite that decision. It supports pausing automatic expansion for a mechanism discussion; it does not authorize changing aspects, thresholds or methods after these outcomes.

## Definitions and analysis limits

- **Extracted aspect count:** `len(problem.aspects)`, one value per unique query; distributions below use 50 queries/dataset, not 100 budget rows.
- **Represented aspects:** count of extracted fragments whose maximum cosine to exposed sentences is at least 0.5. That threshold is diagnostic only; the allocator's coverage reward uses continuous positive increases in maximum similarity, without this 0.5 cutoff.
- **Coverage is cosine, not verified fact coverage.** A whole scientific question can encode several requirements even when the lexical splitter returns one fragment. No causal utility claim follows from cosine scores.
- **Win/loss/tie:** QA compares saved per-query F1 using a numerical tolerance of 1e-12. SciFact compares saved per-claim correctness, because macro-F1 is not a per-claim score. Exact wins differ from the project's separate >0.01 QA HELP/HURT convention; both labels are retained in the paired CSV.
- The main aggregate table uses QA answer F1 and SciFact three-class macro-F1. Relationship tables use **QA F1 difference / SciFact accuracy difference**; this keeps per-query associations interpretable. Full CSVs additionally contain correctly recomputed subset macro-F1. Those subset values are class-composition dependent, nonadditive and unstable for tiny strata; a correctness tie can still change macro-F1 through a different wrong class.
- Relationships are exploratory descriptions of the same selected pilot, not new confirmation, causal effects or independent evidence. No correlation p-values, new multiplicity family or tuning choices are introduced. Each budget uses the same underlying 50 queries. Do not pool budgets as independent observations or pool task metrics.

## 1. Extracted aspect-count distributions

| dataset   |   aspect_count |   queries |   fraction |
|:----------|---------------:|----------:|-----------:|
| HotpotQA  |              1 |        26 |     0.5200 |
| HotpotQA  |              2 |        17 |     0.3400 |
| HotpotQA  |              3 |         6 |     0.1200 |
| HotpotQA  |              4 |         1 |     0.0200 |
| QASPER    |              1 |        49 |     0.9800 |
| QASPER    |              2 |         1 |     0.0200 |
| SciFact   |              1 |        40 |     0.8000 |
| SciFact   |              2 |        10 |     0.2000 |

### Extracted versus represented counts and actual allocation

| dataset   |   budget |   mean_extracted_aspects |   mean_represented_aspects |   one_aspect_queries |   mean_unused_tokens |   threshold_stops |
|:----------|---------:|-------------------------:|---------------------------:|---------------------:|---------------------:|------------------:|
| HotpotQA  |      256 |                   1.6400 |                     1.6400 |                   26 |              30.7400 |                31 |
| HotpotQA  |     1024 |                   1.6400 |                     1.6400 |                   26 |             739.3600 |                50 |
| QASPER    |      256 |                   1.0200 |                     1.0200 |                   49 |              10.2600 |                14 |
| QASPER    |     1024 |                   1.0200 |                     1.0200 |                   49 |             551.2200 |                48 |
| SciFact   |      256 |                   1.2000 |                     1.2000 |                   40 |               5.9200 |                 2 |
| SciFact   |     1024 |                   1.2000 |                     1.2000 |                   40 |             327.9800 |                45 |

## 2. Aggregate quality and wins/losses

Delta is BEA minus the named comparator; all cells contain 50 queries. The SciFact primary delta is macro-F1; its win/loss labels use correctness.

| dataset   |   budget | comparator     | primary_metric   |   delta_primary |   wins |   losses |   ties |
|:----------|---------:|:---------------|:-----------------|----------------:|-------:|---------:|-------:|
| HotpotQA  |      256 | FIXED_SENTENCE | answer F1        |         -0.0685 |      8 |       13 |     29 |
| HotpotQA  |      256 | QBGF_FULL      | answer F1        |         -0.1066 |      9 |       14 |     27 |
| HotpotQA  |     1024 | FIXED_SENTENCE | answer F1        |         -0.0452 |      7 |       12 |     31 |
| HotpotQA  |     1024 | QBGF_FULL      | answer F1        |         -0.0579 |     10 |       13 |     27 |
| QASPER    |      256 | FIXED_SENTENCE | answer F1        |         -0.0799 |      9 |       17 |     24 |
| QASPER    |      256 | QBGF_FULL      | answer F1        |         -0.0773 |     11 |       17 |     22 |
| QASPER    |     1024 | FIXED_SENTENCE | answer F1        |         -0.0812 |      9 |       16 |     25 |
| QASPER    |     1024 | QBGF_FULL      | answer F1        |         -0.0880 |      5 |       21 |     24 |
| SciFact   |      256 | FIXED_SENTENCE | macro-F1         |         -0.0162 |      8 |        9 |     33 |
| SciFact   |      256 | QBGF_FULL      | macro-F1         |          0.0817 |      8 |        4 |     38 |
| SciFact   |     1024 | FIXED_SENTENCE | macro-F1         |          0.0898 |      7 |        6 |     37 |
| SciFact   |     1024 | QBGF_FULL      | macro-F1         |          0.0551 |      6 |        5 |     39 |

### Aspect counts among wins, losses and ties

Primary reference: FIXED_SENTENCE. The complete corresponding QBGF table and exact count distributions are in `outcome_mechanisms.csv` and `aspect_counts_by_outcome.csv`.

| dataset   |   budget | outcome   |   queries |   mean_aspect_count |   mean_unused_tokens |   mean_redundancy_reduction |   full_queries |
|:----------|---------:|:----------|----------:|--------------------:|---------------------:|----------------------------:|---------------:|
| HotpotQA  |      256 | loss      |        13 |              1.4615 |              19.3846 |                      0.0566 |              0 |
| HotpotQA  |      256 | tie       |        29 |              1.7586 |              37.8966 |                      0.0578 |             11 |
| HotpotQA  |      256 | win       |         8 |              1.5000 |              23.2500 |                      0.0474 |              2 |
| HotpotQA  |     1024 | loss      |        12 |              1.3333 |             738.5000 |                     -0.0001 |              2 |
| HotpotQA  |     1024 | tie       |        31 |              1.8387 |             745.2903 |                     -0.0106 |             15 |
| HotpotQA  |     1024 | win       |         7 |              1.2857 |             714.5714 |                      0.0008 |              2 |
| QASPER    |      256 | loss      |        17 |              1.0000 |               8.7059 |                      0.1004 |              0 |
| QASPER    |      256 | tie       |        24 |              1.0000 |              10.9167 |                      0.0989 |              0 |
| QASPER    |      256 | win       |         9 |              1.1111 |              11.4444 |                      0.1133 |              0 |
| QASPER    |     1024 | loss      |        16 |              1.0000 |             608.7500 |                      0.0839 |              0 |
| QASPER    |     1024 | tie       |        25 |              1.0400 |             545.4000 |                      0.0826 |              0 |
| QASPER    |     1024 | win       |         9 |              1.0000 |             465.1111 |                      0.0803 |              0 |
| SciFact   |      256 | loss      |         9 |              1.1111 |               7.3333 |                      0.1161 |              0 |
| SciFact   |      256 | tie       |        33 |              1.2121 |               5.5455 |                      0.1275 |              0 |
| SciFact   |      256 | win       |         8 |              1.2500 |               5.8750 |                      0.1005 |              0 |
| SciFact   |     1024 | loss      |         6 |              1.0000 |             381.8333 |                      0.0939 |              0 |
| SciFact   |     1024 | tie       |        37 |              1.2703 |             300.5946 |                      0.0839 |              1 |
| SciFact   |     1024 | win       |         7 |              1.0000 |             426.5714 |                      0.0765 |              1 |

## 3. Quality difference versus extracted aspect count

The `3+` Hotpot stratum contains seven queries; the two-aspect QASPER stratum contains **one**. A positive value in that lone case cannot establish a general benefit. Hotpot shows no stable monotonic count–quality pattern across both budgets/comparators.

| dataset   |   budget | comparator     | aspect count   |   n |   wins |   losses |   delta F1 / accuracy |
|:----------|---------:|:---------------|:---------------|----:|-------:|---------:|----------------------:|
| HotpotQA  |      256 | FIXED_SENTENCE | 1              |  26 |      5 |        9 |               -0.0929 |
| HotpotQA  |      256 | FIXED_SENTENCE | 2              |  17 |      2 |        2 |               -0.0539 |
| HotpotQA  |      256 | FIXED_SENTENCE | 3+             |   7 |      1 |        2 |               -0.0129 |
| HotpotQA  |      256 | QBGF_FULL      | 1              |  26 |      6 |        8 |               -0.0920 |
| HotpotQA  |      256 | QBGF_FULL      | 2              |  17 |      2 |        3 |               -0.1137 |
| HotpotQA  |      256 | QBGF_FULL      | 3+             |   7 |      1 |        3 |               -0.1433 |
| HotpotQA  |     1024 | FIXED_SENTENCE | 1              |  26 |      6 |        9 |               -0.0276 |
| HotpotQA  |     1024 | FIXED_SENTENCE | 2              |  17 |      0 |        2 |               -0.0739 |
| HotpotQA  |     1024 | FIXED_SENTENCE | 3+             |   7 |      1 |        1 |               -0.0406 |
| HotpotQA  |     1024 | QBGF_FULL      | 1              |  26 |      7 |       11 |               -0.0912 |
| HotpotQA  |     1024 | QBGF_FULL      | 2              |  17 |      1 |        2 |               -0.0478 |
| HotpotQA  |     1024 | QBGF_FULL      | 3+             |   7 |      2 |        0 |                0.0410 |
| QASPER    |      256 | FIXED_SENTENCE | 1              |  49 |      8 |       17 |               -0.0819 |
| QASPER    |      256 | FIXED_SENTENCE | 2              |   1 |      1 |        0 |                0.0147 |
| QASPER    |      256 | QBGF_FULL      | 1              |  49 |     11 |       16 |               -0.0781 |
| QASPER    |      256 | QBGF_FULL      | 2              |   1 |      0 |        1 |               -0.0357 |
| QASPER    |     1024 | FIXED_SENTENCE | 1              |  49 |      9 |       16 |               -0.0828 |
| QASPER    |     1024 | FIXED_SENTENCE | 2              |   1 |      0 |        0 |                0.0000 |
| QASPER    |     1024 | QBGF_FULL      | 1              |  49 |      5 |       20 |               -0.0894 |
| QASPER    |     1024 | QBGF_FULL      | 2              |   1 |      0 |        1 |               -0.0190 |
| SciFact   |      256 | FIXED_SENTENCE | 1              |  40 |      6 |        8 |               -0.0500 |
| SciFact   |      256 | FIXED_SENTENCE | 2              |  10 |      2 |        1 |                0.1000 |
| SciFact   |      256 | QBGF_FULL      | 1              |  40 |      7 |        4 |                0.0750 |
| SciFact   |      256 | QBGF_FULL      | 2              |  10 |      1 |        0 |                0.1000 |
| SciFact   |     1024 | FIXED_SENTENCE | 1              |  40 |      7 |        6 |                0.0250 |
| SciFact   |     1024 | FIXED_SENTENCE | 2              |  10 |      0 |        0 |                0.0000 |
| SciFact   |     1024 | QBGF_FULL      | 1              |  40 |      6 |        5 |                0.0250 |
| SciFact   |     1024 | QBGF_FULL      | 2              |  10 |      0 |        0 |                0.0000 |

## 4. Quality difference versus unused tokens

Ranges below are unused fractions of the selected budget: [0,25%], (25%,50%], (50%,75%], (75%,100%]. These are descriptive bins set for this audit, not candidate policies. QASPER@1024 versus FIXED ranges from +0.0656 F1 for the nine queries using at least 75% of budget to −0.2597 for the eight leaving more than 75% unused. Input difficulty, query type and evidence quality can confound this pattern.

| dataset   |   budget | unused fraction   |   n |   wins |   losses |   delta F1 / accuracy |
|:----------|---------:|:------------------|----:|-------:|---------:|----------------------:|
| HotpotQA  |      256 | 0–25%             |  43 |      7 |       12 |               -0.0569 |
| HotpotQA  |      256 | 25–50%            |   7 |      1 |        1 |               -0.1395 |
| HotpotQA  |     1024 | 25–50%            |   3 |      1 |        1 |               -0.1624 |
| HotpotQA  |     1024 | 50–75%            |  25 |      4 |        4 |                0.0072 |
| HotpotQA  |     1024 | 75–100%           |  22 |      2 |        7 |               -0.0887 |
| QASPER    |      256 | 0–25%             |  49 |      8 |       17 |               -0.0861 |
| QASPER    |      256 | 25–50%            |   1 |      1 |        0 |                0.2222 |
| QASPER    |     1024 | 0–25%             |   9 |      3 |        1 |                0.0656 |
| QASPER    |     1024 | 25–50%            |   7 |      2 |        3 |               -0.0339 |
| QASPER    |     1024 | 50–75%            |  26 |      3 |        8 |               -0.0898 |
| QASPER    |     1024 | 75–100%           |   8 |      1 |        4 |               -0.2597 |
| SciFact   |      256 | 0–25%             |  50 |      8 |        9 |               -0.0200 |
| SciFact   |     1024 | 0–25%             |  16 |      1 |        1 |                0.0000 |
| SciFact   |     1024 | 25–50%            |  27 |      5 |        4 |                0.0370 |
| SciFact   |     1024 | 50–75%            |   7 |      1 |        1 |                0.0000 |

## 5. Quality difference versus redundancy reduction

Reduction = comparator pairwise sentence cosine minus BEA pairwise cosine. Positive means BEA has lower measured redundancy. FULL is measured using its constituent sentences; undefined values are excluded from correlations rather than converted to zero. All current paired values are defined. Reduced redundancy is not a guarantee of preserving complementary evidence, discourse context or answer utility.

| dataset   |   budget | stratum   |   n |   wins |   losses |   delta F1 / accuracy |
|:----------|---------:|:----------|----:|-------:|---------:|----------------------:|
| HotpotQA  |      256 | increased |   3 |      0 |        0 |                0.0000 |
| HotpotQA  |      256 | reduced   |  47 |      8 |       13 |               -0.0728 |
| HotpotQA  |     1024 | increased |  28 |      4 |        6 |               -0.0148 |
| HotpotQA  |     1024 | reduced   |  22 |      3 |        6 |               -0.0839 |
| QASPER    |      256 | reduced   |  50 |      9 |       17 |               -0.0799 |
| QASPER    |     1024 | reduced   |  50 |      9 |       16 |               -0.0812 |
| SciFact   |      256 | reduced   |  50 |      8 |        9 |               -0.0200 |
| SciFact   |     1024 | reduced   |  50 |      7 |        6 |                0.0200 |

### Rank associations for both comparators

Spearman rho uses tied ranks within each dataset/budget/comparator, n=50. Blank entries mean a constant feature (e.g. no FULL activity), not zero correlation. No association here identifies the cause of losses.

| dataset   |   budget | comparator     |   aspect_count |   unused_tokens |   redundancy_reduction |   full_active |
|:----------|---------:|:---------------|---------------:|----------------:|-----------------------:|--------------:|
| HotpotQA  |      256 | FIXED_SENTENCE |         0.0929 |          0.1317 |                -0.1115 |        0.2431 |
| HotpotQA  |      256 | QBGF_FULL      |        -0.0377 |          0.0400 |                -0.1525 |        0.1687 |
| HotpotQA  |     1024 | FIXED_SENTENCE |         0.0341 |         -0.0708 |                -0.0890 |        0.1292 |
| HotpotQA  |     1024 | QBGF_FULL      |         0.1703 |          0.0544 |                -0.0679 |        0.1478 |
| QASPER    |      256 | FIXED_SENTENCE |         0.1837 |         -0.1176 |                 0.0367 |      nan      |
| QASPER    |      256 | QBGF_FULL      |        -0.1087 |         -0.1100 |                 0.0938 |      nan      |
| QASPER    |     1024 | FIXED_SENTENCE |         0.0370 |         -0.3081 |                -0.0974 |      nan      |
| QASPER    |     1024 | QBGF_FULL      |        -0.0682 |         -0.2594 |                -0.0747 |      nan      |
| SciFact   |      256 | FIXED_SENTENCE |         0.1033 |         -0.1180 |                -0.0562 |      nan      |
| SciFact   |      256 | QBGF_FULL      |         0.0139 |          0.0340 |                -0.0214 |      nan      |
| SciFact   |     1024 | FIXED_SENTENCE |        -0.0225 |          0.0749 |                -0.0291 |        0.1933 |
| SciFact   |     1024 | QBGF_FULL      |        -0.0240 |          0.1845 |                 0.2436 |        0.2103 |

## 6. FULL activation and rejection

Every query reached eligibility at both budgets; the two-sentence prerequisite did not prevent **all consideration** on these saved trajectories. It could still affect when eligibility arises; there is no counterfactual relaxed-prerequisite run. Counts of rejection/choice below are repeated candidate-step events, **not distinct queries**.

| dataset   |   budget |   eligible_queries |   full_queries |   chosen |   rejected_cost |   rejected_threshold |   rejected_low_gain |   lost_competition |
|:----------|---------:|-------------------:|---------------:|---------:|----------------:|---------------------:|--------------------:|-------------------:|
| HotpotQA  |      256 |                 50 |             13 |       14 |            1108 |                  594 |                   8 |                246 |
| HotpotQA  |     1024 |                 50 |             19 |       25 |               0 |                 2434 |                   8 |                348 |
| QASPER    |      256 |                 50 |              0 |        0 |            3013 |                   12 |                   0 |                  2 |
| QASPER    |     1024 |                 50 |              0 |        0 |             231 |                 7027 |                   0 |                527 |
| SciFact   |      256 |                 50 |              0 |        0 |            1401 |                   24 |                   0 |                 37 |
| SciFact   |     1024 |                 50 |              2 |        2 |            1281 |                 6490 |                   0 |                254 |

QASPER@256: 3,013 cost rejections versus 12 threshold rejections. QASPER@1024: 7,027 threshold rejections, 231 cost rejections and 527 admissible candidates losing global competition. This supports a cost bottleneck at 256 and a gain-per-token/competition bottleneck at 1024. Counts revisit candidates as selection evolves; they are not independent trials.

### Quality conditioned on FULL activation

Association only: FULL and non-FULL queries differ before treatment. Hotpot@256 FULL queries fare better relative to FIXED (+0.0615 F1, n=13) than non-FULL queries (−0.1141, n=37). That is **not** a same-query ablation showing that FULL caused an improvement. The active SciFact group has only two claims.

| dataset   |   budget | comparator     | stratum     |   n |   wins |   losses |   delta F1 / accuracy |
|:----------|---------:|:---------------|:------------|----:|-------:|---------:|----------------------:|
| HotpotQA  |      256 | FIXED_SENTENCE | FULL active |  13 |      2 |        0 |                0.0615 |
| HotpotQA  |      256 | FIXED_SENTENCE | no FULL     |  37 |      6 |       13 |               -0.1141 |
| HotpotQA  |      256 | QBGF_FULL      | FULL active |  13 |      2 |        1 |               -0.0016 |
| HotpotQA  |      256 | QBGF_FULL      | no FULL     |  37 |      7 |       13 |               -0.1435 |
| HotpotQA  |     1024 | FIXED_SENTENCE | FULL active |  19 |      2 |        2 |               -0.0099 |
| HotpotQA  |     1024 | FIXED_SENTENCE | no FULL     |  31 |      5 |       10 |               -0.0668 |
| HotpotQA  |     1024 | QBGF_FULL      | FULL active |  19 |      3 |        2 |                0.0206 |
| HotpotQA  |     1024 | QBGF_FULL      | no FULL     |  31 |      7 |       11 |               -0.1061 |
| QASPER    |      256 | FIXED_SENTENCE | no FULL     |  50 |      9 |       17 |               -0.0799 |
| QASPER    |      256 | QBGF_FULL      | no FULL     |  50 |     11 |       17 |               -0.0773 |
| QASPER    |     1024 | FIXED_SENTENCE | no FULL     |  50 |      9 |       16 |               -0.0812 |
| QASPER    |     1024 | QBGF_FULL      | no FULL     |  50 |      5 |       21 |               -0.0880 |
| SciFact   |      256 | FIXED_SENTENCE | no FULL     |  50 |      8 |        9 |               -0.0200 |
| SciFact   |      256 | QBGF_FULL      | no FULL     |  50 |      8 |        4 |                0.0800 |
| SciFact   |     1024 | FIXED_SENTENCE | FULL active |   2 |      1 |        0 |                0.5000 |
| SciFact   |     1024 | FIXED_SENTENCE | no FULL     |  48 |      6 |        6 |                0.0000 |
| SciFact   |     1024 | QBGF_FULL      | FULL active |   2 |      1 |        0 |                0.5000 |
| SciFact   |     1024 | QBGF_FULL      | no FULL     |  48 |      5 |        5 |                0.0000 |

## 7. Stop reasons for wins and losses

Counts below include ties for complete denominators. `BUDGET_EXHAUSTED` means no remaining positive action fit; `LOW_MARGINAL_VALUE` means positive fitting actions existed but failed the threshold. Hotpot@1024 stops at threshold for **every** query, including wins and losses, so that stop reason alone cannot explain which queries lose. QASPER@256 has 12/17 losses ending for budget fit and 5/17 ending at threshold, again ruling out threshold stopping as the only explanation.

| dataset   |   budget | comparator     | outcome   |   BUDGET_EXHAUSTED |   LOW_MARGINAL_VALUE |
|:----------|---------:|:---------------|:----------|-------------------:|---------------------:|
| HotpotQA  |      256 | FIXED_SENTENCE | loss      |                  6 |                    7 |
| HotpotQA  |      256 | FIXED_SENTENCE | tie       |                 11 |                   18 |
| HotpotQA  |      256 | FIXED_SENTENCE | win       |                  2 |                    6 |
| HotpotQA  |      256 | QBGF_FULL      | loss      |                  7 |                    7 |
| HotpotQA  |      256 | QBGF_FULL      | tie       |                  9 |                   18 |
| HotpotQA  |      256 | QBGF_FULL      | win       |                  3 |                    6 |
| HotpotQA  |     1024 | FIXED_SENTENCE | loss      |                  0 |                   12 |
| HotpotQA  |     1024 | FIXED_SENTENCE | tie       |                  0 |                   31 |
| HotpotQA  |     1024 | FIXED_SENTENCE | win       |                  0 |                    7 |
| HotpotQA  |     1024 | QBGF_FULL      | loss      |                  0 |                   13 |
| HotpotQA  |     1024 | QBGF_FULL      | tie       |                  0 |                   27 |
| HotpotQA  |     1024 | QBGF_FULL      | win       |                  0 |                   10 |
| QASPER    |      256 | FIXED_SENTENCE | loss      |                 12 |                    5 |
| QASPER    |      256 | FIXED_SENTENCE | tie       |                 17 |                    7 |
| QASPER    |      256 | FIXED_SENTENCE | win       |                  7 |                    2 |
| QASPER    |      256 | QBGF_FULL      | loss      |                 13 |                    4 |
| QASPER    |      256 | QBGF_FULL      | tie       |                 17 |                    5 |
| QASPER    |      256 | QBGF_FULL      | win       |                  6 |                    5 |
| QASPER    |     1024 | FIXED_SENTENCE | loss      |                  1 |                   15 |
| QASPER    |     1024 | FIXED_SENTENCE | tie       |                  0 |                   25 |
| QASPER    |     1024 | FIXED_SENTENCE | win       |                  1 |                    8 |
| QASPER    |     1024 | QBGF_FULL      | loss      |                  1 |                   20 |
| QASPER    |     1024 | QBGF_FULL      | tie       |                  0 |                   24 |
| QASPER    |     1024 | QBGF_FULL      | win       |                  1 |                    4 |
| SciFact   |      256 | FIXED_SENTENCE | loss      |                  8 |                    1 |
| SciFact   |      256 | FIXED_SENTENCE | tie       |                 32 |                    1 |
| SciFact   |      256 | FIXED_SENTENCE | win       |                  8 |                    0 |
| SciFact   |      256 | QBGF_FULL      | loss      |                  4 |                    0 |
| SciFact   |      256 | QBGF_FULL      | tie       |                 36 |                    2 |
| SciFact   |      256 | QBGF_FULL      | win       |                  8 |                    0 |
| SciFact   |     1024 | FIXED_SENTENCE | loss      |                  0 |                    6 |
| SciFact   |     1024 | FIXED_SENTENCE | tie       |                  5 |                   32 |
| SciFact   |     1024 | FIXED_SENTENCE | win       |                  0 |                    7 |
| SciFact   |     1024 | QBGF_FULL      | loss      |                  1 |                    4 |
| SciFact   |     1024 | QBGF_FULL      | tie       |                  4 |                   35 |
| SciFact   |     1024 | QBGF_FULL      | win       |                  0 |                    6 |

## 8. Ten saved aspect examples per dataset

The first ten query IDs in lexicographic order are shown without selecting by outcome. Exact aspect lists come from the saved problem cache. Original raw questions were not retained in these cache objects; we do **not** reconstruct them or load fresh dataset records. IDs identify the queries; the strings below are the exact cached fragments. The CSV preserves them in full. All 150 lists are in `aspect_inventory.csv`.

### HotpotQA

| query_id                 |   count | aspects                                                                                                                                                                          |
|:-------------------------|--------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 5a7100435542994082a3e4a3 |       2 | Are Finding Kraftland  /  Shikashika both frozen drinks                                                                                                                          |
| 5a737d3c55429908901be2c8 |       3 | I Can See Your Voice  /  a reality show from South Korea  /  offers what kind of performers a chance to make their dreams of stardom a reality                                   |
| 5a739d8c55429905862fe073 |       1 | Near which town did the family of author of the "Adventures of Huckleberry Finn" own several tracts of land                                                                      |
| 5a7741775542994aec3b724f |       2 | What is the name of the character portrayed by Anna Camp in the film centered on the fictional Barden University  /  The Bellas                                                  |
| 5a7949d45542994bb9456fdd |       1 | What is the Latin name for one of the five "big cats" that the word "Panther" may refer to when outside of the Americas                                                          |
| 5a7a1e635542990783324e69 |       2 | What division are the 2013 Valdosta State Blazers football team  /  the Gulf South Conference a part of                                                                          |
| 5a7c76275542990527d554b4 |       2 | Which film was released first out of The Hunchback of Notre Dame  /  Miracle of the White Stallions                                                                              |
| 5a7df25a5542990b8f503b02 |       1 | What did the screewriter for a 2013 psychological thriller film directed by Denis Villeneuve get a Goya Award nomination for Best Adapted Screenplay at the 27th Goya Awards for |
| 5a7e30b255429934daa2fbeb |       1 | Katrina Ross starred in "The Singing Nun" with who as the title role                                                                                                             |
| 5a807e6f5542995d8a8ddf80 |       1 | Who were the guns stolen from the HMS ship "Marlin" originally produced for                                                                                                      |

### QASPER

| query_id                                 |   count | aspects                                                       |
|:-----------------------------------------|--------:|:--------------------------------------------------------------|
| 01a41c0a4a7365cd37d28690735114f2ff5229f2 |       1 | What social media platform did they look at                   |
| 05887a8466e0a2f0df4d6a5ffc5815acd7d9066a |       1 | Which SVM approach resulted in the best performance           |
| 113d791df6fcfc9cecfb7b1bebaf32cc2e4402ab |       1 | how is model compactness measured                             |
| 15e481e668114e4afe0c78eefb716ffe1646b494 |       1 | What alternative to Gibbs sampling is used                    |
| 1bb7eb5c3d029d95d1abf9f2892c1ec7b6eef306 |       1 | what was their character error rate                           |
| 21548433abd21346659505296fb0576e78287a74 |       1 | What dataset is used for this work                            |
| 2ccc26e11df4eb26fcccdd1f446dc749aff5d572 |       1 | Do they report results only on English data                   |
| 37a79be0148e1751ffb2daabe4c8ec6680036106 |       1 | What topic is covered in the Chinese Facebook data            |
| 3de0487276bb5961586acc6e9f82934ef8cb668c |       1 | What are the clinical datasets used in the paper              |
| 443d2448136364235389039cbead07e80922ec5c |       1 | What summarization algorithms did the authors experiment with |

### SciFact

|   query_id |   count | aspects                                                                                                                                                       |
|-----------:|--------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------|
|       1012 |       1 | Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume                                                                                 |
|       1019 |       1 | Rapid phosphotransfer rates govern fidelity in two component systems                                                                                          |
|       1029 |       1 | Reduced responsiveness to interleukin-2 in regulatory T cells is associated with greater resistance to autoimmune diseases such as Type 1 Diabetes            |
|       1049 |       2 | Ribosomopathies have a low degree of cell  /  tissue specific pathology                                                                                       |
|       1132 |       1 | TCR/CD3 microdomains are a required to induce the immunologic synapse to activate T cells                                                                     |
|       1202 |       1 | The center of the granuloma in an immune cell induces a pro-inflammatory immune response                                                                      |
|       1216 |       1 | The extracellular domain of TMEM27 is cleaved in human beta cells                                                                                             |
|       1266 |       2 | The risk of breast cancer among parous women increases with placental weight of pregnancies  /  this association is strongest for premenopausal breast cancer |
|       1290 |       2 | There is an inverse relationship between hip fractures  /  statin use                                                                                         |
|       1292 |       2 | There is no association between HNF4A mutations  /  diabetes risks                                                                                            |

## What this establishes before expansion

- The present QASPER run is operationally **query-fragment relevance/coverage + redundancy-aware sentence allocation + stopping**, with no FULL escalation. Describing it as empirically demonstrated multi-granular fusion on QASPER would overstate the saved evidence.
- It does **not** establish that absent semantic decomposition caused the losses: QASPER has almost no within-dataset aspect-count variation; many questions ask one attribute; and the 256 losses persist despite near-full budgets. The single-aspect reward still changes selection, so it is not equivalent to deleting coverage.
- At 1024 the under-allocation association warrants caution, while the tight-budget failures indicate that evidence content/ranking or the utility surrogate also needs explanation. These are hypotheses, not resolved causal findings.
- FULL is available but mostly unused outside Hotpot. Its query/event denominator and gain-per-token rejection behavior must be reported accurately. No inference that the prerequisite or threshold should be changed follows from this audit alone.
- **No tuning or expanded run was performed.** The historical final method and pilot freeze/gate remain intact. Do not present these post-hoc correlations as a new validation result or a successful no-coverage/no-FULL ablation.

## Integrity and saved artifacts

The audit validated 900 answer envelopes and checksum/identity pairing, 150 problem caches and 900 saved selections, reproduced all 18 existing primary-quality summary cells to tolerance 1e-12, and verified **1,954 saved input files remained byte-identical**. Source hashes and the script hash are in `INPUT_MANIFEST.json`. The audit imports no model, tokenizer, encoder or experiment runner. No benchmark file or scientific implementation was edited.

All CSVs and provenance: `results/post_w12/bea_qbgf/v3/pilot/aspect_mechanism_audit/`. The per-query CSV has **600 paired rows** (300 BEA query/budget actions × two comparators); the unique-query aspect distribution has 150 rows. Main displayed relationships use FIXED where specified; complete QBGF tables are retained for unused-token and redundancy strata as well.

Reproduce with:

```bash
sara_env/bin/python W12-POST/scripts/audit_saved_pilot.py
```

Branch: gagan. All analysis uses saved pilot data only.
