# W11-A completed retrieval pilot — interpretation

Reviewed 2026-10-27T21:34:25.535511+00:00; branch gagan. All 1,350 stored rows passed checksums, pairing, cohort, metric-range/monotonicity and frozen-input identity checks. 150 queries, 50 per dataset; 9 configurations per query. Approximate run/report wall time: 203.8 minutes.

## Main findings

- SFR passage any-gold recall@6: QASPER 0.6250, HotpotQA 1.0000, SciFact 0.9200. BM25: 0.5208, 0.9800, 0.9600 respectively.
- HotpotQA SFR sentence retrieval retains any-gold recall@6=0.9600 with 199.56 mean tokens, compared with passage 1.0000 at 776.52 tokens. SciFact SFR sentence recall@6=0.9600 at 266.12 tokens, compared with passage 0.9200 at 1,931.00 tokens. These are different unit/token budgets, not controlled downstream quality comparisons.
- QASPER fine units have low full-annotation recall but higher partial coverage. Paragraph-sized annotations can require multiple fine units before any annotation is fully covered. Low strict any-gold recall alone does not establish inferior answer utility.
- SFR vs BM25 passage coverage@6: QASPER +0.0508, 95% CI [-0.0724, 0.1795]; HotpotQA +0.1433, CI [0.0700, 0.2200]; SciFact -0.0400, CI [-0.1600, 0.0800]. QASPER/SciFact passage superiority is not established.
- Within-granularity SFR sentence/proposition coverage contrasts against BM25 have positive nominal CIs on this pilot. Multiple comparisons and the small development samples require exploratory interpretation.
- No answer generation, answer F1, HELP/HURT or deployable RAG improvement was measured in Part A.

## Flags and limits

1. **QASPER source mapping:** seven queries contain unmapped annotations: 1109, 1214, 1496, 1704, 2102, 2192, 343. Four of these have no mapped annotation; three have a partial annotation mapping. Original metrics retain these misses. A separate post-hoc fully mapped sensitivity table has 41 evaluable queries; it does not replace the frozen results.
2. **Recall denominators:** QASPER 48/50, HotpotQA 50/50, SciFact 25/50 have gold evidence annotations. SciFact has 25 NOT_ENOUGH_INFO claims with no annotated evidence; their recall is null, not an observed zero.
3. **HotpotQA ceiling:** passage@10 exposes all ten supplied distractor-context paragraphs; 100% annotated recall there is a corpus-exhaustion ceiling, not evidence of superior ranking.
4. **Unequal budgets:** k passages cost far more tokens than k sentences/clauses. Phase C must compare shared token budgets before drawing granularity/utilization conclusions.
5. **Timing:** retrieval latency includes cold document-embedding/index work where encountered; it excludes some outer evaluation/report overhead. It is not a steady-state per-query serving benchmark. Shared caches make subsequent timings conditional on this run order.
6. **Pilot scope:** 50 queries per dataset, with paper-cluster uncertainty on QASPER. Results are exploratory; thousands of bootstrap draws do not increase the number of independent examples.
7. **No global winner by raw pooling:** retrieval metrics have different annotation semantics across datasets. We do not pool them into an overall task score.

## Next-phase decision

Continue B with its prespecified SFR P0/P1/P2/P3 passage-representation comparison. Keep SFR as the controlled primary retriever; retain BM25/MiniLM baselines for later robustness. Do not silently replace the frozen corpus, query cohort, metric semantics, or downstream controls after observing pilot outcomes. No answer-quality winner is frozen here.

## Complete configuration summary at k=6

| dataset   | retriever   | evidence_granularity   |   gold_metric_denominator |   any_gold_recall |   all_gold_recall |   evidence_coverage |   retrieved_tokens |    mrr |
|:----------|:------------|:-----------------------|--------------------------:|------------------:|------------------:|--------------------:|-------------------:|-------:|
| HotpotQA  | BM25        | PASSAGE                |                        50 |            0.9800 |            0.6800 |              0.8300 |           812.3400 | 0.9029 |
| HotpotQA  | MiniLM      | PASSAGE                |                        50 |            1.0000 |            0.8000 |              0.9067 |           761.7200 | 0.9217 |
| HotpotQA  | SFR         | PASSAGE                |                        50 |            1.0000 |            0.9400 |              0.9733 |           776.5200 | 0.9700 |
| HotpotQA  | BM25        | PROPOSITION            |                        50 |            0.8400 |            0.1600 |              0.5284 |           171.2400 | 0.6733 |
| HotpotQA  | MiniLM      | PROPOSITION            |                        50 |            0.8600 |            0.1800 |              0.5623 |           144.4000 | 0.6061 |
| HotpotQA  | SFR         | PROPOSITION            |                        50 |            0.9600 |            0.5200 |              0.7815 |           182.1800 | 0.8115 |
| HotpotQA  | BM25        | SENTENCE               |                        50 |            0.8800 |            0.2200 |              0.5588 |           179.3600 | 0.7117 |
| HotpotQA  | MiniLM      | SENTENCE               |                        50 |            0.9400 |            0.2600 |              0.6047 |           155.5800 | 0.6494 |
| HotpotQA  | SFR         | SENTENCE               |                        50 |            0.9600 |            0.5800 |              0.7952 |           199.5600 | 0.8924 |
| QASPER    | BM25        | PASSAGE                |                        48 |            0.5208 |            0.3750 |              0.5404 |          1485.4800 | 0.3102 |
| QASPER    | MiniLM      | PASSAGE                |                        48 |            0.4792 |            0.4167 |              0.5253 |          1474.2200 | 0.2851 |
| QASPER    | SFR         | PASSAGE                |                        48 |            0.6250 |            0.4583 |              0.5911 |          1475.4000 | 0.3220 |
| QASPER    | BM25        | PROPOSITION            |                        48 |            0.0208 |            0.0000 |              0.0867 |           160.7600 | 0.0212 |
| QASPER    | MiniLM      | PROPOSITION            |                        48 |            0.0625 |            0.0000 |              0.1038 |           156.4200 | 0.0457 |
| QASPER    | SFR         | PROPOSITION            |                        48 |            0.0417 |            0.0000 |              0.1648 |           205.7400 | 0.0446 |
| QASPER    | BM25        | SENTENCE               |                        48 |            0.0417 |            0.0000 |              0.1021 |           169.9000 | 0.0227 |
| QASPER    | MiniLM      | SENTENCE               |                        48 |            0.0625 |            0.0000 |              0.1083 |           161.8800 | 0.0462 |
| QASPER    | SFR         | SENTENCE               |                        48 |            0.0625 |            0.0000 |              0.1787 |           216.2000 | 0.0461 |
| SciFact   | BM25        | PASSAGE                |                        25 |            0.9600 |            0.9600 |              0.9600 |          1996.5400 | 0.8586 |
| SciFact   | MiniLM      | PASSAGE                |                        25 |            0.8800 |            0.8800 |              0.8800 |          1965.2800 | 0.7102 |
| SciFact   | SFR         | PASSAGE                |                        25 |            0.9200 |            0.9200 |              0.9200 |          1931.0000 | 0.8530 |
| SciFact   | BM25        | PROPOSITION            |                        25 |            0.6000 |            0.4000 |              0.5415 |           180.5800 | 0.4197 |
| SciFact   | MiniLM      | PROPOSITION            |                        25 |            0.7200 |            0.5200 |              0.6782 |           189.1200 | 0.6114 |
| SciFact   | SFR         | PROPOSITION            |                        25 |            0.8800 |            0.6800 |              0.8100 |           250.9400 | 0.6977 |
| SciFact   | BM25        | SENTENCE               |                        25 |            0.7200 |            0.4800 |              0.5867 |           187.2200 | 0.4989 |
| SciFact   | MiniLM      | SENTENCE               |                        25 |            0.8000 |            0.6000 |              0.7033 |           192.4600 | 0.6956 |
| SciFact   | SFR         | SENTENCE               |                        25 |            0.9600 |            0.7600 |              0.8600 |           266.1200 | 0.8014 |

## Complete configuration summary at k=10

| dataset   | retriever   | evidence_granularity   |   gold_metric_denominator |   any_gold_recall |   all_gold_recall |   evidence_coverage |   retrieved_tokens |    mrr |
|:----------|:------------|:-----------------------|--------------------------:|------------------:|------------------:|--------------------:|-------------------:|-------:|
| HotpotQA  | BM25        | PASSAGE                |                        50 |            1.0000 |            1.0000 |              1.0000 |          1351.0400 | 0.9029 |
| HotpotQA  | MiniLM      | PASSAGE                |                        50 |            1.0000 |            1.0000 |              1.0000 |          1351.0400 | 0.9217 |
| HotpotQA  | SFR         | PASSAGE                |                        50 |            1.0000 |            1.0000 |              1.0000 |          1351.0400 | 0.9700 |
| HotpotQA  | BM25        | PROPOSITION            |                        50 |            0.9400 |            0.3000 |              0.6551 |           301.8800 | 0.6733 |
| HotpotQA  | MiniLM      | PROPOSITION            |                        50 |            0.9000 |            0.3400 |              0.6856 |           249.3800 | 0.6061 |
| HotpotQA  | SFR         | PROPOSITION            |                        50 |            1.0000 |            0.6800 |              0.8776 |           302.3000 | 0.8115 |
| HotpotQA  | BM25        | SENTENCE               |                        50 |            0.9600 |            0.4400 |              0.7105 |           314.7200 | 0.7117 |
| HotpotQA  | MiniLM      | SENTENCE               |                        50 |            0.9800 |            0.4400 |              0.7410 |           266.6200 | 0.6494 |
| HotpotQA  | SFR         | SENTENCE               |                        50 |            1.0000 |            0.7200 |              0.8950 |           319.8400 | 0.8924 |
| QASPER    | BM25        | PASSAGE                |                        48 |            0.6875 |            0.5625 |              0.6892 |          2399.1800 | 0.3102 |
| QASPER    | MiniLM      | PASSAGE                |                        48 |            0.6250 |            0.5625 |              0.6901 |          2388.7600 | 0.2851 |
| QASPER    | SFR         | PASSAGE                |                        48 |            0.7708 |            0.6458 |              0.7488 |          2387.7200 | 0.3220 |
| QASPER    | BM25        | PROPOSITION            |                        48 |            0.0625 |            0.0208 |              0.1421 |           275.9800 | 0.0212 |
| QASPER    | MiniLM      | PROPOSITION            |                        48 |            0.0833 |            0.0000 |              0.1464 |           264.5600 | 0.0457 |
| QASPER    | SFR         | PROPOSITION            |                        48 |            0.1042 |            0.0000 |              0.2594 |           353.4600 | 0.0446 |
| QASPER    | BM25        | SENTENCE               |                        48 |            0.0625 |            0.0417 |              0.1568 |           284.8600 | 0.0227 |
| QASPER    | MiniLM      | SENTENCE               |                        48 |            0.0833 |            0.0208 |              0.1595 |           276.2600 | 0.0462 |
| QASPER    | SFR         | SENTENCE               |                        48 |            0.1042 |            0.0000 |              0.2671 |           367.7600 | 0.0461 |
| SciFact   | BM25        | PASSAGE                |                        25 |            0.9600 |            0.9600 |              0.9600 |          3394.5600 | 0.8586 |
| SciFact   | MiniLM      | PASSAGE                |                        25 |            0.9200 |            0.9200 |              0.9200 |          3284.2200 | 0.7102 |
| SciFact   | SFR         | PASSAGE                |                        25 |            1.0000 |            1.0000 |              1.0000 |          3208.9400 | 0.8530 |
| SciFact   | BM25        | PROPOSITION            |                        25 |            0.6000 |            0.4400 |              0.5615 |           301.6400 | 0.4197 |
| SciFact   | MiniLM      | PROPOSITION            |                        25 |            0.7600 |            0.5200 |              0.6915 |           309.0600 | 0.6114 |
| SciFact   | SFR         | PROPOSITION            |                        25 |            0.8800 |            0.6800 |              0.8100 |           430.7400 | 0.6977 |
| SciFact   | BM25        | SENTENCE               |                        25 |            0.7200 |            0.5200 |              0.6067 |           310.7200 | 0.4989 |
| SciFact   | MiniLM      | SENTENCE               |                        25 |            0.8800 |            0.6000 |              0.7367 |           315.3000 | 0.6956 |
| SciFact   | SFR         | SENTENCE               |                        25 |            0.9600 |            0.7600 |              0.8600 |           459.5200 | 0.8014 |

## Labelled paired coverage comparisons at k=6

| dataset   | retriever_a   | granularity_a   | retriever_b   | granularity_b   |   point |   ci_low |   ci_high |   resamples | cluster           |
|:----------|:--------------|:----------------|:--------------|:----------------|--------:|---------:|----------:|------------:|:------------------|
| HotpotQA  | SFR           | PROPOSITION     | BM25          | PROPOSITION     |  0.2531 |   0.1708 |    0.3387 |       10000 | query_or_claim_id |
| HotpotQA  | MiniLM        | PASSAGE         | BM25          | PASSAGE         |  0.0767 |  -0.0000 |    0.1533 |       10000 | query_or_claim_id |
| HotpotQA  | MiniLM        | SENTENCE        | BM25          | SENTENCE        |  0.0458 |  -0.0402 |    0.1345 |       10000 | query_or_claim_id |
| HotpotQA  | SFR           | SENTENCE        | BM25          | SENTENCE        |  0.2363 |   0.1523 |    0.3220 |       10000 | query_or_claim_id |
| HotpotQA  | MiniLM        | PROPOSITION     | BM25          | PROPOSITION     |  0.0339 |  -0.0528 |    0.1218 |       10000 | query_or_claim_id |
| HotpotQA  | SFR           | PASSAGE         | BM25          | PASSAGE         |  0.1433 |   0.0700 |    0.2200 |       10000 | query_or_claim_id |
| QASPER    | SFR           | PROPOSITION     | BM25          | PROPOSITION     |  0.0782 |   0.0172 |    0.1389 |       10000 | paper_id          |
| QASPER    | MiniLM        | PASSAGE         | BM25          | PASSAGE         | -0.0151 |  -0.1548 |    0.1211 |       10000 | paper_id          |
| QASPER    | MiniLM        | SENTENCE        | BM25          | SENTENCE        |  0.0062 |  -0.0441 |    0.0589 |       10000 | paper_id          |
| QASPER    | SFR           | SENTENCE        | BM25          | SENTENCE        |  0.0766 |   0.0154 |    0.1377 |       10000 | paper_id          |
| QASPER    | MiniLM        | PROPOSITION     | BM25          | PROPOSITION     |  0.0171 |  -0.0267 |    0.0637 |       10000 | paper_id          |
| QASPER    | SFR           | PASSAGE         | BM25          | PASSAGE         |  0.0508 |  -0.0724 |    0.1795 |       10000 | paper_id          |
| SciFact   | SFR           | PROPOSITION     | BM25          | PROPOSITION     |  0.2685 |   0.1285 |    0.4200 |       10000 | query_or_claim_id |
| SciFact   | MiniLM        | PASSAGE         | BM25          | PASSAGE         | -0.0800 |  -0.2400 |    0.0800 |       10000 | query_or_claim_id |
| SciFact   | MiniLM        | SENTENCE        | BM25          | SENTENCE        |  0.1167 |  -0.0100 |    0.2600 |       10000 | query_or_claim_id |
| SciFact   | SFR           | SENTENCE        | BM25          | SENTENCE        |  0.2733 |   0.1333 |    0.4267 |       10000 | query_or_claim_id |
| SciFact   | MiniLM        | PROPOSITION     | BM25          | PROPOSITION     |  0.1367 |   0.0233 |    0.2733 |       10000 | query_or_claim_id |
| SciFact   | SFR           | PASSAGE         | BM25          | PASSAGE         | -0.0400 |  -0.1600 |    0.0800 |       10000 | query_or_claim_id |

Nominal 95% paired CIs; no multiplicity adjustment.

## Fully mapped QASPER sensitivity at k=6 and k=10

| retriever   | evidence_granularity   |   k |   any_gold_recall |   all_gold_recall |   evidence_coverage |   n_gold_queries |
|:------------|:-----------------------|----:|------------------:|------------------:|--------------------:|-----------------:|
| BM25        | PASSAGE                |   6 |            0.5366 |            0.4390 |              0.5965 |               41 |
| BM25        | PASSAGE                |  10 |            0.7317 |            0.6585 |              0.7707 |               41 |
| BM25        | PROPOSITION            |   6 |            0.0244 |            0.0000 |              0.0932 |               41 |
| BM25        | PROPOSITION            |  10 |            0.0732 |            0.0244 |              0.1581 |               41 |
| BM25        | SENTENCE               |   6 |            0.0488 |            0.0000 |              0.1113 |               41 |
| BM25        | SENTENCE               |  10 |            0.0732 |            0.0488 |              0.1753 |               41 |
| MiniLM      | PASSAGE                |   6 |            0.4878 |            0.4878 |              0.5789 |               41 |
| MiniLM      | PASSAGE                |  10 |            0.6585 |            0.6585 |              0.7718 |               41 |
| MiniLM      | PROPOSITION            |   6 |            0.0732 |            0.0000 |              0.1132 |               41 |
| MiniLM      | PROPOSITION            |  10 |            0.0732 |            0.0000 |              0.1563 |               41 |
| MiniLM      | SENTENCE               |   6 |            0.0732 |            0.0000 |              0.1186 |               41 |
| MiniLM      | SENTENCE               |  10 |            0.0732 |            0.0244 |              0.1717 |               41 |
| SFR         | PASSAGE                |   6 |            0.6585 |            0.5366 |              0.6493 |               41 |
| SFR         | PASSAGE                |  10 |            0.8293 |            0.7561 |              0.8339 |               41 |
| SFR         | PROPOSITION            |   6 |            0.0244 |            0.0000 |              0.1725 |               41 |
| SFR         | PROPOSITION            |  10 |            0.0732 |            0.0000 |              0.2787 |               41 |
| SFR         | SENTENCE               |   6 |            0.0488 |            0.0000 |              0.1888 |               41 |
| SFR         | SENTENCE               |  10 |            0.0732 |            0.0000 |              0.2878 |               41 |

## All flagged query records

| dataset   |   query_id |   unmapped_gold |   gold_annotation_count | kind                 |
|:----------|-----------:|----------------:|------------------------:|:---------------------|
| QASPER    |       1964 |               0 |                       0 | no_gold_annotations  |
| QASPER    |       2014 |               0 |                       0 | no_gold_annotations  |
| QASPER    |       1109 |               1 |                       1 | unmapped_annotations |
| QASPER    |       1214 |               1 |                       2 | unmapped_annotations |
| QASPER    |       1496 |               1 |                       1 | unmapped_annotations |
| QASPER    |       1704 |               1 |                       4 | unmapped_annotations |
| QASPER    |       2102 |               2 |                       2 | unmapped_annotations |
| QASPER    |       2192 |               1 |                       1 | unmapped_annotations |
| QASPER    |        343 |               1 |                       2 | unmapped_annotations |
| SciFact   |       1099 |               0 |                       0 | no_gold_annotations  |
| SciFact   |       1100 |               0 |                       0 | no_gold_annotations  |
| SciFact   |       1194 |               0 |                       0 | no_gold_annotations  |
| SciFact   |       1196 |               0 |                       0 | no_gold_annotations  |
| SciFact   |       1199 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        128 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        268 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        269 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        294 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        312 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        362 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        384 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        440 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        502 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        507 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        517 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        535 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        544 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        575 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        577 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        727 |               0 |                       0 | no_gold_annotations  |
| SciFact   |         75 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        820 |               0 |                       0 | no_gold_annotations  |
| SciFact   |        914 |               0 |                       0 | no_gold_annotations  |
| SciFact   |         94 |               0 |                       0 | no_gold_annotations  |

## Full recall surface (all recorded k values)

| dataset   | retriever   | evidence_granularity   |   k |   gold_metric_denominator |   any_gold_recall |   all_gold_recall |   evidence_coverage |   retrieved_tokens |
|:----------|:------------|:-----------------------|----:|--------------------------:|------------------:|------------------:|--------------------:|-------------------:|
| HotpotQA  | BM25        | PASSAGE                |   1 |                        50 |            0.8400 |            0.0000 |              0.4295 |           111.5600 |
| HotpotQA  | BM25        | PASSAGE                |   2 |                        50 |            0.9200 |            0.4000 |              0.6770 |           241.2600 |
| HotpotQA  | BM25        | PASSAGE                |   4 |                        50 |            0.9800 |            0.5800 |              0.7800 |           539.4400 |
| HotpotQA  | BM25        | PASSAGE                |   6 |                        50 |            0.9800 |            0.6800 |              0.8300 |           812.3400 |
| HotpotQA  | BM25        | PASSAGE                |   8 |                        50 |            1.0000 |            0.9000 |              0.9500 |          1073.9000 |
| HotpotQA  | BM25        | PASSAGE                |  10 |                        50 |            1.0000 |            1.0000 |              1.0000 |          1351.0400 |
| HotpotQA  | MiniLM      | PASSAGE                |   1 |                        50 |            0.8600 |            0.0000 |              0.4418 |            92.2000 |
| HotpotQA  | MiniLM      | PASSAGE                |   2 |                        50 |            0.9600 |            0.3600 |              0.6628 |           209.8400 |
| HotpotQA  | MiniLM      | PASSAGE                |   4 |                        50 |            1.0000 |            0.6600 |              0.8400 |           488.9600 |
| HotpotQA  | MiniLM      | PASSAGE                |   6 |                        50 |            1.0000 |            0.8000 |              0.9067 |           761.7200 |
| HotpotQA  | MiniLM      | PASSAGE                |   8 |                        50 |            1.0000 |            0.8600 |              0.9333 |          1046.1000 |
| HotpotQA  | MiniLM      | PASSAGE                |  10 |                        50 |            1.0000 |            1.0000 |              1.0000 |          1351.0400 |
| HotpotQA  | SFR         | PASSAGE                |   1 |                        50 |            0.9400 |            0.0000 |              0.4745 |            92.8400 |
| HotpotQA  | SFR         | PASSAGE                |   2 |                        50 |            1.0000 |            0.6600 |              0.8463 |           207.0400 |
| HotpotQA  | SFR         | PASSAGE                |   4 |                        50 |            1.0000 |            0.8800 |              0.9500 |           473.4600 |
| HotpotQA  | SFR         | PASSAGE                |   6 |                        50 |            1.0000 |            0.9400 |              0.9733 |           776.5200 |
| HotpotQA  | SFR         | PASSAGE                |   8 |                        50 |            1.0000 |            1.0000 |              1.0000 |          1048.0000 |
| HotpotQA  | SFR         | PASSAGE                |  10 |                        50 |            1.0000 |            1.0000 |              1.0000 |          1351.0400 |
| HotpotQA  | BM25        | PROPOSITION            |   1 |                        50 |            0.5400 |            0.0000 |              0.2235 |            24.3200 |
| HotpotQA  | BM25        | PROPOSITION            |   2 |                        50 |            0.6800 |            0.0600 |              0.3603 |            50.4600 |
| HotpotQA  | BM25        | PROPOSITION            |   4 |                        50 |            0.8200 |            0.1000 |              0.4718 |           112.5600 |
| HotpotQA  | BM25        | PROPOSITION            |   6 |                        50 |            0.8400 |            0.1600 |              0.5284 |           171.2400 |
| HotpotQA  | BM25        | PROPOSITION            |   8 |                        50 |            0.9200 |            0.2000 |              0.5876 |           239.1000 |
| HotpotQA  | BM25        | PROPOSITION            |  10 |                        50 |            0.9400 |            0.3000 |              0.6551 |           301.8800 |
| HotpotQA  | MiniLM      | PROPOSITION            |   1 |                        50 |            0.4200 |            0.0000 |              0.1891 |            20.0000 |
| HotpotQA  | MiniLM      | PROPOSITION            |   2 |                        50 |            0.6600 |            0.0200 |              0.3175 |            45.6000 |
| HotpotQA  | MiniLM      | PROPOSITION            |   4 |                        50 |            0.8000 |            0.1200 |              0.4789 |            90.3200 |
| HotpotQA  | MiniLM      | PROPOSITION            |   6 |                        50 |            0.8600 |            0.1800 |              0.5623 |           144.4000 |
| HotpotQA  | MiniLM      | PROPOSITION            |   8 |                        50 |            0.8800 |            0.2400 |              0.6045 |           197.5000 |
| HotpotQA  | MiniLM      | PROPOSITION            |  10 |                        50 |            0.9000 |            0.3400 |              0.6856 |           249.3800 |
| HotpotQA  | SFR         | PROPOSITION            |   1 |                        50 |            0.7200 |            0.0000 |              0.3122 |            30.1600 |
| HotpotQA  | SFR         | PROPOSITION            |   2 |                        50 |            0.8200 |            0.1200 |              0.4901 |            64.1800 |
| HotpotQA  | SFR         | PROPOSITION            |   4 |                        50 |            0.9200 |            0.3600 |              0.6658 |           123.3600 |
| HotpotQA  | SFR         | PROPOSITION            |   6 |                        50 |            0.9600 |            0.5200 |              0.7815 |           182.1800 |
| HotpotQA  | SFR         | PROPOSITION            |   8 |                        50 |            0.9800 |            0.6200 |              0.8349 |           244.0400 |
| HotpotQA  | SFR         | PROPOSITION            |  10 |                        50 |            1.0000 |            0.6800 |              0.8776 |           302.3000 |
| HotpotQA  | BM25        | SENTENCE               |   1 |                        50 |            0.5800 |            0.0000 |              0.2345 |            26.7800 |
| HotpotQA  | BM25        | SENTENCE               |   2 |                        50 |            0.7200 |            0.0800 |              0.3767 |            56.4800 |
| HotpotQA  | BM25        | SENTENCE               |   4 |                        50 |            0.8600 |            0.1400 |              0.4982 |           119.9000 |
| HotpotQA  | BM25        | SENTENCE               |   6 |                        50 |            0.8800 |            0.2200 |              0.5588 |           179.3600 |
| HotpotQA  | BM25        | SENTENCE               |   8 |                        50 |            0.9400 |            0.3000 |              0.6255 |           250.7400 |
| HotpotQA  | BM25        | SENTENCE               |  10 |                        50 |            0.9600 |            0.4400 |              0.7105 |           314.7200 |
| HotpotQA  | MiniLM      | SENTENCE               |   1 |                        50 |            0.4600 |            0.0000 |              0.1822 |            20.7600 |
| HotpotQA  | MiniLM      | SENTENCE               |   2 |                        50 |            0.7000 |            0.0400 |              0.3168 |            47.3800 |
| HotpotQA  | MiniLM      | SENTENCE               |   4 |                        50 |            0.8600 |            0.1800 |              0.5117 |            95.8800 |
| HotpotQA  | MiniLM      | SENTENCE               |   6 |                        50 |            0.9400 |            0.2600 |              0.6047 |           155.5800 |
| HotpotQA  | MiniLM      | SENTENCE               |   8 |                        50 |            0.9600 |            0.3400 |              0.6653 |           211.8400 |
| HotpotQA  | MiniLM      | SENTENCE               |  10 |                        50 |            0.9800 |            0.4400 |              0.7410 |           266.6200 |
| HotpotQA  | SFR         | SENTENCE               |   1 |                        50 |            0.8400 |            0.0000 |              0.3445 |            34.1800 |
| HotpotQA  | SFR         | SENTENCE               |   2 |                        50 |            0.9000 |            0.2400 |              0.5503 |            69.0600 |
| HotpotQA  | SFR         | SENTENCE               |   4 |                        50 |            0.9400 |            0.5000 |              0.7253 |           134.8800 |
| HotpotQA  | SFR         | SENTENCE               |   6 |                        50 |            0.9600 |            0.5800 |              0.7952 |           199.5600 |
| HotpotQA  | SFR         | SENTENCE               |   8 |                        50 |            1.0000 |            0.6800 |              0.8727 |           261.3800 |
| HotpotQA  | SFR         | SENTENCE               |  10 |                        50 |            1.0000 |            0.7200 |              0.8950 |           319.8400 |
| QASPER    | BM25        | PASSAGE                |   1 |                        48 |            0.1250 |            0.0417 |              0.1239 |           253.8800 |
| QASPER    | BM25        | PASSAGE                |   2 |                        48 |            0.2917 |            0.1458 |              0.2831 |           506.8000 |
| QASPER    | BM25        | PASSAGE                |   4 |                        48 |            0.5000 |            0.3125 |              0.4765 |          1006.1000 |
| QASPER    | BM25        | PASSAGE                |   6 |                        48 |            0.5208 |            0.3750 |              0.5404 |          1485.4800 |
| QASPER    | BM25        | PASSAGE                |   8 |                        48 |            0.6250 |            0.4792 |              0.6343 |          1948.7200 |
| QASPER    | BM25        | PASSAGE                |  10 |                        48 |            0.6875 |            0.5625 |              0.6892 |          2399.1800 |
| QASPER    | MiniLM      | PASSAGE                |   1 |                        48 |            0.1250 |            0.0625 |              0.1076 |           249.3600 |
| QASPER    | MiniLM      | PASSAGE                |   2 |                        48 |            0.2292 |            0.1250 |              0.2372 |           503.4000 |
| QASPER    | MiniLM      | PASSAGE                |   4 |                        48 |            0.4583 |            0.3333 |              0.4456 |          1009.2400 |
| QASPER    | MiniLM      | PASSAGE                |   6 |                        48 |            0.4792 |            0.4167 |              0.5253 |          1474.2200 |
| QASPER    | MiniLM      | PASSAGE                |   8 |                        48 |            0.5833 |            0.5000 |              0.6419 |          1939.5200 |
| QASPER    | MiniLM      | PASSAGE                |  10 |                        48 |            0.6250 |            0.5625 |              0.6901 |          2388.7600 |
| QASPER    | SFR         | PASSAGE                |   1 |                        48 |            0.1667 |            0.1042 |              0.1730 |           248.7800 |
| QASPER    | SFR         | PASSAGE                |   2 |                        48 |            0.2500 |            0.1667 |              0.2691 |           491.8000 |
| QASPER    | SFR         | PASSAGE                |   4 |                        48 |            0.3750 |            0.2500 |              0.4085 |           990.9800 |
| QASPER    | SFR         | PASSAGE                |   6 |                        48 |            0.6250 |            0.4583 |              0.5911 |          1475.4000 |
| QASPER    | SFR         | PASSAGE                |   8 |                        48 |            0.6875 |            0.5625 |              0.6751 |          1937.3600 |
| QASPER    | SFR         | PASSAGE                |  10 |                        48 |            0.7708 |            0.6458 |              0.7488 |          2387.7200 |
| QASPER    | BM25        | PROPOSITION            |   1 |                        48 |            0.0000 |            0.0000 |              0.0254 |            23.3000 |
| QASPER    | BM25        | PROPOSITION            |   2 |                        48 |            0.0000 |            0.0000 |              0.0347 |            51.3600 |
| QASPER    | BM25        | PROPOSITION            |   4 |                        48 |            0.0000 |            0.0000 |              0.0700 |           102.7600 |
| QASPER    | BM25        | PROPOSITION            |   6 |                        48 |            0.0208 |            0.0000 |              0.0867 |           160.7600 |
| QASPER    | BM25        | PROPOSITION            |   8 |                        48 |            0.0208 |            0.0000 |              0.1131 |           220.0800 |
| QASPER    | BM25        | PROPOSITION            |  10 |                        48 |            0.0625 |            0.0208 |              0.1421 |           275.9800 |
| QASPER    | MiniLM      | PROPOSITION            |   1 |                        48 |            0.0208 |            0.0000 |              0.0426 |            22.1000 |
| QASPER    | MiniLM      | PROPOSITION            |   2 |                        48 |            0.0208 |            0.0000 |              0.0595 |            50.0400 |
| QASPER    | MiniLM      | PROPOSITION            |   4 |                        48 |            0.0417 |            0.0000 |              0.0825 |           104.8600 |
| QASPER    | MiniLM      | PROPOSITION            |   6 |                        48 |            0.0625 |            0.0000 |              0.1038 |           156.4200 |
| QASPER    | MiniLM      | PROPOSITION            |   8 |                        48 |            0.0833 |            0.0000 |              0.1404 |           210.4000 |
| QASPER    | MiniLM      | PROPOSITION            |  10 |                        48 |            0.0833 |            0.0000 |              0.1464 |           264.5600 |
| QASPER    | SFR         | PROPOSITION            |   1 |                        48 |            0.0208 |            0.0000 |              0.0576 |            31.4000 |
| QASPER    | SFR         | PROPOSITION            |   2 |                        48 |            0.0208 |            0.0000 |              0.0662 |            65.3200 |
| QASPER    | SFR         | PROPOSITION            |   4 |                        48 |            0.0208 |            0.0000 |              0.1207 |           135.2400 |
| QASPER    | SFR         | PROPOSITION            |   6 |                        48 |            0.0417 |            0.0000 |              0.1648 |           205.7400 |
| QASPER    | SFR         | PROPOSITION            |   8 |                        48 |            0.0625 |            0.0000 |              0.1983 |           280.8800 |
| QASPER    | SFR         | PROPOSITION            |  10 |                        48 |            0.1042 |            0.0000 |              0.2594 |           353.4600 |
| QASPER    | BM25        | SENTENCE               |   1 |                        48 |            0.0000 |            0.0000 |              0.0254 |            24.6200 |
| QASPER    | BM25        | SENTENCE               |   2 |                        48 |            0.0000 |            0.0000 |              0.0347 |            53.2000 |
| QASPER    | BM25        | SENTENCE               |   4 |                        48 |            0.0000 |            0.0000 |              0.0700 |           107.0000 |
| QASPER    | BM25        | SENTENCE               |   6 |                        48 |            0.0417 |            0.0000 |              0.1021 |           169.9000 |
| QASPER    | BM25        | SENTENCE               |   8 |                        48 |            0.0417 |            0.0000 |              0.1278 |           226.0000 |
| QASPER    | BM25        | SENTENCE               |  10 |                        48 |            0.0625 |            0.0417 |              0.1568 |           284.8600 |
| QASPER    | MiniLM      | SENTENCE               |   1 |                        48 |            0.0208 |            0.0000 |              0.0426 |            23.1000 |
| QASPER    | MiniLM      | SENTENCE               |   2 |                        48 |            0.0208 |            0.0000 |              0.0595 |            49.6000 |
| QASPER    | MiniLM      | SENTENCE               |   4 |                        48 |            0.0417 |            0.0000 |              0.0825 |           107.2400 |
| QASPER    | MiniLM      | SENTENCE               |   6 |                        48 |            0.0625 |            0.0000 |              0.1083 |           161.8800 |
| QASPER    | MiniLM      | SENTENCE               |   8 |                        48 |            0.0833 |            0.0208 |              0.1554 |           222.1800 |
| QASPER    | MiniLM      | SENTENCE               |  10 |                        48 |            0.0833 |            0.0208 |              0.1595 |           276.2600 |
| QASPER    | SFR         | SENTENCE               |   1 |                        48 |            0.0208 |            0.0000 |              0.0576 |            31.7200 |
| QASPER    | SFR         | SENTENCE               |   2 |                        48 |            0.0208 |            0.0000 |              0.0649 |            67.7800 |
| QASPER    | SFR         | SENTENCE               |   4 |                        48 |            0.0208 |            0.0000 |              0.1194 |           138.1000 |
| QASPER    | SFR         | SENTENCE               |   6 |                        48 |            0.0625 |            0.0000 |              0.1787 |           216.2000 |
| QASPER    | SFR         | SENTENCE               |   8 |                        48 |            0.0833 |            0.0000 |              0.2122 |           292.4000 |
| QASPER    | SFR         | SENTENCE               |  10 |                        48 |            0.1042 |            0.0000 |              0.2671 |           367.7600 |
| SciFact   | BM25        | PASSAGE                |   1 |                        25 |            0.8000 |            0.8000 |              0.8000 |           331.6600 |
| SciFact   | BM25        | PASSAGE                |   2 |                        25 |            0.8800 |            0.8800 |              0.8800 |           676.6200 |
| SciFact   | BM25        | PASSAGE                |   4 |                        25 |            0.9200 |            0.9200 |              0.9200 |          1338.0200 |
| SciFact   | BM25        | PASSAGE                |   6 |                        25 |            0.9600 |            0.9600 |              0.9600 |          1996.5400 |
| SciFact   | BM25        | PASSAGE                |   8 |                        25 |            0.9600 |            0.9600 |              0.9600 |          2692.1200 |
| SciFact   | BM25        | PASSAGE                |  10 |                        25 |            0.9600 |            0.9600 |              0.9600 |          3394.5600 |
| SciFact   | MiniLM      | PASSAGE                |   1 |                        25 |            0.6000 |            0.6000 |              0.6000 |           325.0200 |
| SciFact   | MiniLM      | PASSAGE                |   2 |                        25 |            0.7200 |            0.7200 |              0.7200 |           635.6600 |
| SciFact   | MiniLM      | PASSAGE                |   4 |                        25 |            0.8400 |            0.8400 |              0.8400 |          1292.7600 |
| SciFact   | MiniLM      | PASSAGE                |   6 |                        25 |            0.8800 |            0.8800 |              0.8800 |          1965.2800 |
| SciFact   | MiniLM      | PASSAGE                |   8 |                        25 |            0.9200 |            0.9200 |              0.9200 |          2605.0600 |
| SciFact   | MiniLM      | PASSAGE                |  10 |                        25 |            0.9200 |            0.9200 |              0.9200 |          3284.2200 |
| SciFact   | SFR         | PASSAGE                |   1 |                        25 |            0.8000 |            0.8000 |              0.8000 |           327.7600 |
| SciFact   | SFR         | PASSAGE                |   2 |                        25 |            0.8400 |            0.8400 |              0.8400 |           641.6400 |
| SciFact   | SFR         | PASSAGE                |   4 |                        25 |            0.9200 |            0.9200 |              0.9200 |          1289.0000 |
| SciFact   | SFR         | PASSAGE                |   6 |                        25 |            0.9200 |            0.9200 |              0.9200 |          1931.0000 |
| SciFact   | SFR         | PASSAGE                |   8 |                        25 |            0.9600 |            0.9600 |              0.9600 |          2556.1400 |
| SciFact   | SFR         | PASSAGE                |  10 |                        25 |            1.0000 |            1.0000 |              1.0000 |          3208.9400 |
| SciFact   | BM25        | PROPOSITION            |   1 |                        25 |            0.2800 |            0.2400 |              0.2776 |            30.7000 |
| SciFact   | BM25        | PROPOSITION            |   2 |                        25 |            0.4800 |            0.3600 |              0.4682 |            60.3400 |
| SciFact   | BM25        | PROPOSITION            |   4 |                        25 |            0.5600 |            0.4000 |              0.5215 |           122.7400 |
| SciFact   | BM25        | PROPOSITION            |   6 |                        25 |            0.6000 |            0.4000 |              0.5415 |           180.5800 |
| SciFact   | BM25        | PROPOSITION            |   8 |                        25 |            0.6000 |            0.4400 |              0.5615 |           239.4200 |
| SciFact   | BM25        | PROPOSITION            |  10 |                        25 |            0.6000 |            0.4400 |              0.5615 |           301.6400 |
| SciFact   | MiniLM      | PROPOSITION            |   1 |                        25 |            0.5600 |            0.4000 |              0.4909 |            33.1200 |
| SciFact   | MiniLM      | PROPOSITION            |   2 |                        25 |            0.6000 |            0.4000 |              0.5358 |            64.9800 |
| SciFact   | MiniLM      | PROPOSITION            |   4 |                        25 |            0.6400 |            0.4000 |              0.5615 |           126.6800 |
| SciFact   | MiniLM      | PROPOSITION            |   6 |                        25 |            0.7200 |            0.5200 |              0.6782 |           189.1200 |
| SciFact   | MiniLM      | PROPOSITION            |   8 |                        25 |            0.7600 |            0.5200 |              0.6915 |           247.9800 |
| SciFact   | MiniLM      | PROPOSITION            |  10 |                        25 |            0.7600 |            0.5200 |              0.6915 |           309.0600 |
| SciFact   | SFR         | PROPOSITION            |   1 |                        25 |            0.6000 |            0.3200 |              0.4982 |            43.6800 |
| SciFact   | SFR         | PROPOSITION            |   2 |                        25 |            0.6800 |            0.4000 |              0.5815 |            85.3600 |
| SciFact   | SFR         | PROPOSITION            |   4 |                        25 |            0.8800 |            0.6000 |              0.7867 |           167.2800 |
| SciFact   | SFR         | PROPOSITION            |   6 |                        25 |            0.8800 |            0.6800 |              0.8100 |           250.9400 |
| SciFact   | SFR         | PROPOSITION            |   8 |                        25 |            0.8800 |            0.6800 |              0.8100 |           345.9200 |
| SciFact   | SFR         | PROPOSITION            |  10 |                        25 |            0.8800 |            0.6800 |              0.8100 |           430.7400 |
| SciFact   | BM25        | SENTENCE               |   1 |                        25 |            0.3200 |            0.2800 |              0.3000 |            31.7800 |
| SciFact   | BM25        | SENTENCE               |   2 |                        25 |            0.6000 |            0.4400 |              0.5133 |            63.0800 |
| SciFact   | BM25        | SENTENCE               |   4 |                        25 |            0.6800 |            0.4800 |              0.5667 |           128.2600 |
| SciFact   | BM25        | SENTENCE               |   6 |                        25 |            0.7200 |            0.4800 |              0.5867 |           187.2200 |
| SciFact   | BM25        | SENTENCE               |   8 |                        25 |            0.7200 |            0.5200 |              0.6067 |           248.0000 |
| SciFact   | BM25        | SENTENCE               |  10 |                        25 |            0.7200 |            0.5200 |              0.6067 |           310.7200 |
| SciFact   | MiniLM      | SENTENCE               |   1 |                        25 |            0.6400 |            0.4800 |              0.5533 |            35.9200 |
| SciFact   | MiniLM      | SENTENCE               |   2 |                        25 |            0.6800 |            0.4800 |              0.5733 |            67.2400 |
| SciFact   | MiniLM      | SENTENCE               |   4 |                        25 |            0.7200 |            0.4800 |              0.5867 |           129.1800 |
| SciFact   | MiniLM      | SENTENCE               |   6 |                        25 |            0.8000 |            0.6000 |              0.7033 |           192.4600 |
| SciFact   | MiniLM      | SENTENCE               |   8 |                        25 |            0.8400 |            0.6000 |              0.7167 |           251.8600 |
| SciFact   | MiniLM      | SENTENCE               |  10 |                        25 |            0.8800 |            0.6000 |              0.7367 |           315.3000 |
| SciFact   | SFR         | SENTENCE               |   1 |                        25 |            0.7200 |            0.4000 |              0.5433 |            46.4200 |
| SciFact   | SFR         | SENTENCE               |   2 |                        25 |            0.8000 |            0.4800 |              0.6267 |            88.0400 |
| SciFact   | SFR         | SENTENCE               |   4 |                        25 |            0.9600 |            0.6800 |              0.8267 |           173.6000 |
| SciFact   | SFR         | SENTENCE               |   6 |                        25 |            0.9600 |            0.7600 |              0.8600 |           266.1200 |
| SciFact   | SFR         | SENTENCE               |   8 |                        25 |            0.9600 |            0.7600 |              0.8600 |           367.8800 |
| SciFact   | SFR         | SENTENCE               |  10 |                        25 |            0.9600 |            0.7600 |              0.8600 |           459.5200 |
