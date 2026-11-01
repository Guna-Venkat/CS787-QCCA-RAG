# W11-B completed passage-representation pilot

Reviewed 2026-10-27T21:56:11.522120+00:00; 600/600 results on the unchanged 150-query cohort. P0=direct passage; P1=mean normalized sentence pooling; P2=max query/sentence cosine; P3=query-weighted sentence pooling (beta=5).

## Findings

- P3 increases QASPER coverage@6 from 0.5911 to 0.6352, while any-gold recall decreases from 0.6250 to 0.6042. This is a metric trade-off, not a uniform improvement.
- On SciFact, P1/P3 raise any/all-gold recall@6 from 0.92 to 1.00 on only 25 evidence-annotated claims. That is two additional claims; it comes with longer retrieved passages (~2,162–2,165 tokens vs 1,931 for P0).
- HotpotQA any-gold recall@6 is 1.00 for every representation; P3 all-gold recall changes 0.94 to 0.96. Top10 returns all available paragraphs and saturates for every method.
- There is no consistent winner across recall, coverage, ranking and tokens. Keep prespecified P0/SFR as the controlled source pool for C; P3 remains an exploratory follow-up rather than a silently promoted primary method.

## Flags

- Same pilot-size, gold-denominator and seven-QASPER-mapping limitations as A. No annotation or original result was repaired/replaced.
- Two SciFact claims can move the recall point estimate by eight percentage points. Use paired CIs below and do not infer a general improvement from this ceiling.
- Different selected passage lengths confound comparisons at fixed k; C evaluates shared token budgets.
- The original B driver repeatedly scanned all units for every passage/configuration. An operational grouping/cache optimization was recorded separately; all 403 existing rankings matched exactly before 197 new rows were saved. Every new row stores its execution-driver hash.
- P0 rankings AND all recorded metrics match the frozen A SFR/PASSAGE results for all 150 queries. Frozen source/config inputs and existing rows are unchanged.
- B does not establish answer F1 or HELP/HURT gains; no answer generation was performed.
- Runtime comparison between the reference and optimized execution epochs is not a scientific latency benchmark. The optimization is output-equivalent bookkeeping and cache reuse; its supplemental provenance is in optimized_execution_identity.json and optimization_equivalence.json.

## Results at k=6 and k=10

| dataset   | passage_representation   |   k |   gold_metric_denominator |   any_gold_recall |   all_gold_recall |   evidence_coverage |   retrieved_tokens |    mrr |
|:----------|:-------------------------|----:|--------------------------:|------------------:|------------------:|--------------------:|-------------------:|-------:|
| HotpotQA  | P0                       |   6 |                        50 |            1.0000 |            0.9400 |              0.9733 |           776.5200 | 0.9700 |
| HotpotQA  | P1                       |   6 |                        50 |            1.0000 |            0.9400 |              0.9700 |           824.4600 | 0.9900 |
| HotpotQA  | P2                       |   6 |                        50 |            1.0000 |            0.9400 |              0.9767 |           776.9200 | 0.9767 |
| HotpotQA  | P3                       |   6 |                        50 |            1.0000 |            0.9600 |              0.9800 |           813.6200 | 0.9700 |
| HotpotQA  | P0                       |  10 |                        50 |            1.0000 |            1.0000 |              1.0000 |          1351.0400 | 0.9700 |
| HotpotQA  | P1                       |  10 |                        50 |            1.0000 |            1.0000 |              1.0000 |          1351.0400 | 0.9900 |
| HotpotQA  | P2                       |  10 |                        50 |            1.0000 |            1.0000 |              1.0000 |          1351.0400 | 0.9767 |
| HotpotQA  | P3                       |  10 |                        50 |            1.0000 |            1.0000 |              1.0000 |          1351.0400 | 0.9700 |
| QASPER    | P0                       |   6 |                        48 |            0.6250 |            0.4583 |              0.5911 |          1475.4000 | 0.3220 |
| QASPER    | P1                       |   6 |                        48 |            0.5833 |            0.5000 |              0.6245 |          1490.3400 | 0.3481 |
| QASPER    | P2                       |   6 |                        48 |            0.5833 |            0.4167 |              0.5855 |          1497.8600 | 0.3379 |
| QASPER    | P3                       |   6 |                        48 |            0.6042 |            0.5000 |              0.6352 |          1491.5400 | 0.3465 |
| QASPER    | P0                       |  10 |                        48 |            0.7708 |            0.6458 |              0.7488 |          2387.7200 | 0.3220 |
| QASPER    | P1                       |  10 |                        48 |            0.7708 |            0.6458 |              0.7678 |          2405.8000 | 0.3481 |
| QASPER    | P2                       |  10 |                        48 |            0.7500 |            0.6458 |              0.7604 |          2404.0800 | 0.3379 |
| QASPER    | P3                       |  10 |                        48 |            0.7500 |            0.6458 |              0.7509 |          2408.9800 | 0.3465 |
| SciFact   | P0                       |   6 |                        25 |            0.9200 |            0.9200 |              0.9200 |          1931.0000 | 0.8530 |
| SciFact   | P1                       |   6 |                        25 |            1.0000 |            1.0000 |              1.0000 |          2165.4600 | 0.8433 |
| SciFact   | P2                       |   6 |                        25 |            0.9600 |            0.9600 |              0.9600 |          2121.2400 | 0.8878 |
| SciFact   | P3                       |   6 |                        25 |            1.0000 |            1.0000 |              1.0000 |          2162.4400 | 0.9100 |
| SciFact   | P0                       |  10 |                        25 |            1.0000 |            1.0000 |              1.0000 |          3208.9400 | 0.8530 |
| SciFact   | P1                       |  10 |                        25 |            1.0000 |            1.0000 |              1.0000 |          3642.1800 | 0.8433 |
| SciFact   | P2                       |  10 |                        25 |            1.0000 |            1.0000 |              1.0000 |          3567.8000 | 0.8878 |
| SciFact   | P3                       |  10 |                        25 |            1.0000 |            1.0000 |              1.0000 |          3645.0800 | 0.9100 |

## Paired pooling contrasts at k=6

| dataset   | representation   | baseline   | metric            |   k |   n_gold | family      |   point |   ci_low |   ci_high |   resamples |   seed | cluster           |
|:----------|:-----------------|:-----------|:------------------|----:|---------:|:------------|--------:|---------:|----------:|------------:|-------:|:------------------|
| HotpotQA  | P1               | P0         | evidence_coverage |   6 |       50 | exploratory | -0.0033 |  -0.0400 |    0.0333 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | P2               | P0         | evidence_coverage |   6 |       50 | exploratory |  0.0033 |  -0.0267 |    0.0333 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | P3               | P0         | evidence_coverage |   6 |       50 | exploratory |  0.0067 |  -0.0233 |    0.0367 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | P1               | P0         | any_gold_recall   |   6 |       50 | exploratory |  0.0000 |   0.0000 |    0.0000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | P2               | P0         | any_gold_recall   |   6 |       50 | exploratory |  0.0000 |   0.0000 |    0.0000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | P3               | P0         | any_gold_recall   |   6 |       50 | exploratory |  0.0000 |   0.0000 |    0.0000 |       10000 |     42 | query_or_claim_id |
| QASPER    | P1               | P0         | evidence_coverage |   6 |       48 | exploratory |  0.0334 |  -0.0520 |    0.1243 |       10000 |     42 | paper_id          |
| QASPER    | P2               | P0         | evidence_coverage |   6 |       48 | exploratory | -0.0056 |  -0.1138 |    0.0998 |       10000 |     42 | paper_id          |
| QASPER    | P3               | P0         | evidence_coverage |   6 |       48 | exploratory |  0.0441 |  -0.0298 |    0.1228 |       10000 |     42 | paper_id          |
| QASPER    | P1               | P0         | any_gold_recall   |   6 |       48 | exploratory | -0.0417 |  -0.1633 |    0.0652 |       10000 |     42 | paper_id          |
| QASPER    | P2               | P0         | any_gold_recall   |   6 |       48 | exploratory | -0.0417 |  -0.1569 |    0.0698 |       10000 |     42 | paper_id          |
| QASPER    | P3               | P0         | any_gold_recall   |   6 |       48 | exploratory | -0.0208 |  -0.1346 |    0.0816 |       10000 |     42 | paper_id          |
| SciFact   | P1               | P0         | evidence_coverage |   6 |       25 | exploratory |  0.0800 |   0.0000 |    0.2000 |       10000 |     42 | query_or_claim_id |
| SciFact   | P2               | P0         | evidence_coverage |   6 |       25 | exploratory |  0.0400 |   0.0000 |    0.1200 |       10000 |     42 | query_or_claim_id |
| SciFact   | P3               | P0         | evidence_coverage |   6 |       25 | exploratory |  0.0800 |   0.0000 |    0.2000 |       10000 |     42 | query_or_claim_id |
| SciFact   | P1               | P0         | any_gold_recall   |   6 |       25 | exploratory |  0.0800 |   0.0000 |    0.2000 |       10000 |     42 | query_or_claim_id |
| SciFact   | P2               | P0         | any_gold_recall   |   6 |       25 | exploratory |  0.0400 |   0.0000 |    0.1200 |       10000 |     42 | query_or_claim_id |
| SciFact   | P3               | P0         | any_gold_recall   |   6 |       25 | exploratory |  0.0800 |   0.0000 |    0.2000 |       10000 |     42 | query_or_claim_id |

Nominal 95% paired bootstrap CIs, 10,000 resamples, seed 42. QASPER clusters by paper; HotpotQA and SciFact by query/claim. These are exploratory, unadjusted comparisons.

## Complete surface

| dataset   | passage_representation   |   k |   gold_metric_denominator |   any_gold_recall |   all_gold_recall |   evidence_coverage |   retrieved_tokens |    mrr |
|:----------|:-------------------------|----:|--------------------------:|------------------:|------------------:|--------------------:|-------------------:|-------:|
| HotpotQA  | P0                       |   1 |                        50 |            0.9400 |            0.0000 |              0.4745 |            92.8400 | 0.9700 |
| HotpotQA  | P1                       |   1 |                        50 |            0.9800 |            0.0000 |              0.4995 |            98.1200 | 0.9900 |
| HotpotQA  | P2                       |   1 |                        50 |            0.9600 |            0.0000 |              0.4838 |            91.3400 | 0.9767 |
| HotpotQA  | P3                       |   1 |                        50 |            0.9400 |            0.0000 |              0.4895 |            95.7400 | 0.9700 |
| HotpotQA  | P0                       |   2 |                        50 |            1.0000 |            0.6600 |              0.8463 |           207.0400 | 0.9700 |
| HotpotQA  | P1                       |   2 |                        50 |            1.0000 |            0.5800 |              0.7997 |           223.6400 | 0.9900 |
| HotpotQA  | P2                       |   2 |                        50 |            0.9800 |            0.6000 |              0.7933 |           218.0400 | 0.9767 |
| HotpotQA  | P3                       |   2 |                        50 |            1.0000 |            0.6400 |              0.8263 |           228.0400 | 0.9700 |
| HotpotQA  | P0                       |   4 |                        50 |            1.0000 |            0.8800 |              0.9500 |           473.4600 | 0.9700 |
| HotpotQA  | P1                       |   4 |                        50 |            1.0000 |            0.8400 |              0.9233 |           535.0000 | 0.9900 |
| HotpotQA  | P2                       |   4 |                        50 |            1.0000 |            0.8400 |              0.9267 |           503.1600 | 0.9767 |
| HotpotQA  | P3                       |   4 |                        50 |            1.0000 |            0.9000 |              0.9500 |           516.2200 | 0.9700 |
| HotpotQA  | P0                       |   6 |                        50 |            1.0000 |            0.9400 |              0.9733 |           776.5200 | 0.9700 |
| HotpotQA  | P1                       |   6 |                        50 |            1.0000 |            0.9400 |              0.9700 |           824.4600 | 0.9900 |
| HotpotQA  | P2                       |   6 |                        50 |            1.0000 |            0.9400 |              0.9767 |           776.9200 | 0.9767 |
| HotpotQA  | P3                       |   6 |                        50 |            1.0000 |            0.9600 |              0.9800 |           813.6200 | 0.9700 |
| HotpotQA  | P0                       |   8 |                        50 |            1.0000 |            1.0000 |              1.0000 |          1048.0000 | 0.9700 |
| HotpotQA  | P1                       |   8 |                        50 |            1.0000 |            0.9800 |              0.9900 |          1078.4400 | 0.9900 |
| HotpotQA  | P2                       |   8 |                        50 |            1.0000 |            0.9800 |              0.9900 |          1061.3400 | 0.9767 |
| HotpotQA  | P3                       |   8 |                        50 |            1.0000 |            0.9800 |              0.9900 |          1085.9800 | 0.9700 |
| HotpotQA  | P0                       |  10 |                        50 |            1.0000 |            1.0000 |              1.0000 |          1351.0400 | 0.9700 |
| HotpotQA  | P1                       |  10 |                        50 |            1.0000 |            1.0000 |              1.0000 |          1351.0400 | 0.9900 |
| HotpotQA  | P2                       |  10 |                        50 |            1.0000 |            1.0000 |              1.0000 |          1351.0400 | 0.9767 |
| HotpotQA  | P3                       |  10 |                        50 |            1.0000 |            1.0000 |              1.0000 |          1351.0400 | 0.9700 |
| QASPER    | P0                       |   1 |                        48 |            0.1667 |            0.1042 |              0.1730 |           248.7800 | 0.3220 |
| QASPER    | P1                       |   1 |                        48 |            0.2083 |            0.1042 |              0.1831 |           253.8600 | 0.3481 |
| QASPER    | P2                       |   1 |                        48 |            0.1875 |            0.1042 |              0.1884 |           252.7400 | 0.3379 |
| QASPER    | P3                       |   1 |                        48 |            0.1875 |            0.1042 |              0.1758 |           255.6600 | 0.3465 |
| QASPER    | P0                       |   2 |                        48 |            0.2500 |            0.1667 |              0.2691 |           491.8000 | 0.3220 |
| QASPER    | P1                       |   2 |                        48 |            0.2500 |            0.1458 |              0.2721 |           507.3000 | 0.3481 |
| QASPER    | P2                       |   2 |                        48 |            0.2500 |            0.1667 |              0.2648 |           505.3400 | 0.3379 |
| QASPER    | P3                       |   2 |                        48 |            0.2708 |            0.1667 |              0.2868 |           509.3600 | 0.3465 |
| QASPER    | P0                       |   4 |                        48 |            0.3750 |            0.2500 |              0.4085 |           990.9800 | 0.3220 |
| QASPER    | P1                       |   4 |                        48 |            0.4375 |            0.3125 |              0.4612 |          1008.6000 | 0.3481 |
| QASPER    | P2                       |   4 |                        48 |            0.5000 |            0.3333 |              0.4935 |          1016.3200 | 0.3379 |
| QASPER    | P3                       |   4 |                        48 |            0.5000 |            0.3542 |              0.4961 |          1009.1200 | 0.3465 |
| QASPER    | P0                       |   6 |                        48 |            0.6250 |            0.4583 |              0.5911 |          1475.4000 | 0.3220 |
| QASPER    | P1                       |   6 |                        48 |            0.5833 |            0.5000 |              0.6245 |          1490.3400 | 0.3481 |
| QASPER    | P2                       |   6 |                        48 |            0.5833 |            0.4167 |              0.5855 |          1497.8600 | 0.3379 |
| QASPER    | P3                       |   6 |                        48 |            0.6042 |            0.5000 |              0.6352 |          1491.5400 | 0.3465 |
| QASPER    | P0                       |   8 |                        48 |            0.6875 |            0.5625 |              0.6751 |          1937.3600 | 0.3220 |
| QASPER    | P1                       |   8 |                        48 |            0.7083 |            0.5833 |              0.7107 |          1956.0400 | 0.3481 |
| QASPER    | P2                       |   8 |                        48 |            0.6667 |            0.5417 |              0.6942 |          1965.1400 | 0.3379 |
| QASPER    | P3                       |   8 |                        48 |            0.7083 |            0.5833 |              0.7017 |          1959.6600 | 0.3465 |
| QASPER    | P0                       |  10 |                        48 |            0.7708 |            0.6458 |              0.7488 |          2387.7200 | 0.3220 |
| QASPER    | P1                       |  10 |                        48 |            0.7708 |            0.6458 |              0.7678 |          2405.8000 | 0.3481 |
| QASPER    | P2                       |  10 |                        48 |            0.7500 |            0.6458 |              0.7604 |          2404.0800 | 0.3379 |
| QASPER    | P3                       |  10 |                        48 |            0.7500 |            0.6458 |              0.7509 |          2408.9800 | 0.3465 |
| SciFact   | P0                       |   1 |                        25 |            0.8000 |            0.8000 |              0.8000 |           327.7600 | 0.8530 |
| SciFact   | P1                       |   1 |                        25 |            0.7600 |            0.7600 |              0.7600 |           343.3600 | 0.8433 |
| SciFact   | P2                       |   1 |                        25 |            0.8400 |            0.8400 |              0.8400 |           357.5200 | 0.8878 |
| SciFact   | P3                       |   1 |                        25 |            0.8800 |            0.8800 |              0.8800 |           331.4200 | 0.9100 |
| SciFact   | P0                       |   2 |                        25 |            0.8400 |            0.8400 |              0.8400 |           641.6400 | 0.8530 |
| SciFact   | P1                       |   2 |                        25 |            0.8400 |            0.8400 |              0.8400 |           718.7600 | 0.8433 |
| SciFact   | P2                       |   2 |                        25 |            0.8800 |            0.8800 |              0.8800 |           697.3600 | 0.8878 |
| SciFact   | P3                       |   2 |                        25 |            0.8800 |            0.8800 |              0.8800 |           691.2400 | 0.9100 |
| SciFact   | P0                       |   4 |                        25 |            0.9200 |            0.9200 |              0.9200 |          1289.0000 | 0.8530 |
| SciFact   | P1                       |   4 |                        25 |            1.0000 |            1.0000 |              1.0000 |          1441.8600 | 0.8433 |
| SciFact   | P2                       |   4 |                        25 |            0.9600 |            0.9600 |              0.9600 |          1433.9000 | 0.8878 |
| SciFact   | P3                       |   4 |                        25 |            0.9600 |            0.9600 |              0.9600 |          1433.5200 | 0.9100 |
| SciFact   | P0                       |   6 |                        25 |            0.9200 |            0.9200 |              0.9200 |          1931.0000 | 0.8530 |
| SciFact   | P1                       |   6 |                        25 |            1.0000 |            1.0000 |              1.0000 |          2165.4600 | 0.8433 |
| SciFact   | P2                       |   6 |                        25 |            0.9600 |            0.9600 |              0.9600 |          2121.2400 | 0.8878 |
| SciFact   | P3                       |   6 |                        25 |            1.0000 |            1.0000 |              1.0000 |          2162.4400 | 0.9100 |
| SciFact   | P0                       |   8 |                        25 |            0.9600 |            0.9600 |              0.9600 |          2556.1400 | 0.8530 |
| SciFact   | P1                       |   8 |                        25 |            1.0000 |            1.0000 |              1.0000 |          2887.0400 | 0.8433 |
| SciFact   | P2                       |   8 |                        25 |            0.9600 |            0.9600 |              0.9600 |          2834.5400 | 0.8878 |
| SciFact   | P3                       |   8 |                        25 |            1.0000 |            1.0000 |              1.0000 |          2907.5400 | 0.9100 |
| SciFact   | P0                       |  10 |                        25 |            1.0000 |            1.0000 |              1.0000 |          3208.9400 | 0.8530 |
| SciFact   | P1                       |  10 |                        25 |            1.0000 |            1.0000 |              1.0000 |          3642.1800 | 0.8433 |
| SciFact   | P2                       |  10 |                        25 |            1.0000 |            1.0000 |              1.0000 |          3567.8000 | 0.8878 |
| SciFact   | P3                       |  10 |                        25 |            1.0000 |            1.0000 |              1.0000 |          3645.0800 | 0.9100 |
