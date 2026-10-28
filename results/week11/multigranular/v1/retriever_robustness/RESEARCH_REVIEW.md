# Week 11 F — retriever robustness: research review

## Completion and protocol

All 2,400 checksummed logical rows are complete: 150 frozen development queries, 50 per dataset, 48 dataset/condition summaries. Qwen is held fixed. Five manually reviewed representation/budget configurations are each evaluated under BM25, MiniLM and SFR, plus one no-evidence baseline per dataset. All policies use the same source corpus and pool-width policy (20), P0 passages, frozen semantic/hybrid downstream selection and decoding. No old result is overwritten.

The five controls are semantic passage/1024, semantic sentence/1024, semantic proposition/1024, hybrid sentence/256 and semantic passage/FULL. They were reviewed before this F run; they are exploratory finalists from the same development pilot, not independently validated winners. FULL is the capped candidate pool and context policy, not unrestricted corpus input.

## Observations

**HotpotQA does not identify a retriever effect in this setup.** All 15 matched pair/configuration contrasts have identical candidate sets, exact evidence hashes and predictions on all 50 queries. Width 20 already includes every provided distractor-context paragraph (10). The downstream semantic/hybrid selector then exposes identical evidence. These are repeated identical interventions, not evidence of general retriever robustness on open-corpus HotpotQA.

**QASPER retriever differences are small on this pilot.** At passage/1024, BM25 F1 is 0.201220, MiniLM 0.197917 and SFR 0.199183. At passage/FULL they are 0.249211, 0.241667 and 0.243652. Candidate-pool differences can affect exposed context, but neither these point estimates nor a zero-containing interval establish equivalence.

**SciFact shows larger point-estimate sensitivity.** At passage/1024, BM25 strict accuracy is 0.76, MiniLM 0.70 and SFR 0.68; the differences are only a few claims out of 50. Compact hybrid sentence/256 gives 0.74, 0.64 and 0.72 respectively. These are task-specific exploratory contrasts, not a universal retriever ranking. Full paired intervals below must govern any superiority claim.

HotpotQA passage/1024 remains 0.480272 F1 for all retrievers, versus no-evidence 0.150250. QASPER no-evidence is 0.091228 F1; SciFact no-evidence is 0.36 strict accuracy. QA F1 and verification accuracy are reported separately, never pooled.

## Matched passage/1024 retriever contrasts

| dataset   | granularity   | selector   |   budget | contrast          |   left_score |   right_score |     point |    ci_low |   ci_high |   resamples |   seed | cluster           |
|:----------|:--------------|:-----------|---------:|:------------------|-------------:|--------------:|----------:|----------:|----------:|------------:|-------:|:------------------|
| HotpotQA  | PASSAGE       | semantic   |     1024 | BM25_minus_MiniLM |     0.480272 |      0.480272 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | PASSAGE       | semantic   |     1024 | BM25_minus_SFR    |     0.480272 |      0.480272 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | PASSAGE       | semantic   |     1024 | MiniLM_minus_SFR  |     0.480272 |      0.480272 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| QASPER    | PASSAGE       | semantic   |     1024 | BM25_minus_MiniLM |     0.201220 |      0.197917 |  0.003303 | -0.000234 |  0.008289 |       10000 |     42 | paper_id          |
| QASPER    | PASSAGE       | semantic   |     1024 | BM25_minus_SFR    |     0.201220 |      0.199183 |  0.002036 | -0.000224 |  0.005319 |       10000 |     42 | paper_id          |
| QASPER    | PASSAGE       | semantic   |     1024 | MiniLM_minus_SFR  |     0.197917 |      0.199183 | -0.001266 | -0.005867 |  0.002042 |       10000 |     42 | paper_id          |
| SciFact   | PASSAGE       | semantic   |     1024 | BM25_minus_MiniLM |     0.760000 |      0.700000 |  0.060000 |  0.000000 |  0.140000 |       10000 |     42 | query_or_claim_id |
| SciFact   | PASSAGE       | semantic   |     1024 | BM25_minus_SFR    |     0.760000 |      0.680000 |  0.080000 |  0.000000 |  0.180000 |       10000 |     42 | query_or_claim_id |
| SciFact   | PASSAGE       | semantic   |     1024 | MiniLM_minus_SFR  |     0.700000 |      0.680000 |  0.020000 | -0.040000 |  0.080000 |       10000 |     42 | query_or_claim_id |

## All 45 matched retriever contrasts

Paired bootstrap uses 10,000 resamples, seed 42; QASPER clusters by paper (40 clusters for 50 queries), other datasets by query/claim (50). These are nominal exploratory 95% intervals without multiple-comparison adjustment. The contrast direction is explicitly named. No inference of superiority is made from point estimates alone.

| dataset   | granularity   | selector   | budget   | contrast          |   left_score |   right_score |     point |    ci_low |   ci_high |   resamples |   seed | cluster           |
|:----------|:--------------|:-----------|:---------|:------------------|-------------:|--------------:|----------:|----------:|----------:|------------:|-------:|:------------------|
| HotpotQA  | PASSAGE       | semantic   | 1024     | BM25_minus_MiniLM |     0.480272 |      0.480272 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | PASSAGE       | semantic   | 1024     | BM25_minus_SFR    |     0.480272 |      0.480272 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | PASSAGE       | semantic   | 1024     | MiniLM_minus_SFR  |     0.480272 |      0.480272 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | PASSAGE       | semantic   | FULL     | BM25_minus_MiniLM |     0.431145 |      0.431145 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | PASSAGE       | semantic   | FULL     | BM25_minus_SFR    |     0.431145 |      0.431145 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | PASSAGE       | semantic   | FULL     | MiniLM_minus_SFR  |     0.431145 |      0.431145 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | PROPOSITION   | semantic   | 1024     | BM25_minus_MiniLM |     0.475548 |      0.475548 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | PROPOSITION   | semantic   | 1024     | BM25_minus_SFR    |     0.475548 |      0.475548 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | PROPOSITION   | semantic   | 1024     | MiniLM_minus_SFR  |     0.475548 |      0.475548 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | SENTENCE      | hybrid     | 256      | BM25_minus_MiniLM |     0.362142 |      0.362142 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | SENTENCE      | hybrid     | 256      | BM25_minus_SFR    |     0.362142 |      0.362142 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | SENTENCE      | hybrid     | 256      | MiniLM_minus_SFR  |     0.362142 |      0.362142 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | SENTENCE      | semantic   | 1024     | BM25_minus_MiniLM |     0.472093 |      0.472093 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | SENTENCE      | semantic   | 1024     | BM25_minus_SFR    |     0.472093 |      0.472093 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | SENTENCE      | semantic   | 1024     | MiniLM_minus_SFR  |     0.472093 |      0.472093 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| QASPER    | PASSAGE       | semantic   | 1024     | BM25_minus_MiniLM |     0.201220 |      0.197917 |  0.003303 | -0.000234 |  0.008289 |       10000 |     42 | paper_id          |
| QASPER    | PASSAGE       | semantic   | 1024     | BM25_minus_SFR    |     0.201220 |      0.199183 |  0.002036 | -0.000224 |  0.005319 |       10000 |     42 | paper_id          |
| QASPER    | PASSAGE       | semantic   | 1024     | MiniLM_minus_SFR  |     0.197917 |      0.199183 | -0.001266 | -0.005867 |  0.002042 |       10000 |     42 | paper_id          |
| QASPER    | PASSAGE       | semantic   | FULL     | BM25_minus_MiniLM |     0.249211 |      0.241667 |  0.007544 | -0.012232 |  0.030364 |       10000 |     42 | paper_id          |
| QASPER    | PASSAGE       | semantic   | FULL     | BM25_minus_SFR    |     0.249211 |      0.243652 |  0.005559 | -0.006447 |  0.019946 |       10000 |     42 | paper_id          |
| QASPER    | PASSAGE       | semantic   | FULL     | MiniLM_minus_SFR  |     0.241667 |      0.243652 | -0.001985 | -0.030624 |  0.021593 |       10000 |     42 | paper_id          |
| QASPER    | PROPOSITION   | semantic   | 1024     | BM25_minus_MiniLM |     0.200371 |      0.203683 | -0.003312 | -0.015264 |  0.007744 |       10000 |     42 | paper_id          |
| QASPER    | PROPOSITION   | semantic   | 1024     | BM25_minus_SFR    |     0.200371 |      0.200596 | -0.000225 | -0.012516 |  0.009792 |       10000 |     42 | paper_id          |
| QASPER    | PROPOSITION   | semantic   | 1024     | MiniLM_minus_SFR  |     0.203683 |      0.200596 |  0.003087 | -0.011086 |  0.017273 |       10000 |     42 | paper_id          |
| QASPER    | SENTENCE      | hybrid     | 256      | BM25_minus_MiniLM |     0.192792 |      0.192379 |  0.000413 | -0.007639 |  0.009525 |       10000 |     42 | paper_id          |
| QASPER    | SENTENCE      | hybrid     | 256      | BM25_minus_SFR    |     0.192792 |      0.197049 | -0.004258 | -0.010655 |  0.000000 |       10000 |     42 | paper_id          |
| QASPER    | SENTENCE      | hybrid     | 256      | MiniLM_minus_SFR  |     0.192379 |      0.197049 | -0.004670 | -0.012187 |  0.000000 |       10000 |     42 | paper_id          |
| QASPER    | SENTENCE      | semantic   | 1024     | BM25_minus_MiniLM |     0.202240 |      0.204941 | -0.002702 | -0.013130 |  0.004697 |       10000 |     42 | paper_id          |
| QASPER    | SENTENCE      | semantic   | 1024     | BM25_minus_SFR    |     0.202240 |      0.204816 | -0.002577 | -0.010269 |  0.003609 |       10000 |     42 | paper_id          |
| QASPER    | SENTENCE      | semantic   | 1024     | MiniLM_minus_SFR  |     0.204941 |      0.204816 |  0.000125 | -0.008589 |  0.008904 |       10000 |     42 | paper_id          |
| SciFact   | PASSAGE       | semantic   | 1024     | BM25_minus_MiniLM |     0.760000 |      0.700000 |  0.060000 |  0.000000 |  0.140000 |       10000 |     42 | query_or_claim_id |
| SciFact   | PASSAGE       | semantic   | 1024     | BM25_minus_SFR    |     0.760000 |      0.680000 |  0.080000 |  0.000000 |  0.180000 |       10000 |     42 | query_or_claim_id |
| SciFact   | PASSAGE       | semantic   | 1024     | MiniLM_minus_SFR  |     0.700000 |      0.680000 |  0.020000 | -0.040000 |  0.080000 |       10000 |     42 | query_or_claim_id |
| SciFact   | PASSAGE       | semantic   | FULL     | BM25_minus_MiniLM |     0.640000 |      0.700000 | -0.060000 | -0.180000 |  0.060000 |       10000 |     42 | query_or_claim_id |
| SciFact   | PASSAGE       | semantic   | FULL     | BM25_minus_SFR    |     0.640000 |      0.660000 | -0.020000 | -0.120000 |  0.080000 |       10000 |     42 | query_or_claim_id |
| SciFact   | PASSAGE       | semantic   | FULL     | MiniLM_minus_SFR  |     0.700000 |      0.660000 |  0.040000 | -0.060000 |  0.160000 |       10000 |     42 | query_or_claim_id |
| SciFact   | PROPOSITION   | semantic   | 1024     | BM25_minus_MiniLM |     0.660000 |      0.620000 |  0.040000 | -0.040000 |  0.120000 |       10000 |     42 | query_or_claim_id |
| SciFact   | PROPOSITION   | semantic   | 1024     | BM25_minus_SFR    |     0.660000 |      0.620000 |  0.040000 | -0.040000 |  0.120000 |       10000 |     42 | query_or_claim_id |
| SciFact   | PROPOSITION   | semantic   | 1024     | MiniLM_minus_SFR  |     0.620000 |      0.620000 |  0.000000 | -0.080000 |  0.080000 |       10000 |     42 | query_or_claim_id |
| SciFact   | SENTENCE      | hybrid     | 256      | BM25_minus_MiniLM |     0.740000 |      0.640000 |  0.100000 |  0.000000 |  0.200000 |       10000 |     42 | query_or_claim_id |
| SciFact   | SENTENCE      | hybrid     | 256      | BM25_minus_SFR    |     0.740000 |      0.720000 |  0.020000 | -0.060000 |  0.100000 |       10000 |     42 | query_or_claim_id |
| SciFact   | SENTENCE      | hybrid     | 256      | MiniLM_minus_SFR  |     0.640000 |      0.720000 | -0.080000 | -0.160000 | -0.020000 |       10000 |     42 | query_or_claim_id |
| SciFact   | SENTENCE      | semantic   | 1024     | BM25_minus_MiniLM |     0.660000 |      0.640000 |  0.020000 | -0.060000 |  0.100000 |       10000 |     42 | query_or_claim_id |
| SciFact   | SENTENCE      | semantic   | 1024     | BM25_minus_SFR    |     0.660000 |      0.620000 |  0.040000 | -0.040000 |  0.120000 |       10000 |     42 | query_or_claim_id |
| SciFact   | SENTENCE      | semantic   | 1024     | MiniLM_minus_SFR  |     0.640000 |      0.620000 |  0.020000 | -0.060000 |  0.100000 |       10000 |     42 | query_or_claim_id |

## Candidate/evidence identity diagnostic

| dataset   | granularity   | selector   | budget   | comparison     |   queries |   same_candidate_set |   same_evidence |   same_prediction |
|:----------|:--------------|:-----------|:---------|:---------------|----------:|---------------------:|----------------:|------------------:|
| HotpotQA  | PASSAGE       | semantic   | 1024     | BM25_vs_MiniLM |        50 |                   50 |              50 |                50 |
| HotpotQA  | PASSAGE       | semantic   | 1024     | BM25_vs_SFR    |        50 |                   50 |              50 |                50 |
| HotpotQA  | PASSAGE       | semantic   | 1024     | MiniLM_vs_SFR  |        50 |                   50 |              50 |                50 |
| HotpotQA  | PASSAGE       | semantic   | FULL     | BM25_vs_MiniLM |        50 |                   50 |              50 |                50 |
| HotpotQA  | PASSAGE       | semantic   | FULL     | BM25_vs_SFR    |        50 |                   50 |              50 |                50 |
| HotpotQA  | PASSAGE       | semantic   | FULL     | MiniLM_vs_SFR  |        50 |                   50 |              50 |                50 |
| HotpotQA  | PROPOSITION   | semantic   | 1024     | BM25_vs_MiniLM |        50 |                   50 |              50 |                50 |
| HotpotQA  | PROPOSITION   | semantic   | 1024     | BM25_vs_SFR    |        50 |                   50 |              50 |                50 |
| HotpotQA  | PROPOSITION   | semantic   | 1024     | MiniLM_vs_SFR  |        50 |                   50 |              50 |                50 |
| HotpotQA  | SENTENCE      | hybrid     | 256      | BM25_vs_MiniLM |        50 |                   50 |              50 |                50 |
| HotpotQA  | SENTENCE      | hybrid     | 256      | BM25_vs_SFR    |        50 |                   50 |              50 |                50 |
| HotpotQA  | SENTENCE      | hybrid     | 256      | MiniLM_vs_SFR  |        50 |                   50 |              50 |                50 |
| HotpotQA  | SENTENCE      | semantic   | 1024     | BM25_vs_MiniLM |        50 |                   50 |              50 |                50 |
| HotpotQA  | SENTENCE      | semantic   | 1024     | BM25_vs_SFR    |        50 |                   50 |              50 |                50 |
| HotpotQA  | SENTENCE      | semantic   | 1024     | MiniLM_vs_SFR  |        50 |                   50 |              50 |                50 |
| QASPER    | PASSAGE       | semantic   | 1024     | BM25_vs_MiniLM |        50 |                   25 |              45 |                47 |
| QASPER    | PASSAGE       | semantic   | 1024     | BM25_vs_SFR    |        50 |                   26 |              46 |                47 |
| QASPER    | PASSAGE       | semantic   | 1024     | MiniLM_vs_SFR  |        50 |                   26 |              46 |                48 |
| QASPER    | PASSAGE       | semantic   | FULL     | BM25_vs_MiniLM |        50 |                   25 |              25 |                38 |
| QASPER    | PASSAGE       | semantic   | FULL     | BM25_vs_SFR    |        50 |                   26 |              26 |                37 |
| QASPER    | PASSAGE       | semantic   | FULL     | MiniLM_vs_SFR  |        50 |                   26 |              26 |                40 |
| QASPER    | PROPOSITION   | semantic   | 1024     | BM25_vs_MiniLM |        50 |                   25 |              34 |                39 |
| QASPER    | PROPOSITION   | semantic   | 1024     | BM25_vs_SFR    |        50 |                   26 |              32 |                37 |
| QASPER    | PROPOSITION   | semantic   | 1024     | MiniLM_vs_SFR  |        50 |                   26 |              35 |                40 |
| QASPER    | SENTENCE      | hybrid     | 256      | BM25_vs_MiniLM |        50 |                   25 |              38 |                43 |
| QASPER    | SENTENCE      | hybrid     | 256      | BM25_vs_SFR    |        50 |                   26 |              42 |                45 |
| QASPER    | SENTENCE      | hybrid     | 256      | MiniLM_vs_SFR  |        50 |                   26 |              41 |                47 |
| QASPER    | SENTENCE      | semantic   | 1024     | BM25_vs_MiniLM |        50 |                   25 |              35 |                42 |
| QASPER    | SENTENCE      | semantic   | 1024     | BM25_vs_SFR    |        50 |                   26 |              33 |                42 |
| QASPER    | SENTENCE      | semantic   | 1024     | MiniLM_vs_SFR  |        50 |                   26 |              36 |                43 |
| SciFact   | PASSAGE       | semantic   | 1024     | BM25_vs_MiniLM |        50 |                    0 |              16 |                47 |
| SciFact   | PASSAGE       | semantic   | 1024     | BM25_vs_SFR    |        50 |                    0 |              18 |                44 |
| SciFact   | PASSAGE       | semantic   | 1024     | MiniLM_vs_SFR  |        50 |                    0 |              25 |                47 |
| SciFact   | PASSAGE       | semantic   | FULL     | BM25_vs_MiniLM |        50 |                    0 |               0 |                41 |
| SciFact   | PASSAGE       | semantic   | FULL     | BM25_vs_SFR    |        50 |                    0 |               0 |                43 |
| SciFact   | PASSAGE       | semantic   | FULL     | MiniLM_vs_SFR  |        50 |                    0 |               0 |                42 |
| SciFact   | PROPOSITION   | semantic   | 1024     | BM25_vs_MiniLM |        50 |                    0 |               0 |                42 |
| SciFact   | PROPOSITION   | semantic   | 1024     | BM25_vs_SFR    |        50 |                    0 |               0 |                43 |
| SciFact   | PROPOSITION   | semantic   | 1024     | MiniLM_vs_SFR  |        50 |                    0 |               0 |                45 |
| SciFact   | SENTENCE      | hybrid     | 256      | BM25_vs_MiniLM |        50 |                    0 |               6 |                41 |
| SciFact   | SENTENCE      | hybrid     | 256      | BM25_vs_SFR    |        50 |                    0 |               6 |                44 |
| SciFact   | SENTENCE      | hybrid     | 256      | MiniLM_vs_SFR  |        50 |                    0 |               5 |                44 |
| SciFact   | SENTENCE      | semantic   | 1024     | BM25_vs_MiniLM |        50 |                    0 |               0 |                42 |
| SciFact   | SENTENCE      | semantic   | 1024     | BM25_vs_SFR    |        50 |                    0 |               0 |                43 |
| SciFact   | SENTENCE      | semantic   | 1024     | MiniLM_vs_SFR  |        50 |                    0 |               0 |                44 |

Candidate identity compares sets rather than retrieval rank order. Evidence hashes check the actual generator exposure; prediction equality is separate. Shared predictions from a cache are not independent repeated model experiments.

## Flags and cost

The mechanical audit found zero empty-evidence rows, zero generation-cap hits and zero invalid SciFact labels. There are 112 annotation-mapping flag rows: the same seven QASPER questions across 16 conditions, not 112 distinct failed queries. Gold mapping affects retrospective evidence metrics, not sentence/passage ranking or answer utility scoring. Full identifiers remain in flagged_queries.csv.

There were 519 physical generation calls for 2,400 logical policy/query rows; 1,881 reused compatible cached outputs. Conceptual policy inference cost remains one answer-generation call even on a cache hit. Selection/embedding preprocessing is not free. Final/cumulative prompt tokens count actual LM input, not all corpus text in memory. Cached timing fields exclude preparation/loading/IPC overhead; end-to-end serving latency was not measured and remains null. No wall-clock speed superiority is claimed.

## Continuation and limits

Retain Qwen/SFR for G exactly as planned, rather than silently changing the controlled hierarchy experiment after observing these F scores. F does not establish a universal retriever winner or open-corpus robustness. The 50-per-dataset cohort is a development pilot; stronger claims need larger held-out confirmation. Optional LLMLingua-2 remains pending elsewhere; F itself has no pending executable conditions.

## Complete 48-condition table

| dataset   | retriever   | evidence_granularity   | selector   | budget   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |   prompt_tokens |   generated_tokens |
|:----------|:------------|:-----------------------|:-----------|:---------|-------------:|-------------------:|---------:|---------:|-------------------------:|----------------:|-------------------:|
| HotpotQA  | SFR         | NONE                   | none       | 0        |     0.150250 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       52.800000 |          13.820000 |
| HotpotQA  | BM25        | PASSAGE                | semantic   | 1024     |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 |      851.160000 |          11.980000 |
| HotpotQA  | MiniLM      | PASSAGE                | semantic   | 1024     |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 |      851.160000 |          11.980000 |
| HotpotQA  | SFR         | PASSAGE                | semantic   | 1024     |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 |      851.160000 |          11.980000 |
| HotpotQA  | BM25        | PASSAGE                | semantic   | FULL     |     0.431145 |           0.280895 | 0.580000 | 0.060000 |              1351.060000 |     1403.860000 |          13.520000 |
| HotpotQA  | MiniLM      | PASSAGE                | semantic   | FULL     |     0.431145 |           0.280895 | 0.580000 | 0.060000 |              1351.060000 |     1403.860000 |          13.520000 |
| HotpotQA  | SFR         | PASSAGE                | semantic   | FULL     |     0.431145 |           0.280895 | 0.580000 | 0.060000 |              1351.060000 |     1403.860000 |          13.520000 |
| HotpotQA  | BM25        | PROPOSITION            | semantic   | 1024     |     0.475548 |           0.325299 | 0.600000 | 0.040000 |               865.040000 |      917.840000 |          10.760000 |
| HotpotQA  | MiniLM      | PROPOSITION            | semantic   | 1024     |     0.475548 |           0.325299 | 0.600000 | 0.040000 |               865.040000 |      917.840000 |          10.760000 |
| HotpotQA  | SFR         | PROPOSITION            | semantic   | 1024     |     0.475548 |           0.325299 | 0.600000 | 0.040000 |               865.040000 |      917.840000 |          10.760000 |
| HotpotQA  | BM25        | SENTENCE               | hybrid     | 256      |     0.362142 |           0.211892 | 0.460000 | 0.060000 |               205.120000 |      257.920000 |          11.260000 |
| HotpotQA  | MiniLM      | SENTENCE               | hybrid     | 256      |     0.362142 |           0.211892 | 0.460000 | 0.060000 |               205.120000 |      257.920000 |          11.260000 |
| HotpotQA  | SFR         | SENTENCE               | hybrid     | 256      |     0.362142 |           0.211892 | 0.460000 | 0.060000 |               205.120000 |      257.920000 |          11.260000 |
| HotpotQA  | BM25        | SENTENCE               | semantic   | 1024     |     0.472093 |           0.321844 | 0.620000 | 0.040000 |               866.740000 |      919.540000 |          11.060000 |
| HotpotQA  | MiniLM      | SENTENCE               | semantic   | 1024     |     0.472093 |           0.321844 | 0.620000 | 0.040000 |               866.740000 |      919.540000 |          11.060000 |
| HotpotQA  | SFR         | SENTENCE               | semantic   | 1024     |     0.472093 |           0.321844 | 0.620000 | 0.040000 |               866.740000 |      919.540000 |          11.060000 |
| QASPER    | SFR         | NONE                   | none       | 0        |     0.091228 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       41.980000 |          18.160000 |
| QASPER    | BM25        | PASSAGE                | semantic   | 1024     |     0.201220 |           0.109992 | 0.600000 | 0.100000 |               771.700000 |      814.320000 |          20.880000 |
| QASPER    | MiniLM      | PASSAGE                | semantic   | 1024     |     0.197917 |           0.106689 | 0.600000 | 0.100000 |               772.960000 |      815.600000 |          20.260000 |
| QASPER    | SFR         | PASSAGE                | semantic   | 1024     |     0.199183 |           0.107956 | 0.600000 | 0.100000 |               773.900000 |      816.520000 |          20.380000 |
| QASPER    | BM25        | PASSAGE                | semantic   | FULL     |     0.249211 |           0.157984 | 0.720000 | 0.060000 |              4144.320000 |     4186.440000 |          22.360000 |
| QASPER    | MiniLM      | PASSAGE                | semantic   | FULL     |     0.241667 |           0.150440 | 0.720000 | 0.060000 |              4135.880000 |     4178.020000 |          20.860000 |
| QASPER    | SFR         | PASSAGE                | semantic   | FULL     |     0.243652 |           0.152425 | 0.740000 | 0.060000 |              4144.280000 |     4186.440000 |          21.200000 |
| QASPER    | BM25        | PROPOSITION            | semantic   | 1024     |     0.200371 |           0.109144 | 0.580000 | 0.160000 |               872.920000 |      914.980000 |          20.600000 |
| QASPER    | MiniLM      | PROPOSITION            | semantic   | 1024     |     0.203683 |           0.112456 | 0.600000 | 0.140000 |               872.640000 |      914.720000 |          20.400000 |
| QASPER    | SFR         | PROPOSITION            | semantic   | 1024     |     0.200596 |           0.109368 | 0.560000 | 0.140000 |               870.780000 |      912.860000 |          20.400000 |
| QASPER    | BM25        | SENTENCE               | hybrid     | 256      |     0.192792 |           0.101564 | 0.500000 | 0.220000 |               200.020000 |      242.060000 |          17.660000 |
| QASPER    | MiniLM      | SENTENCE               | hybrid     | 256      |     0.192379 |           0.101151 | 0.480000 | 0.220000 |               200.360000 |      242.400000 |          17.380000 |
| QASPER    | SFR         | SENTENCE               | hybrid     | 256      |     0.197049 |           0.105822 | 0.500000 | 0.200000 |               200.560000 |      242.600000 |          17.620000 |
| QASPER    | BM25        | SENTENCE               | semantic   | 1024     |     0.202240 |           0.111012 | 0.560000 | 0.160000 |               871.420000 |      913.480000 |          20.580000 |
| QASPER    | MiniLM      | SENTENCE               | semantic   | 1024     |     0.204941 |           0.113714 | 0.580000 | 0.140000 |               870.920000 |      912.980000 |          20.720000 |
| QASPER    | SFR         | SENTENCE               | semantic   | 1024     |     0.204816 |           0.113589 | 0.560000 | 0.140000 |               872.960000 |      915.020000 |          20.680000 |
| SciFact   | SFR         | NONE                   | none       | 0        |     0.360000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       58.300000 |           3.720000 |
| SciFact   | BM25        | PASSAGE                | semantic   | 1024     |     0.760000 |           0.400000 | 0.520000 | 0.120000 |               655.000000 |      713.320000 |           3.840000 |
| SciFact   | MiniLM      | PASSAGE                | semantic   | 1024     |     0.700000 |           0.340000 | 0.460000 | 0.120000 |               657.200000 |      715.520000 |           3.720000 |
| SciFact   | SFR         | PASSAGE                | semantic   | 1024     |     0.680000 |           0.320000 | 0.440000 | 0.120000 |               667.620000 |      725.940000 |           3.680000 |
| SciFact   | BM25        | PASSAGE                | semantic   | FULL     |     0.640000 |           0.280000 | 0.360000 | 0.080000 |              4868.960000 |     4927.260000 |           3.880000 |
| SciFact   | MiniLM      | PASSAGE                | semantic   | FULL     |     0.700000 |           0.340000 | 0.460000 | 0.120000 |              4954.120000 |     5012.460000 |           3.880000 |
| SciFact   | SFR         | PASSAGE                | semantic   | FULL     |     0.660000 |           0.300000 | 0.420000 | 0.120000 |              4936.440000 |     4994.740000 |           3.840000 |
| SciFact   | BM25        | PROPOSITION            | semantic   | 1024     |     0.660000 |           0.300000 | 0.460000 | 0.160000 |               808.320000 |      866.640000 |           3.600000 |
| SciFact   | MiniLM      | PROPOSITION            | semantic   | 1024     |     0.620000 |           0.260000 | 0.400000 | 0.140000 |               816.700000 |      875.000000 |           3.480000 |
| SciFact   | SFR         | PROPOSITION            | semantic   | 1024     |     0.620000 |           0.260000 | 0.420000 | 0.160000 |               819.320000 |      877.620000 |           3.480000 |
| SciFact   | BM25        | SENTENCE               | hybrid     | 256      |     0.740000 |           0.380000 | 0.460000 | 0.080000 |               188.280000 |      246.580000 |           3.720000 |
| SciFact   | MiniLM      | SENTENCE               | hybrid     | 256      |     0.640000 |           0.280000 | 0.460000 | 0.180000 |               186.680000 |      244.980000 |           3.680000 |
| SciFact   | SFR         | SENTENCE               | hybrid     | 256      |     0.720000 |           0.360000 | 0.480000 | 0.120000 |               188.460000 |      246.780000 |           3.680000 |
| SciFact   | BM25        | SENTENCE               | semantic   | 1024     |     0.660000 |           0.300000 | 0.460000 | 0.160000 |               815.220000 |      873.540000 |           3.600000 |
| SciFact   | MiniLM      | SENTENCE               | semantic   | 1024     |     0.640000 |           0.280000 | 0.400000 | 0.120000 |               817.740000 |      876.040000 |           3.520000 |
| SciFact   | SFR         | SENTENCE               | semantic   | 1024     |     0.620000 |           0.260000 | 0.420000 | 0.160000 |               824.500000 |      882.800000 |           3.480000 |
