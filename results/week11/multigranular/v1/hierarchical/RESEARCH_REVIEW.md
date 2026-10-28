# Week 11 G — deterministic hierarchical exposure: research review

## Completion and controlled experiment

The available surface contains 3,150 checksummed rows: the same 150 development queries, Qwen/SFR/P0, top-20 capped passage pools, one no-evidence baseline and five available strategies at 256/512/1024/FULL. H4 (LLMLingua-2 tail compression) remains pending: 600 configured rows. This is a frozen deterministic hierarchy prototype, not a trained dynamic controller.

| Strategy | Mechanism |
|---|---|
| H0 | SFR-ranked whole passages, strict budget prefix |
| H1 | First ranked passage in full, then semantic sentences from remaining passages |
| H2 | First ranked passage in full, then semantic exact-clause propositions from remaining passages |
| H3 | Semantic sentences from first three ranked passages, then semantic propositions from remaining passages |
| H4 | Full head + sentence middle + token-compressed tail; pending, no results |
| H5 | Token-density ranking over first full passage plus all candidate sentences/propositions, gamma=1 |

All available exposures are exact source spans, overlap is suppressed, and selected spans are presented in original source order. Numeric budgets use the shared maximum across the three pinned reader tokenizers. Strict allocation stops at the first oversized unit, so H1/H2 can inherit the same empty-context failure as H0 when their first full passage exceeds budget. FULL remains capped; propositions are deterministic clauses, not generated facts. No outcome or gold label enters selection.

## Best observed point per strategy

These budget peaks are retrospective within the same 50-per-dataset pilot. QA utility is F1; SciFact is strict label accuracy. They are descriptive and do not validate a budget selector.

| dataset   | strategy   | budget   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |   prompt_tokens |   generated_tokens |
|:----------|:-----------|:---------|-------------:|-------------------:|---------:|---------:|-------------------------:|----------------:|-------------------:|
| HotpotQA  | H0         | 1024     |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 |      851.160000 |          11.980000 |
| HotpotQA  | H1         | 1024     |     0.479379 |           0.329129 | 0.620000 | 0.040000 |               867.640000 |      920.440000 |          11.380000 |
| HotpotQA  | H2         | 1024     |     0.475304 |           0.325054 | 0.620000 | 0.040000 |               868.060000 |      920.860000 |          11.620000 |
| HotpotQA  | H3         | 1024     |     0.487169 |           0.336919 | 0.640000 | 0.040000 |               864.260000 |      917.060000 |          11.120000 |
| HotpotQA  | H5         | FULL     |     0.405590 |           0.255340 | 0.580000 | 0.060000 |              1350.500000 |     1403.300000 |          13.080000 |
| QASPER    | H0         | FULL     |     0.243652 |           0.152425 | 0.740000 | 0.060000 |              4144.280000 |     4186.440000 |          21.200000 |
| QASPER    | H1         | FULL     |     0.243726 |           0.152498 | 0.720000 | 0.060000 |              4146.780000 |     4188.940000 |          21.060000 |
| QASPER    | H2         | FULL     |     0.245822 |           0.154594 | 0.720000 | 0.080000 |              4146.420000 |     4188.580000 |          21.460000 |
| QASPER    | H3         | FULL     |     0.242887 |           0.151659 | 0.720000 | 0.080000 |              4146.800000 |     4188.960000 |          21.460000 |
| QASPER    | H5         | FULL     |     0.242887 |           0.151659 | 0.720000 | 0.080000 |              4146.660000 |     4188.820000 |          21.460000 |
| SciFact   | H0         | 1024     |     0.680000 |           0.320000 | 0.440000 | 0.120000 |               667.620000 |      725.940000 |           3.680000 |
| SciFact   | H1         | FULL     |     0.660000 |           0.300000 | 0.440000 | 0.140000 |              4942.700000 |     5001.000000 |           3.720000 |
| SciFact   | H2         | FULL     |     0.680000 |           0.320000 | 0.460000 | 0.140000 |              4928.660000 |     4986.960000 |           3.720000 |
| SciFact   | H3         | 256      |     0.700000 |           0.340000 | 0.440000 | 0.100000 |               182.840000 |      241.200000 |           3.720000 |
| SciFact   | H5         | FULL     |     0.700000 |           0.340000 | 0.480000 | 0.140000 |              4929.840000 |     4988.140000 |           3.760000 |

## Interpretation

H3 at 256 requested tokens improves over raw H0 on all three datasets: QASPER +0.085380 F1, CI [0.031609, 0.143167]; HotpotQA +0.099869 F1, CI [0.025947, 0.180600]; SciFact +0.32 accuracy, CI [0.14, 0.50]. These are the only three hierarchy-minus-H0 intervals with strictly positive lower bounds. The contrast includes avoiding oversized first passages and exposing smaller units; it is not a clean causal demonstration that mixing granularities itself is necessary.

Against fixed semantic sentences at 256, H3 differences are QASPER −0.028979 (CI [−0.082065, 0.024135]), HotpotQA +0.016114 ([−0.062096, 0.093277]) and SciFact +0.02 ([−0.10, 0.14]). All include zero. Thus the supported low-budget finding is compact source exposure over raw strict-prefix passages; a hierarchy advantage over simple sentence selection is not established.

At 1024, H3 versus H0 is QASPER −0.017076 (CI [−0.050733, 0.014475]), HotpotQA +0.006897 ([−0.051911, 0.068590]) and SciFact −0.02 ([−0.06, 0.00]). The HotpotQA best point estimate of 0.487169 therefore does not establish superiority over raw 0.480272. QASPER FULL strategies converge around 0.243–0.246, also offering no demonstrated hierarchy breakthrough.

H0 exactly reproduces all 600 corresponding C semantic-passage evidence blocks and predictions, confirming the raw control. The oracle headroom is substantial but retrospective: it motivates carefully tested policy research, not a claim that adaptive control already works.

## Matched hierarchy versus H0

Each row pairs the same queries, budget, reader and retriever. Bootstrap: 10,000 resamples, seed 42; QASPER paper clusters (40), others query/claim clusters (50). All intervals are nominal exploratory 95% intervals without correction for multiple comparisons.

| dataset   | budget   | strategy   | contrast    |   candidate_score |   control_score |     point |    ci_low |   ci_high |   resamples |   seed | cluster           |
|:----------|:---------|:-----------|:------------|------------------:|----------------:|----------:|----------:|----------:|------------:|-------:|:------------------|
| QASPER    | 256      | H1         | H1_minus_H0 |          0.103456 |        0.096925 |  0.006531 |  0.000000 |  0.016538 |       10000 |     42 | paper_id          |
| QASPER    | 256      | H2         | H2_minus_H0 |          0.103456 |        0.096925 |  0.006531 |  0.000000 |  0.016538 |       10000 |     42 | paper_id          |
| QASPER    | 256      | H3         | H3_minus_H0 |          0.182305 |        0.096925 |  0.085380 |  0.031609 |  0.143167 |       10000 |     42 | paper_id          |
| QASPER    | 256      | H5         | H5_minus_H0 |          0.118661 |        0.096925 |  0.021736 | -0.023567 |  0.066686 |       10000 |     42 | paper_id          |
| QASPER    | 512      | H1         | H1_minus_H0 |          0.196084 |        0.158710 |  0.037374 | -0.014745 |  0.089010 |       10000 |     42 | paper_id          |
| QASPER    | 512      | H2         | H2_minus_H0 |          0.183501 |        0.158710 |  0.024791 | -0.030272 |  0.079599 |       10000 |     42 | paper_id          |
| QASPER    | 512      | H3         | H3_minus_H0 |          0.169549 |        0.158710 |  0.010840 | -0.037417 |  0.058570 |       10000 |     42 | paper_id          |
| QASPER    | 512      | H5         | H5_minus_H0 |          0.131073 |        0.158710 | -0.027637 | -0.066470 |  0.010611 |       10000 |     42 | paper_id          |
| QASPER    | 1024     | H1         | H1_minus_H0 |          0.195769 |        0.199183 | -0.003414 | -0.048859 |  0.042878 |       10000 |     42 | paper_id          |
| QASPER    | 1024     | H2         | H2_minus_H0 |          0.193604 |        0.199183 | -0.005579 | -0.050851 |  0.040636 |       10000 |     42 | paper_id          |
| QASPER    | 1024     | H3         | H3_minus_H0 |          0.182107 |        0.199183 | -0.017076 | -0.050733 |  0.014475 |       10000 |     42 | paper_id          |
| QASPER    | 1024     | H5         | H5_minus_H0 |          0.188630 |        0.199183 | -0.010553 | -0.058254 |  0.037823 |       10000 |     42 | paper_id          |
| QASPER    | FULL     | H1         | H1_minus_H0 |          0.243726 |        0.243652 |  0.000074 | -0.026191 |  0.021900 |       10000 |     42 | paper_id          |
| QASPER    | FULL     | H2         | H2_minus_H0 |          0.245822 |        0.243652 |  0.002170 | -0.010952 |  0.017461 |       10000 |     42 | paper_id          |
| QASPER    | FULL     | H3         | H3_minus_H0 |          0.242887 |        0.243652 | -0.000765 | -0.013450 |  0.012118 |       10000 |     42 | paper_id          |
| QASPER    | FULL     | H5         | H5_minus_H0 |          0.242887 |        0.243652 | -0.000765 | -0.013450 |  0.012118 |       10000 |     42 | paper_id          |
| HotpotQA  | 256      | H1         | H1_minus_H0 |          0.380352 |        0.337510 |  0.042842 | -0.039477 |  0.131247 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 256      | H2         | H2_minus_H0 |          0.387303 |        0.337510 |  0.049793 | -0.040404 |  0.141519 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 256      | H3         | H3_minus_H0 |          0.437379 |        0.337510 |  0.099869 |  0.025947 |  0.180600 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 256      | H5         | H5_minus_H0 |          0.287253 |        0.337510 | -0.050257 | -0.163653 |  0.061470 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 512      | H1         | H1_minus_H0 |          0.414020 |        0.456280 | -0.042260 | -0.116814 |  0.030462 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 512      | H2         | H2_minus_H0 |          0.404341 |        0.456280 | -0.051939 | -0.128437 |  0.022767 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 512      | H3         | H3_minus_H0 |          0.442910 |        0.456280 | -0.013370 | -0.068931 |  0.045875 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 512      | H5         | H5_minus_H0 |          0.346082 |        0.456280 | -0.110197 | -0.189884 | -0.040292 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 1024     | H1         | H1_minus_H0 |          0.479379 |        0.480272 | -0.000893 | -0.070619 |  0.067101 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 1024     | H2         | H2_minus_H0 |          0.475304 |        0.480272 | -0.004968 | -0.075347 |  0.063314 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 1024     | H3         | H3_minus_H0 |          0.487169 |        0.480272 |  0.006897 | -0.051911 |  0.068590 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 1024     | H5         | H5_minus_H0 |          0.391268 |        0.480272 | -0.089004 | -0.161290 | -0.025617 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | FULL     | H1         | H1_minus_H0 |          0.403771 |        0.431145 | -0.027374 | -0.068636 |  0.004444 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | FULL     | H2         | H2_minus_H0 |          0.419882 |        0.431145 | -0.011263 | -0.055556 |  0.028030 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | FULL     | H3         | H3_minus_H0 |          0.410590 |        0.431145 | -0.020556 | -0.060556 |  0.008889 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | FULL     | H5         | H5_minus_H0 |          0.405590 |        0.431145 | -0.025556 | -0.065569 |  0.004444 |       10000 |     42 | query_or_claim_id |
| SciFact   | 256      | H1         | H1_minus_H0 |          0.380000 |        0.380000 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 256      | H2         | H2_minus_H0 |          0.380000 |        0.380000 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 256      | H3         | H3_minus_H0 |          0.700000 |        0.380000 |  0.320000 |  0.140000 |  0.500000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 256      | H5         | H5_minus_H0 |          0.400000 |        0.380000 |  0.020000 | -0.180000 |  0.220000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 512      | H1         | H1_minus_H0 |          0.620000 |        0.660000 | -0.040000 | -0.160000 |  0.060000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 512      | H2         | H2_minus_H0 |          0.600000 |        0.660000 | -0.060000 | -0.180000 |  0.060000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 512      | H3         | H3_minus_H0 |          0.700000 |        0.660000 |  0.040000 | -0.080000 |  0.160000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 512      | H5         | H5_minus_H0 |          0.460000 |        0.660000 | -0.200000 | -0.340000 | -0.060000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 1024     | H1         | H1_minus_H0 |          0.620000 |        0.680000 | -0.060000 | -0.200000 |  0.060000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 1024     | H2         | H2_minus_H0 |          0.580000 |        0.680000 | -0.100000 | -0.220000 |  0.020000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 1024     | H3         | H3_minus_H0 |          0.660000 |        0.680000 | -0.020000 | -0.060000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 1024     | H5         | H5_minus_H0 |          0.460000 |        0.680000 | -0.220000 | -0.360000 | -0.080000 |       10000 |     42 | query_or_claim_id |
| SciFact   | FULL     | H1         | H1_minus_H0 |          0.660000 |        0.660000 |  0.000000 | -0.080000 |  0.080000 |       10000 |     42 | query_or_claim_id |
| SciFact   | FULL     | H2         | H2_minus_H0 |          0.680000 |        0.660000 |  0.020000 | -0.060000 |  0.100000 |       10000 |     42 | query_or_claim_id |
| SciFact   | FULL     | H3         | H3_minus_H0 |          0.660000 |        0.660000 |  0.000000 | -0.100000 |  0.100000 |       10000 |     42 | query_or_claim_id |
| SciFact   | FULL     | H5         | H5_minus_H0 |          0.700000 |        0.660000 |  0.040000 | -0.060000 |  0.140000 |       10000 |     42 | query_or_claim_id |

3 of 48 hierarchy-minus-H0 intervals have a strictly positive lower bound; 4 have a strictly negative upper bound. These counts are exploratory, not multiplicity-adjusted discoveries.

## Fixed-granularity controls from C

These controls hold Qwen, SFR/P0, query, budget and candidate-pool policy fixed and compare against semantic sentence/proposition allocation. They distinguish a hierarchy effect from merely exposing smaller source units. The additional 96 intervals are exploratory.

| dataset   | budget   | strategy   | contrast                   |   candidate_score |   control_score |     point |    ci_low |   ci_high |   resamples |   seed | cluster           |
|:----------|:---------|:-----------|:---------------------------|------------------:|----------------:|----------:|----------:|----------:|------------:|-------:|:------------------|
| QASPER    | 256      | H1         | H1_minus_fixed_SENTENCE    |          0.103456 |        0.211284 | -0.107828 | -0.168338 | -0.047016 |       10000 |     42 | paper_id          |
| QASPER    | 256      | H2         | H2_minus_fixed_SENTENCE    |          0.103456 |        0.211284 | -0.107828 | -0.168338 | -0.047016 |       10000 |     42 | paper_id          |
| QASPER    | 256      | H3         | H3_minus_fixed_SENTENCE    |          0.182305 |        0.211284 | -0.028979 | -0.082065 |  0.024135 |       10000 |     42 | paper_id          |
| QASPER    | 256      | H5         | H5_minus_fixed_SENTENCE    |          0.118661 |        0.211284 | -0.092623 | -0.152586 | -0.031998 |       10000 |     42 | paper_id          |
| QASPER    | 256      | H1         | H1_minus_fixed_PROPOSITION |          0.103456 |        0.209378 | -0.105922 | -0.167268 | -0.042551 |       10000 |     42 | paper_id          |
| QASPER    | 256      | H2         | H2_minus_fixed_PROPOSITION |          0.103456 |        0.209378 | -0.105922 | -0.167268 | -0.042551 |       10000 |     42 | paper_id          |
| QASPER    | 256      | H3         | H3_minus_fixed_PROPOSITION |          0.182305 |        0.209378 | -0.027073 | -0.080202 |  0.025382 |       10000 |     42 | paper_id          |
| QASPER    | 256      | H5         | H5_minus_fixed_PROPOSITION |          0.118661 |        0.209378 | -0.090718 | -0.151594 | -0.029356 |       10000 |     42 | paper_id          |
| QASPER    | 512      | H1         | H1_minus_fixed_SENTENCE    |          0.196084 |        0.191808 |  0.004276 | -0.018333 |  0.025967 |       10000 |     42 | paper_id          |
| QASPER    | 512      | H2         | H2_minus_fixed_SENTENCE    |          0.183501 |        0.191808 | -0.008307 | -0.027316 |  0.009780 |       10000 |     42 | paper_id          |
| QASPER    | 512      | H3         | H3_minus_fixed_SENTENCE    |          0.169549 |        0.191808 | -0.022259 | -0.057226 |  0.012125 |       10000 |     42 | paper_id          |
| QASPER    | 512      | H5         | H5_minus_fixed_SENTENCE    |          0.131073 |        0.191808 | -0.060735 | -0.098636 | -0.022549 |       10000 |     42 | paper_id          |
| QASPER    | 512      | H1         | H1_minus_fixed_PROPOSITION |          0.196084 |        0.189278 |  0.006806 | -0.019359 |  0.035395 |       10000 |     42 | paper_id          |
| QASPER    | 512      | H2         | H2_minus_fixed_PROPOSITION |          0.183501 |        0.189278 | -0.005777 | -0.024731 |  0.011707 |       10000 |     42 | paper_id          |
| QASPER    | 512      | H3         | H3_minus_fixed_PROPOSITION |          0.169549 |        0.189278 | -0.019729 | -0.056142 |  0.016013 |       10000 |     42 | paper_id          |
| QASPER    | 512      | H5         | H5_minus_fixed_PROPOSITION |          0.131073 |        0.189278 | -0.058205 | -0.097529 | -0.019038 |       10000 |     42 | paper_id          |
| QASPER    | 1024     | H1         | H1_minus_fixed_SENTENCE    |          0.195769 |        0.204816 | -0.009048 | -0.040892 |  0.016318 |       10000 |     42 | paper_id          |
| QASPER    | 1024     | H2         | H2_minus_fixed_SENTENCE    |          0.193604 |        0.204816 | -0.011212 | -0.043624 |  0.014676 |       10000 |     42 | paper_id          |
| QASPER    | 1024     | H3         | H3_minus_fixed_SENTENCE    |          0.182107 |        0.204816 | -0.022710 | -0.062286 |  0.012803 |       10000 |     42 | paper_id          |
| QASPER    | 1024     | H5         | H5_minus_fixed_SENTENCE    |          0.188630 |        0.204816 | -0.016186 | -0.070638 |  0.041043 |       10000 |     42 | paper_id          |
| QASPER    | 1024     | H1         | H1_minus_fixed_PROPOSITION |          0.195769 |        0.200596 | -0.004827 | -0.032648 |  0.018579 |       10000 |     42 | paper_id          |
| QASPER    | 1024     | H2         | H2_minus_fixed_PROPOSITION |          0.193604 |        0.200596 | -0.006992 | -0.035275 |  0.016559 |       10000 |     42 | paper_id          |
| QASPER    | 1024     | H3         | H3_minus_fixed_PROPOSITION |          0.182107 |        0.200596 | -0.018489 | -0.054857 |  0.015176 |       10000 |     42 | paper_id          |
| QASPER    | 1024     | H5         | H5_minus_fixed_PROPOSITION |          0.188630 |        0.200596 | -0.011965 | -0.064286 |  0.044419 |       10000 |     42 | paper_id          |
| QASPER    | FULL     | H1         | H1_minus_fixed_SENTENCE    |          0.243726 |        0.235650 |  0.008076 | -0.000552 |  0.019910 |       10000 |     42 | paper_id          |
| QASPER    | FULL     | H2         | H2_minus_fixed_SENTENCE    |          0.245822 |        0.235650 |  0.010172 | -0.001013 |  0.026138 |       10000 |     42 | paper_id          |
| QASPER    | FULL     | H3         | H3_minus_fixed_SENTENCE    |          0.242887 |        0.235650 |  0.007237 |  0.000000 |  0.020165 |       10000 |     42 | paper_id          |
| QASPER    | FULL     | H5         | H5_minus_fixed_SENTENCE    |          0.242887 |        0.235650 |  0.007237 |  0.000000 |  0.020165 |       10000 |     42 | paper_id          |
| QASPER    | FULL     | H1         | H1_minus_fixed_PROPOSITION |          0.243726 |        0.242887 |  0.000839 | -0.015979 |  0.016880 |       10000 |     42 | paper_id          |
| QASPER    | FULL     | H2         | H2_minus_fixed_PROPOSITION |          0.245822 |        0.242887 |  0.002935 | -0.002242 |  0.011502 |       10000 |     42 | paper_id          |
| QASPER    | FULL     | H3         | H3_minus_fixed_PROPOSITION |          0.242887 |        0.242887 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | paper_id          |
| QASPER    | FULL     | H5         | H5_minus_fixed_PROPOSITION |          0.242887 |        0.242887 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | paper_id          |
| HotpotQA  | 256      | H1         | H1_minus_fixed_SENTENCE    |          0.380352 |        0.421265 | -0.040913 | -0.118827 |  0.031617 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 256      | H2         | H2_minus_fixed_SENTENCE    |          0.387303 |        0.421265 | -0.033962 | -0.117112 |  0.046562 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 256      | H3         | H3_minus_fixed_SENTENCE    |          0.437379 |        0.421265 |  0.016114 | -0.062096 |  0.093277 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 256      | H5         | H5_minus_fixed_SENTENCE    |          0.287253 |        0.421265 | -0.134012 | -0.229615 | -0.046088 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 256      | H1         | H1_minus_fixed_PROPOSITION |          0.380352 |        0.407943 | -0.027592 | -0.083943 |  0.028137 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 256      | H2         | H2_minus_fixed_PROPOSITION |          0.387303 |        0.407943 | -0.020641 | -0.077768 |  0.025680 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 256      | H3         | H3_minus_fixed_PROPOSITION |          0.437379 |        0.407943 |  0.029435 | -0.062386 |  0.120276 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 256      | H5         | H5_minus_fixed_PROPOSITION |          0.287253 |        0.407943 | -0.120690 | -0.227308 | -0.014290 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 512      | H1         | H1_minus_fixed_SENTENCE    |          0.414020 |        0.420340 | -0.006320 | -0.022961 |  0.008000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 512      | H2         | H2_minus_fixed_SENTENCE    |          0.404341 |        0.420340 | -0.015999 | -0.036429 |  0.002289 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 512      | H3         | H3_minus_fixed_SENTENCE    |          0.442910 |        0.420340 |  0.022570 | -0.020414 |  0.069683 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 512      | H5         | H5_minus_fixed_SENTENCE    |          0.346082 |        0.420340 | -0.074258 | -0.167811 |  0.013518 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 512      | H1         | H1_minus_fixed_PROPOSITION |          0.414020 |        0.430585 | -0.016565 | -0.055069 |  0.013281 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 512      | H2         | H2_minus_fixed_PROPOSITION |          0.404341 |        0.430585 | -0.026244 | -0.071286 |  0.007899 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 512      | H3         | H3_minus_fixed_PROPOSITION |          0.442910 |        0.430585 |  0.012325 | -0.023175 |  0.047809 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 512      | H5         | H5_minus_fixed_PROPOSITION |          0.346082 |        0.430585 | -0.084502 | -0.177860 | -0.000496 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 1024     | H1         | H1_minus_fixed_SENTENCE    |          0.479379 |        0.472093 |  0.007286 | -0.037000 |  0.050857 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 1024     | H2         | H2_minus_fixed_SENTENCE    |          0.475304 |        0.472093 |  0.003211 | -0.057933 |  0.066071 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 1024     | H3         | H3_minus_fixed_SENTENCE    |          0.487169 |        0.472093 |  0.015075 | -0.033399 |  0.065467 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 1024     | H5         | H5_minus_fixed_SENTENCE    |          0.391268 |        0.472093 | -0.080825 | -0.162019 | -0.007288 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 1024     | H1         | H1_minus_fixed_PROPOSITION |          0.479379 |        0.475548 |  0.003831 | -0.056435 |  0.060985 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 1024     | H2         | H2_minus_fixed_PROPOSITION |          0.475304 |        0.475548 | -0.000244 | -0.058292 |  0.056660 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 1024     | H3         | H3_minus_fixed_PROPOSITION |          0.487169 |        0.475548 |  0.011620 | -0.032104 |  0.056746 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | 1024     | H5         | H5_minus_fixed_PROPOSITION |          0.391268 |        0.475548 | -0.084280 | -0.172318 | -0.004277 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | FULL     | H1         | H1_minus_fixed_SENTENCE    |          0.403771 |        0.405590 | -0.001818 | -0.005455 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | FULL     | H2         | H2_minus_fixed_SENTENCE    |          0.419882 |        0.405590 |  0.014293 | -0.003636 |  0.042626 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | FULL     | H3         | H3_minus_fixed_SENTENCE    |          0.410590 |        0.405590 |  0.005000 |  0.000000 |  0.015000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | FULL     | H5         | H5_minus_fixed_SENTENCE    |          0.405590 |        0.405590 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | FULL     | H1         | H1_minus_fixed_PROPOSITION |          0.403771 |        0.418010 | -0.014239 | -0.041111 |  0.003896 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | FULL     | H2         | H2_minus_fixed_PROPOSITION |          0.419882 |        0.418010 |  0.001872 | -0.004567 |  0.009405 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | FULL     | H3         | H3_minus_fixed_PROPOSITION |          0.410590 |        0.418010 | -0.007421 | -0.030833 |  0.008571 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | FULL     | H5         | H5_minus_fixed_PROPOSITION |          0.405590 |        0.418010 | -0.012421 | -0.038254 |  0.005714 |       10000 |     42 | query_or_claim_id |
| SciFact   | 256      | H1         | H1_minus_fixed_SENTENCE    |          0.380000 |        0.680000 | -0.300000 | -0.480000 | -0.120000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 256      | H2         | H2_minus_fixed_SENTENCE    |          0.380000 |        0.680000 | -0.300000 | -0.480000 | -0.120000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 256      | H3         | H3_minus_fixed_SENTENCE    |          0.700000 |        0.680000 |  0.020000 | -0.100000 |  0.140000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 256      | H5         | H5_minus_fixed_SENTENCE    |          0.400000 |        0.680000 | -0.280000 | -0.460000 | -0.100000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 256      | H1         | H1_minus_fixed_PROPOSITION |          0.380000 |        0.660000 | -0.280000 | -0.460000 | -0.100000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 256      | H2         | H2_minus_fixed_PROPOSITION |          0.380000 |        0.660000 | -0.280000 | -0.460000 | -0.100000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 256      | H3         | H3_minus_fixed_PROPOSITION |          0.700000 |        0.660000 |  0.040000 | -0.060000 |  0.160000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 256      | H5         | H5_minus_fixed_PROPOSITION |          0.400000 |        0.660000 | -0.260000 | -0.440000 | -0.080000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 512      | H1         | H1_minus_fixed_SENTENCE    |          0.620000 |        0.680000 | -0.060000 | -0.180000 |  0.060000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 512      | H2         | H2_minus_fixed_SENTENCE    |          0.600000 |        0.680000 | -0.080000 | -0.200000 |  0.040000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 512      | H3         | H3_minus_fixed_SENTENCE    |          0.700000 |        0.680000 |  0.020000 | -0.100000 |  0.140000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 512      | H5         | H5_minus_fixed_SENTENCE    |          0.460000 |        0.680000 | -0.220000 | -0.380000 | -0.059500 |       10000 |     42 | query_or_claim_id |
| SciFact   | 512      | H1         | H1_minus_fixed_PROPOSITION |          0.620000 |        0.620000 |  0.000000 | -0.140000 |  0.140000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 512      | H2         | H2_minus_fixed_PROPOSITION |          0.600000 |        0.620000 | -0.020000 | -0.160000 |  0.120000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 512      | H3         | H3_minus_fixed_PROPOSITION |          0.700000 |        0.620000 |  0.080000 | -0.040000 |  0.200000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 512      | H5         | H5_minus_fixed_PROPOSITION |          0.460000 |        0.620000 | -0.160000 | -0.320000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 1024     | H1         | H1_minus_fixed_SENTENCE    |          0.620000 |        0.620000 |  0.000000 | -0.080000 |  0.080000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 1024     | H2         | H2_minus_fixed_SENTENCE    |          0.580000 |        0.620000 | -0.040000 | -0.100000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 1024     | H3         | H3_minus_fixed_SENTENCE    |          0.660000 |        0.620000 |  0.040000 | -0.060000 |  0.160000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 1024     | H5         | H5_minus_fixed_SENTENCE    |          0.460000 |        0.620000 | -0.160000 | -0.320000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 1024     | H1         | H1_minus_fixed_PROPOSITION |          0.620000 |        0.620000 |  0.000000 | -0.080000 |  0.080000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 1024     | H2         | H2_minus_fixed_PROPOSITION |          0.580000 |        0.620000 | -0.040000 | -0.100000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 1024     | H3         | H3_minus_fixed_PROPOSITION |          0.660000 |        0.620000 |  0.040000 | -0.060000 |  0.160000 |       10000 |     42 | query_or_claim_id |
| SciFact   | 1024     | H5         | H5_minus_fixed_PROPOSITION |          0.460000 |        0.620000 | -0.160000 | -0.320000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | FULL     | H1         | H1_minus_fixed_SENTENCE    |          0.660000 |        0.680000 | -0.020000 | -0.060000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | FULL     | H2         | H2_minus_fixed_SENTENCE    |          0.680000 |        0.680000 |  0.000000 | -0.060000 |  0.060000 |       10000 |     42 | query_or_claim_id |
| SciFact   | FULL     | H3         | H3_minus_fixed_SENTENCE    |          0.660000 |        0.680000 | -0.020000 | -0.100000 |  0.040000 |       10000 |     42 | query_or_claim_id |
| SciFact   | FULL     | H5         | H5_minus_fixed_SENTENCE    |          0.700000 |        0.680000 |  0.020000 | -0.040000 |  0.080000 |       10000 |     42 | query_or_claim_id |
| SciFact   | FULL     | H1         | H1_minus_fixed_PROPOSITION |          0.660000 |        0.680000 | -0.020000 | -0.080000 |  0.040000 |       10000 |     42 | query_or_claim_id |
| SciFact   | FULL     | H2         | H2_minus_fixed_PROPOSITION |          0.680000 |        0.680000 |  0.000000 | -0.060000 |  0.060000 |       10000 |     42 | query_or_claim_id |
| SciFact   | FULL     | H3         | H3_minus_fixed_PROPOSITION |          0.660000 |        0.680000 | -0.020000 | -0.060000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | FULL     | H5         | H5_minus_fixed_PROPOSITION |          0.700000 |        0.680000 |  0.020000 |  0.000000 |  0.060000 |       10000 |     42 | query_or_claim_id |

## H0 versus C semantic passage identity

| dataset   | budget   |   queries |   same_evidence |   same_prediction |
|:----------|:---------|----------:|----------------:|------------------:|
| QASPER    | 256      |        50 |              50 |                50 |
| QASPER    | 512      |        50 |              50 |                50 |
| QASPER    | 1024     |        50 |              50 |                50 |
| QASPER    | FULL     |        50 |              50 |                50 |
| HotpotQA  | 256      |        50 |              50 |                50 |
| HotpotQA  | 512      |        50 |              50 |                50 |
| HotpotQA  | 1024     |        50 |              50 |                50 |
| HotpotQA  | FULL     |        50 |              50 |                50 |
| SciFact   | 256      |        50 |              50 |                50 |
| SciFact   | 512      |        50 |              50 |                50 |
| SciFact   | 1024     |        50 |              50 |                50 |
| SciFact   | FULL     |        50 |              50 |                50 |

This identity diagnostic checks evidence and predictions separately. H0 follows retrieval ordering, whereas C explicitly ranks passages with the selector; the table shows whether they realize the same intervention. Exact duplicated outputs are not independent repeated trials.

## Retrospective oracle

| dataset   | reader      | best_fixed           |   best_fixed_score |   exact_oracle |   epsilon_oracle |   oracle_headroom |   epsilon |
|:----------|:------------|:---------------------|-------------------:|---------------:|-----------------:|------------------:|----------:|
| HotpotQA  | ALL_READERS | 0962fbdcf76305eedb48 |           0.487169 |       0.573730 |         0.573730 |          0.086562 |  0.010000 |
| HotpotQA  | Qwen        | 0962fbdcf76305eedb48 |           0.487169 |       0.573730 |         0.573730 |          0.086562 |  0.010000 |
| QASPER    | ALL_READERS | 217a31e1b7342ff3de03 |           0.245822 |       0.341862 |         0.341690 |          0.096040 |  0.010000 |
| QASPER    | Qwen        | 217a31e1b7342ff3de03 |           0.245822 |       0.341862 |         0.341690 |          0.096040 |  0.010000 |
| SciFact   | ALL_READERS | 0c5b120e284f0e5299dc |           0.700000 |       0.960000 |         0.960000 |          0.260000 |  0.000000 |
| SciFact   | Qwen        | 0c5b120e284f0e5299dc |           0.700000 |       0.960000 |         0.960000 |          0.260000 |  0.000000 |

These ceilings choose actions after seeing the true per-query outcomes. They do not demonstrate a deployable controller, causal evidence discovery, or a successful stopping policy. The oracle includes the available G action surface only; unqualified H4 is absent. Best-fixed scores and ceiling gaps are sensitive to the number of actions tried on this small pilot. H consolidates the broader saved action surface separately.

## Flags

| dataset   | strategy   |   empty |   cap |   invalid |
|:----------|:-----------|--------:|------:|----------:|
| HotpotQA  | FIXED      |       0 |     0 |         0 |
| HotpotQA  | H0         |       0 |     0 |         0 |
| HotpotQA  | H1         |       0 |     0 |         0 |
| HotpotQA  | H2         |       0 |     0 |         0 |
| HotpotQA  | H3         |       0 |     0 |         0 |
| HotpotQA  | H5         |       0 |     0 |         0 |
| QASPER    | FIXED      |       0 |     0 |         0 |
| QASPER    | H0         |      45 |     0 |         0 |
| QASPER    | H1         |      45 |     0 |         0 |
| QASPER    | H2         |      45 |     0 |         0 |
| QASPER    | H3         |       0 |     0 |         0 |
| QASPER    | H5         |       0 |     0 |         0 |
| SciFact   | FIXED      |       0 |     0 |         0 |
| SciFact   | H0         |      54 |     0 |         0 |
| SciFact   | H1         |      54 |     0 |         0 |
| SciFact   | H2         |      54 |     0 |         0 |
| SciFact   | H3         |       0 |     0 |         0 |
| SciFact   | H5         |       0 |     0 |         0 |

Total flags derived here: 297 empty-evidence rows, 0 generation-cap rows, 0 invalid SciFact labels. The frozen annotation-mapping flags persist for the same seven QASPER queries; see flagged_queries.csv for identifiers. No flagged row is dropped, reparsed or regenerated.

## Cost and scope

1863 physical generation calls served 3,150 logical rows; compatible cache reuse does not make conceptual inference cost zero. No extraction LM or learned controller is added. Embedding preprocessing and ranking have nonzero conceptual cost. Prompt tokens count actual LM input. End-to-end latency was not measured; cached generation/selection fields exclude some loading, preparation and IPC overhead and cannot establish serving speed superiority.

The hypothesis is whether deterministic heterogeneous source exposure improves the utility/token tradeoff under fixed reader/retriever controls. Mixed or zero-containing intervals do not establish equivalence. Any apparent winner still needs a larger preregistered held-out study. No automatic new policy or reader selection is made; H/I are saved-result analyses only.

## Complete available condition table

| dataset   | strategy   | budget   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |   prompt_tokens |   generated_tokens |
|:----------|:-----------|:---------|-------------:|-------------------:|---------:|---------:|-------------------------:|----------------:|-------------------:|
| HotpotQA  | FIXED      | 0        |     0.150250 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       52.800000 |          13.820000 |
| HotpotQA  | H0         | 1024     |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 |      851.160000 |          11.980000 |
| HotpotQA  | H0         | 256      |     0.337510 |           0.187260 | 0.400000 | 0.120000 |               165.900000 |      218.700000 |          12.440000 |
| HotpotQA  | H0         | 512      |     0.456280 |           0.306030 | 0.600000 | 0.040000 |               389.100000 |      441.920000 |          12.140000 |
| HotpotQA  | H0         | FULL     |     0.431145 |           0.280895 | 0.580000 | 0.060000 |              1351.060000 |     1403.860000 |          13.520000 |
| HotpotQA  | H1         | 1024     |     0.479379 |           0.329129 | 0.620000 | 0.040000 |               867.640000 |      920.440000 |          11.380000 |
| HotpotQA  | H1         | 256      |     0.380352 |           0.230102 | 0.440000 | 0.080000 |               205.460000 |      258.260000 |          12.400000 |
| HotpotQA  | H1         | 512      |     0.414020 |           0.263770 | 0.540000 | 0.060000 |               427.580000 |      480.380000 |          13.020000 |
| HotpotQA  | H1         | FULL     |     0.403771 |           0.253522 | 0.580000 | 0.060000 |              1351.020000 |     1403.820000 |          13.020000 |
| HotpotQA  | H2         | 1024     |     0.475304 |           0.325054 | 0.620000 | 0.040000 |               868.060000 |      920.860000 |          11.620000 |
| HotpotQA  | H2         | 256      |     0.387303 |           0.237053 | 0.480000 | 0.080000 |               205.260000 |      258.060000 |          12.240000 |
| HotpotQA  | H2         | 512      |     0.404341 |           0.254091 | 0.520000 | 0.060000 |               427.860000 |      480.660000 |          13.040000 |
| HotpotQA  | H2         | FULL     |     0.419882 |           0.269633 | 0.600000 | 0.060000 |              1352.580000 |     1405.380000 |          12.940000 |
| HotpotQA  | H3         | 1024     |     0.487169 |           0.336919 | 0.640000 | 0.040000 |               864.260000 |      917.060000 |          11.120000 |
| HotpotQA  | H3         | 256      |     0.437379 |           0.287129 | 0.540000 | 0.060000 |               201.720000 |      254.520000 |          11.720000 |
| HotpotQA  | H3         | 512      |     0.442910 |           0.292660 | 0.560000 | 0.040000 |               426.060000 |      478.880000 |          12.040000 |
| HotpotQA  | H3         | FULL     |     0.410590 |           0.260340 | 0.580000 | 0.060000 |              1352.260000 |     1405.060000 |          13.180000 |
| HotpotQA  | H5         | 1024     |     0.391268 |           0.241019 | 0.460000 | 0.080000 |               842.880000 |      895.720000 |          12.460000 |
| HotpotQA  | H5         | 256      |     0.287253 |           0.137003 | 0.340000 | 0.100000 |               193.240000 |      246.460000 |          14.340000 |
| HotpotQA  | H5         | 512      |     0.346082 |           0.195833 | 0.420000 | 0.120000 |               406.420000 |      459.320000 |          13.060000 |
| HotpotQA  | H5         | FULL     |     0.405590 |           0.255340 | 0.580000 | 0.060000 |              1350.500000 |     1403.300000 |          13.080000 |
| QASPER    | FIXED      | 0        |     0.091228 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       41.980000 |          18.160000 |
| QASPER    | H0         | 1024     |     0.199183 |           0.107956 | 0.600000 | 0.100000 |               773.900000 |      816.520000 |          20.380000 |
| QASPER    | H0         | 256      |     0.096925 |           0.005698 | 0.060000 | 0.020000 |                18.720000 |       60.740000 |          17.960000 |
| QASPER    | H0         | 512      |     0.158710 |           0.067482 | 0.460000 | 0.180000 |               267.100000 |      309.800000 |          19.320000 |
| QASPER    | H0         | FULL     |     0.243652 |           0.152425 | 0.740000 | 0.060000 |              4144.280000 |     4186.440000 |          21.200000 |
| QASPER    | H1         | 1024     |     0.195769 |           0.104541 | 0.580000 | 0.160000 |               884.000000 |      926.080000 |          20.500000 |
| QASPER    | H1         | 256      |     0.103456 |           0.012229 | 0.080000 | 0.020000 |                23.440000 |       65.460000 |          18.060000 |
| QASPER    | H1         | 512      |     0.196084 |           0.104857 | 0.580000 | 0.140000 |               435.920000 |      478.040000 |          20.180000 |
| QASPER    | H1         | FULL     |     0.243726 |           0.152498 | 0.720000 | 0.060000 |              4146.780000 |     4188.940000 |          21.060000 |
| QASPER    | H2         | 1024     |     0.193604 |           0.102377 | 0.580000 | 0.160000 |               881.620000 |      923.720000 |          20.180000 |
| QASPER    | H2         | 256      |     0.103456 |           0.012229 | 0.080000 | 0.020000 |                23.440000 |       65.460000 |          18.060000 |
| QASPER    | H2         | 512      |     0.183501 |           0.092273 | 0.540000 | 0.160000 |               438.600000 |      480.740000 |          20.020000 |
| QASPER    | H2         | FULL     |     0.245822 |           0.154594 | 0.720000 | 0.080000 |              4146.420000 |     4188.580000 |          21.460000 |
| QASPER    | H3         | 1024     |     0.182107 |           0.090879 | 0.580000 | 0.120000 |               864.500000 |      906.780000 |          19.540000 |
| QASPER    | H3         | 256      |     0.182305 |           0.091077 | 0.500000 | 0.120000 |               204.260000 |      246.320000 |          19.200000 |
| QASPER    | H3         | 512      |     0.169549 |           0.078322 | 0.520000 | 0.120000 |               426.340000 |      468.520000 |          19.540000 |
| QASPER    | H3         | FULL     |     0.242887 |           0.151659 | 0.720000 | 0.080000 |              4146.800000 |     4188.960000 |          21.460000 |
| QASPER    | H5         | 1024     |     0.188630 |           0.097403 | 0.560000 | 0.100000 |               822.040000 |      864.220000 |          18.060000 |
| QASPER    | H5         | 256      |     0.118661 |           0.027433 | 0.360000 | 0.180000 |               190.320000 |      232.700000 |          17.640000 |
| QASPER    | H5         | 512      |     0.131073 |           0.039845 | 0.440000 | 0.120000 |               397.340000 |      439.580000 |          18.120000 |
| QASPER    | H5         | FULL     |     0.242887 |           0.151659 | 0.720000 | 0.080000 |              4146.660000 |     4188.820000 |          21.460000 |
| SciFact   | FIXED      | 0        |     0.360000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       58.300000 |           3.720000 |
| SciFact   | H0         | 1024     |     0.680000 |           0.320000 | 0.440000 | 0.120000 |               667.620000 |      725.940000 |           3.680000 |
| SciFact   | H0         | 256      |     0.380000 |           0.020000 | 0.040000 | 0.020000 |                19.360000 |       77.660000 |           3.800000 |
| SciFact   | H0         | 512      |     0.660000 |           0.300000 | 0.400000 | 0.100000 |               237.700000 |      296.000000 |           3.840000 |
| SciFact   | H0         | FULL     |     0.660000 |           0.300000 | 0.420000 | 0.120000 |              4936.440000 |     4994.740000 |           3.840000 |
| SciFact   | H1         | 1024     |     0.620000 |           0.260000 | 0.380000 | 0.120000 |               819.280000 |      877.580000 |           3.400000 |
| SciFact   | H1         | 256      |     0.380000 |           0.020000 | 0.040000 | 0.020000 |                23.200000 |       81.500000 |           3.800000 |
| SciFact   | H1         | 512      |     0.620000 |           0.260000 | 0.360000 | 0.100000 |               316.480000 |      374.780000 |           3.640000 |
| SciFact   | H1         | FULL     |     0.660000 |           0.300000 | 0.440000 | 0.140000 |              4942.700000 |     5001.000000 |           3.720000 |
| SciFact   | H2         | 1024     |     0.580000 |           0.220000 | 0.380000 | 0.160000 |               820.860000 |      879.160000 |           3.400000 |
| SciFact   | H2         | 256      |     0.380000 |           0.020000 | 0.040000 | 0.020000 |                23.200000 |       81.500000 |           3.800000 |
| SciFact   | H2         | 512      |     0.600000 |           0.240000 | 0.340000 | 0.100000 |               316.340000 |      374.640000 |           3.600000 |
| SciFact   | H2         | FULL     |     0.680000 |           0.320000 | 0.460000 | 0.140000 |              4928.660000 |     4986.960000 |           3.720000 |
| SciFact   | H3         | 1024     |     0.660000 |           0.300000 | 0.420000 | 0.120000 |               814.300000 |      872.640000 |           3.640000 |
| SciFact   | H3         | 256      |     0.700000 |           0.340000 | 0.440000 | 0.100000 |               182.840000 |      241.200000 |           3.720000 |
| SciFact   | H3         | 512      |     0.700000 |           0.340000 | 0.480000 | 0.140000 |               400.940000 |      459.280000 |           3.760000 |
| SciFact   | H3         | FULL     |     0.660000 |           0.300000 | 0.460000 | 0.160000 |              4931.520000 |     4989.820000 |           3.680000 |
| SciFact   | H5         | 1024     |     0.460000 |           0.100000 | 0.280000 | 0.180000 |               779.080000 |      837.400000 |           3.920000 |
| SciFact   | H5         | 256      |     0.400000 |           0.040000 | 0.320000 | 0.280000 |               183.460000 |      241.840000 |           4.400000 |
| SciFact   | H5         | 512      |     0.460000 |           0.100000 | 0.320000 | 0.220000 |               377.380000 |      435.740000 |           4.200000 |
| SciFact   | H5         | FULL     |     0.700000 |           0.340000 | 0.480000 | 0.140000 |              4929.840000 |     4988.140000 |           3.760000 |
