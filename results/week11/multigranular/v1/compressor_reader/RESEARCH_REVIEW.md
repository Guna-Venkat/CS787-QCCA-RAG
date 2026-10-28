# Week 11 E — compressor × reader: full research review

**E is complete for the available surface:** 4,150 checksummed logical rows, 83 dataset/condition summaries, and the same 150 unique development queries as C/D. The primary scores and frozen status are unchanged. Optional C1/LLMLingua-2 conditions (1,800 configured rows) remain pending. C4/faithful SeleCom is future work, not an executed experiment.

## Experiments and controls

| Policy | Executed mechanism | Scope |
|---|---|---|
| C0 | Raw passage evidence, semantic ranking, strict budget prefix | All three readers/datasets; 256/512/1024/FULL |
| C1 | LLMLingua-2 token compression | Pending qualification; no results |
| C2 | Query-only SFR ranking of exact source sentences, retain at most eight, restore source order, enforce budget | All three readers/datasets; 256/512/1024/FULL |
| C3 | Existing adapted SARA Mistral checkpoint: top-four text passages plus remaining candidate vectors | QASPER/Mistral only, natural FULL point and own adapted no-evidence baseline |

C0/C2 share the exact SFR/P0 candidate pool (retrieval width 10, capped by the frozen shared-context policy), query and source hash. This was checked for all 1,800 paired C2/C0 query conditions. C2 is non-generative extraction and mechanically source-grounded. Gold labels never select source units. A numeric budget is a shared tokenizer maximum, not an exact realized-length match. The rank-prefix rule can leave C0 empty when its first passage is too large.

E uses pool width 10; D used width 20. Therefore D-to-E raw score differences cannot be attributed only to compression. Controlled compressor comparisons belong within E. C3 also changes checkpoint/interface/prompt construction, so its result is separate from the portable public-reader comparison.

## Main findings

1. **Sentence filtering is useful under tight budgets, but it is not a universal quality improvement.** C2 avoids many low-budget empty/coarse contexts and can expose a compact selection. With more budget, raw passages often retain higher QA point estimates. The contrast includes budget fit and unit exposure; it does not isolate a causal effect of removing irrelevant text alone.

2. **The top-eight cap saturates.** C2 FULL is still at most eight sentences, not the full candidate context. Identical outputs at different requested budgets can be expected cache reuse, not a stalled experiment. The exact identity table below checks this directly.

3. **The QASPER SARA specialist is a strong separate reference.** Its F1 is 0.403749 versus its own adapted no-evidence F1 0.190914. Its gain is not evidence that generic token compression or latent vectors alone caused the improvement: checkpoint adaptation, prompt/interface and exposed information differ from C0/C2.

4. **The SciFact formatting confound persists.** Mistral and Llama strict classification results are dominated by violations of the exact-label contract. Preserve the primary scores and examine the separately labelled compliance diagnostic; do not infer that zero strict accuracy means no claim reasoning ability.

## Best observed public-reader condition within each compressor

Each row is a retrospective peak chosen on the same 50-query pilot. QA uses F1; SciFact uses strict accuracy. These are descriptive peaks, not independently confirmed selections.

| dataset   | reader   | compressor   | budget   |   task_score |     HELP |     HURT |   evidence_tokens |   prompt_tokens |
|:----------|:---------|:-------------|:---------|-------------:|---------:|---------:|------------------:|----------------:|
| HotpotQA  | Llama    | C0           | FULL     |     0.429692 | 0.680000 | 0.060000 |       1293.580000 |     1369.020000 |
| HotpotQA  | Llama    | C2           | 256      |     0.388219 | 0.640000 | 0.080000 |        184.620000 |      260.060000 |
| HotpotQA  | Mistral  | C0           | 512      |     0.324281 | 0.580000 | 0.140000 |        439.180000 |      493.060000 |
| HotpotQA  | Mistral  | C2           | 512      |     0.255692 | 0.620000 | 0.180000 |        297.580000 |      351.160000 |
| HotpotQA  | Qwen     | C0           | 1024     |     0.480272 | 0.620000 | 0.060000 |        798.360000 |      851.160000 |
| HotpotQA  | Qwen     | C2           | 256      |     0.437931 | 0.560000 | 0.040000 |        194.920000 |      247.720000 |
| QASPER    | Llama    | C0           | FULL     |     0.262209 | 0.680000 | 0.100000 |       2351.320000 |     2416.720000 |
| QASPER    | Llama    | C2           | 256      |     0.184874 | 0.560000 | 0.160000 |        195.920000 |      260.920000 |
| QASPER    | Mistral  | C0           | 1024     |     0.192379 | 0.700000 | 0.100000 |        848.440000 |      890.280000 |
| QASPER    | Mistral  | C2           | 256      |     0.185692 | 0.620000 | 0.160000 |        228.600000 |      270.440000 |
| QASPER    | Qwen     | C0           | FULL     |     0.218722 | 0.660000 | 0.100000 |       2394.940000 |     2437.380000 |
| QASPER    | Qwen     | C2           | 256      |     0.206276 | 0.560000 | 0.180000 |        198.700000 |      240.740000 |
| SciFact   | Llama    | C0           | 256      |     0.000000 | 0.000000 | 0.000000 |         19.140000 |      100.360000 |
| SciFact   | Llama    | C2           | 256      |     0.000000 | 0.000000 | 0.000000 |        183.780000 |      265.000000 |
| SciFact   | Mistral  | C0           | 256      |     0.000000 | 0.000000 | 0.000000 |         23.440000 |       90.200000 |
| SciFact   | Mistral  | C2           | 256      |     0.000000 | 0.000000 | 0.000000 |        226.640000 |      293.440000 |
| SciFact   | Qwen     | C0           | 1024     |     0.680000 | 0.440000 | 0.120000 |        667.620000 |      725.940000 |
| SciFact   | Qwen     | C2           | 256      |     0.700000 | 0.440000 | 0.100000 |        186.960000 |      245.260000 |

## Matched-budget C2 minus C0 comparisons

All rows pair the same queries under the same reader and candidate pool. Paired bootstrap: 10,000 resamples, seed 42; QASPER paper clusters (40), other datasets query/claim clusters (50). Intervals are nominal and exploratory. Actual evidence tokens are shown because equal requested budgets do not guarantee equal lengths.

| dataset   | reader   | budget   | contrast    |   C2_score |   C0_score |   C2_evidence_tokens |   C0_evidence_tokens |     point |    ci_low |   ci_high |   resamples |   seed | cluster           |
|:----------|:---------|:---------|:------------|-----------:|-----------:|---------------------:|---------------------:|----------:|----------:|----------:|------------:|-------:|:------------------|
| QASPER    | Mistral  | 256      | C2_minus_C0 |   0.185692 |   0.064574 |           228.600000 |            19.660000 |  0.121118 |  0.073714 |  0.170077 |       10000 |     42 | paper_id          |
| QASPER    | Mistral  | 512      | C2_minus_C0 |   0.178653 |   0.140820 |           329.300000 |           289.480000 |  0.037834 | -0.008810 |  0.083814 |       10000 |     42 | paper_id          |
| QASPER    | Mistral  | 1024     | C2_minus_C0 |   0.179171 |   0.192379 |           334.740000 |           848.440000 | -0.013207 | -0.062038 |  0.035959 |       10000 |     42 | paper_id          |
| QASPER    | Mistral  | FULL     | C2_minus_C0 |   0.179171 |   0.187197 |           334.740000 |          2632.440000 | -0.008025 | -0.056856 |  0.040639 |       10000 |     42 | paper_id          |
| QASPER    | Qwen     | 256      | C2_minus_C0 |   0.206276 |   0.096925 |           198.700000 |            18.720000 |  0.109351 |  0.046333 |  0.171307 |       10000 |     42 | paper_id          |
| QASPER    | Qwen     | 512      | C2_minus_C0 |   0.195583 |   0.158710 |           287.260000 |           267.100000 |  0.036874 | -0.018645 |  0.099594 |       10000 |     42 | paper_id          |
| QASPER    | Qwen     | 1024     | C2_minus_C0 |   0.195984 |   0.199183 |           292.020000 |           773.900000 | -0.003200 | -0.048704 |  0.043553 |       10000 |     42 | paper_id          |
| QASPER    | Qwen     | FULL     | C2_minus_C0 |   0.195984 |   0.218722 |           292.020000 |          2394.940000 | -0.022738 | -0.062595 |  0.017499 |       10000 |     42 | paper_id          |
| QASPER    | Llama    | 256      | C2_minus_C0 |   0.184874 |   0.085587 |           195.920000 |            18.560000 |  0.099288 |  0.054627 |  0.146952 |       10000 |     42 | paper_id          |
| QASPER    | Llama    | 512      | C2_minus_C0 |   0.182285 |   0.131191 |           283.120000 |           262.660000 |  0.051094 |  0.002604 |  0.101398 |       10000 |     42 | paper_id          |
| QASPER    | Llama    | 1024     | C2_minus_C0 |   0.177285 |   0.198886 |           287.780000 |           759.700000 | -0.021601 | -0.077015 |  0.022410 |       10000 |     42 | paper_id          |
| QASPER    | Llama    | FULL     | C2_minus_C0 |   0.177285 |   0.262209 |           287.780000 |          2351.320000 | -0.084924 | -0.134988 | -0.038663 |       10000 |     42 | paper_id          |
| HotpotQA  | Mistral  | 256      | C2_minus_C0 |   0.254523 |   0.238869 |           222.480000 |           185.200000 |  0.015654 | -0.019745 |  0.051597 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Mistral  | 512      | C2_minus_C0 |   0.255692 |   0.324281 |           297.580000 |           439.180000 | -0.068589 | -0.151430 |  0.005883 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Mistral  | 1024     | C2_minus_C0 |   0.254890 |   0.309067 |           300.020000 |           903.240000 | -0.054176 | -0.130132 |  0.014283 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Mistral  | FULL     | C2_minus_C0 |   0.254890 |   0.320750 |           300.020000 |          1527.300000 | -0.065860 | -0.142493 |  0.003805 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Qwen     | 256      | C2_minus_C0 |   0.437931 |   0.337510 |           194.920000 |           165.900000 |  0.100422 | -0.005202 |  0.210173 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Qwen     | 512      | C2_minus_C0 |   0.425640 |   0.456280 |           260.040000 |           389.100000 | -0.030639 | -0.121164 |  0.061972 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Qwen     | 1024     | C2_minus_C0 |   0.420640 |   0.480272 |           262.260000 |           798.360000 | -0.059632 | -0.141493 |  0.020385 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Qwen     | FULL     | C2_minus_C0 |   0.420640 |   0.431145 |           262.260000 |          1351.060000 | -0.010505 | -0.075074 |  0.061023 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Llama    | 256      | C2_minus_C0 |   0.388219 |   0.284481 |           184.620000 |           157.560000 |  0.103738 |  0.012994 |  0.196675 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Llama    | 512      | C2_minus_C0 |   0.354328 |   0.386535 |           246.920000 |           370.380000 | -0.032207 | -0.108961 |  0.046769 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Llama    | 1024     | C2_minus_C0 |   0.354328 |   0.424796 |           248.920000 |           762.960000 | -0.070468 | -0.160559 |  0.020062 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Llama    | FULL     | C2_minus_C0 |   0.354328 |   0.429692 |           248.920000 |          1293.580000 | -0.075364 | -0.168968 |  0.017919 |       10000 |     42 | query_or_claim_id |
| SciFact   | Mistral  | 256      | C2_minus_C0 |   0.000000 |   0.000000 |           226.640000 |            23.440000 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Mistral  | 512      | C2_minus_C0 |   0.000000 |   0.000000 |           414.880000 |           285.200000 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Mistral  | 1024     | C2_minus_C0 |   0.000000 |   0.000000 |           431.820000 |           800.320000 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Mistral  | FULL     | C2_minus_C0 |   0.000000 |   0.000000 |           431.820000 |          3832.180000 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Qwen     | 256      | C2_minus_C0 |   0.700000 |   0.380000 |           186.960000 |            19.360000 |  0.320000 |  0.140000 |  0.500000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Qwen     | 512      | C2_minus_C0 |   0.680000 |   0.660000 |           342.240000 |           237.700000 |  0.020000 | -0.120000 |  0.160000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Qwen     | 1024     | C2_minus_C0 |   0.680000 |   0.680000 |           357.100000 |           667.620000 |  0.000000 | -0.120000 |  0.100000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Qwen     | FULL     | C2_minus_C0 |   0.680000 |   0.640000 |           357.100000 |          3199.000000 |  0.040000 | -0.080000 |  0.160000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Llama    | 256      | C2_minus_C0 |   0.000000 |   0.000000 |           183.780000 |            19.140000 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Llama    | 512      | C2_minus_C0 |   0.000000 |   0.000000 |           335.580000 |           234.680000 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Llama    | 1024     | C2_minus_C0 |   0.000000 |   0.000000 |           349.340000 |           653.920000 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Llama    | FULL     | C2_minus_C0 |   0.000000 |   0.000000 |           349.340000 |          3117.260000 |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |

The strongest primary-reader low-budget signals are QASPER C2/256 minus C0/256: +0.109351 F1, nominal CI [0.046333, 0.171307], and SciFact: +0.32 accuracy, CI [0.14, 0.50]. In contrast, HotpotQA +0.100422 F1 has CI [-0.005202, 0.210173] and is inconclusive. QASPER/SciFact C0 at that budget expose only about 19 mean evidence tokens, versus roughly 187–199 for C2, so budget-fit failures contribute to the contrast.

## Cross-budget cost diagnostic

These additional post-hoc contrasts compare C2 at 256 requested tokens against larger C0 budgets. Smaller prompt counts are a measured token-burden difference, not a serving-latency claim. Similar point estimates with broad CIs do not prove equivalence or non-inferiority. A CI containing zero does not establish no quality loss.

| dataset   | reader   |   C2_budget | C0_budget   |   C2_score |   C0_score |   C2_prompt_tokens |   C0_prompt_tokens |   prompt_reduction | analysis                         |     point |    ci_low |   ci_high |   resamples |   seed | cluster           |
|:----------|:---------|------------:|:------------|-----------:|-----------:|-------------------:|-------------------:|-------------------:|:---------------------------------|----------:|----------:|----------:|------------:|-------:|:------------------|
| QASPER    | Mistral  |         256 | 512         |   0.185692 |   0.140820 |         270.440000 |         331.260000 |           0.183602 | posthoc_cross_budget_descriptive |  0.044873 | -0.003533 |  0.087322 |       10000 |     42 | paper_id          |
| QASPER    | Mistral  |         256 | 1024        |   0.185692 |   0.192379 |         270.440000 |         890.280000 |           0.696230 | posthoc_cross_budget_descriptive | -0.006686 | -0.057728 |  0.040445 |       10000 |     42 | paper_id          |
| QASPER    | Mistral  |         256 | FULL        |   0.185692 |   0.187197 |         270.440000 |        2674.240000 |           0.898872 | posthoc_cross_budget_descriptive | -0.001505 | -0.047423 |  0.040537 |       10000 |     42 | paper_id          |
| QASPER    | Qwen     |         256 | 512         |   0.206276 |   0.158710 |         240.740000 |         309.800000 |           0.222918 | posthoc_cross_budget_descriptive |  0.047566 | -0.016358 |  0.115068 |       10000 |     42 | paper_id          |
| QASPER    | Qwen     |         256 | 1024        |   0.206276 |   0.199183 |         240.740000 |         816.520000 |           0.705163 | posthoc_cross_budget_descriptive |  0.007093 | -0.047208 |  0.059599 |       10000 |     42 | paper_id          |
| QASPER    | Qwen     |         256 | FULL        |   0.206276 |   0.218722 |         240.740000 |        2437.380000 |           0.901230 | posthoc_cross_budget_descriptive | -0.012446 | -0.063888 |  0.035431 |       10000 |     42 | paper_id          |
| QASPER    | Llama    |         256 | 512         |   0.184874 |   0.131191 |         260.920000 |         328.320000 |           0.205288 | posthoc_cross_budget_descriptive |  0.053683 | -0.004809 |  0.111218 |       10000 |     42 | paper_id          |
| QASPER    | Llama    |         256 | 1024        |   0.184874 |   0.198886 |         260.920000 |         825.280000 |           0.683841 | posthoc_cross_budget_descriptive | -0.014012 | -0.082248 |  0.049052 |       10000 |     42 | paper_id          |
| QASPER    | Llama    |         256 | FULL        |   0.184874 |   0.262209 |         260.920000 |        2416.720000 |           0.892035 | posthoc_cross_budget_descriptive | -0.077335 | -0.135422 | -0.027907 |       10000 |     42 | paper_id          |
| HotpotQA  | Mistral  |         256 | 512         |   0.254523 |   0.324281 |         276.100000 |         493.060000 |           0.440028 | posthoc_cross_budget_descriptive | -0.069758 | -0.152614 |  0.004211 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Mistral  |         256 | 1024        |   0.254523 |   0.309067 |         276.100000 |         957.060000 |           0.711512 | posthoc_cross_budget_descriptive | -0.054543 | -0.132537 |  0.016599 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Mistral  |         256 | FULL        |   0.254523 |   0.320750 |         276.100000 |        1581.120000 |           0.825377 | posthoc_cross_budget_descriptive | -0.066227 | -0.144344 |  0.003965 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Qwen     |         256 | 512         |   0.437931 |   0.456280 |         247.720000 |         441.920000 |           0.439446 | posthoc_cross_budget_descriptive | -0.018348 | -0.115887 |  0.083908 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Qwen     |         256 | 1024        |   0.437931 |   0.480272 |         247.720000 |         851.160000 |           0.708962 | posthoc_cross_budget_descriptive | -0.042341 | -0.125610 |  0.042461 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Qwen     |         256 | FULL        |   0.437931 |   0.431145 |         247.720000 |        1403.860000 |           0.823544 | posthoc_cross_budget_descriptive |  0.006786 | -0.072950 |  0.093535 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Llama    |         256 | 512         |   0.388219 |   0.386535 |         260.060000 |         445.820000 |           0.416670 | posthoc_cross_budget_descriptive |  0.001685 | -0.091109 |  0.094240 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Llama    |         256 | 1024        |   0.388219 |   0.424796 |         260.060000 |         838.400000 |           0.689814 | posthoc_cross_budget_descriptive | -0.036577 | -0.129904 |  0.054534 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Llama    |         256 | FULL        |   0.388219 |   0.429692 |         260.060000 |        1369.020000 |           0.810039 | posthoc_cross_budget_descriptive | -0.041473 | -0.133732 |  0.048518 |       10000 |     42 | query_or_claim_id |
| SciFact   | Mistral  |         256 | 512         |   0.000000 |   0.000000 |         293.440000 |         351.980000 |           0.166316 | posthoc_cross_budget_descriptive |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Mistral  |         256 | 1024        |   0.000000 |   0.000000 |         293.440000 |         867.260000 |           0.661647 | posthoc_cross_budget_descriptive |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Mistral  |         256 | FULL        |   0.000000 |   0.000000 |         293.440000 |        3899.100000 |           0.924742 | posthoc_cross_budget_descriptive |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Qwen     |         256 | 512         |   0.700000 |   0.660000 |         245.260000 |         296.000000 |           0.171419 | posthoc_cross_budget_descriptive |  0.040000 | -0.100000 |  0.180000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Qwen     |         256 | 1024        |   0.700000 |   0.680000 |         245.260000 |         725.940000 |           0.662148 | posthoc_cross_budget_descriptive |  0.020000 | -0.100000 |  0.140000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Qwen     |         256 | FULL        |   0.700000 |   0.640000 |         245.260000 |        3257.300000 |           0.924705 | posthoc_cross_budget_descriptive |  0.060000 | -0.080000 |  0.200000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Llama    |         256 | 512         |   0.000000 |   0.000000 |         265.000000 |         315.900000 |           0.161127 | posthoc_cross_budget_descriptive |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Llama    |         256 | 1024        |   0.000000 |   0.000000 |         265.000000 |         735.160000 |           0.639534 | posthoc_cross_budget_descriptive |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Llama    |         256 | FULL        |   0.000000 |   0.000000 |         265.000000 |        3198.480000 |           0.917148 | posthoc_cross_budget_descriptive |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |

Relative to Qwen C0/1024, C2/256 reduces mean final prompt tokens by 70.5% on QASPER, 70.9% on HotpotQA and 66.2% on SciFact. The corresponding quality-difference intervals allow losses of about 0.047 F1, 0.126 F1 and 0.10 accuracy respectively. These token savings therefore do not establish quality-preserving compression.

For Qwen, C2/256 reaches HotpotQA F1 0.437931 versus C0/1024 0.480272, QASPER 0.206276 versus 0.199183, and SciFact strict accuracy 0.70 versus 0.68. These mixed point-estimate tradeoffs support a compact-context hypothesis, not a universal compression win. The paired intervals above determine how much uncertainty remains.

## Top-eight saturation check: C2/1024 versus C2/FULL

| dataset   | reader   |   exact_evidence_matches |   identical_predictions |   queries |
|:----------|:---------|-------------------------:|------------------------:|----------:|
| QASPER    | Mistral  |                       50 |                      50 |        50 |
| QASPER    | Qwen     |                       50 |                      50 |        50 |
| QASPER    | Llama    |                       50 |                      50 |        50 |
| HotpotQA  | Mistral  |                       50 |                      50 |        50 |
| HotpotQA  | Qwen     |                       50 |                      50 |        50 |
| HotpotQA  | Llama    |                       50 |                      50 |        50 |
| SciFact   | Mistral  |                       50 |                      50 |        50 |
| SciFact   | Qwen     |                       50 |                      50 |        50 |
| SciFact   | Llama    |                       50 |                      50 |        50 |

A matching evidence hash means exact equality of the generator evidence text. Prediction equality is reported separately. This comparison never treats cached repeated outputs as independent replications.

## QASPER-only SARA specialist

| dataset   | reader   | reader_variant               | compressor   | evidence_granularity   | budget   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |   prompt_tokens |   generated_tokens |
|:----------|:---------|:-----------------------------|:-------------|:-----------------------|:---------|-------------:|-------------------:|---------:|---------:|-------------------------:|----------------:|-------------------:|
| QASPER    | Mistral  | MISTRAL_ONLY_LATENT_BASELINE | C3           | PASSAGE                | FULL     |     0.403749 |           0.212834 | 0.500000 | 0.080000 |              1090.340000 |     1215.140000 |          21.680000 |
| QASPER    | Mistral  | MISTRAL_ONLY_LATENT_BASELINE | C3           | NONE                   | 0        |     0.190914 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       70.660000 |          27.600000 |

|    point |   ci_low |   ci_high |   resamples |   seed | cluster   |
|---------:|---------:|----------:|------------:|-------:|:----------|
| 0.212834 | 0.118354 |  0.304464 |       10000 |     42 | paper_id  |

This interval compares C3 evidence intervention with the same adapted checkpoint no-evidence baseline. It is a natural policy contrast, not a matched public-reader compression ablation. SARA uses a distinct canonical prompt/latent interface, trained projector and restored added-token weights. Its text evidence tokens exclude non-text vector information; final prompt tokens include the actual interface tokens. Source-span support is not asserted for projected latent information. A future matched adapted full-text/latent ablation is needed to isolate the contribution of latents.

## Flags

| dataset   | reader   |   empty_evidence_rows |   generation_cap_rows |   invalid_label_rows |
|:----------|:---------|----------------------:|----------------------:|---------------------:|
| HotpotQA  | Llama    |                     0 |                     0 |                    0 |
| HotpotQA  | Mistral  |                     0 |                     0 |                    0 |
| HotpotQA  | Qwen     |                     0 |                     0 |                    0 |
| QASPER    | Llama    |                    45 |                     1 |                    0 |
| QASPER    | Mistral  |                    45 |                    14 |                    0 |
| QASPER    | Qwen     |                    45 |                     0 |                    0 |
| SciFact   | Llama    |                    54 |                   146 |                  450 |
| SciFact   | Mistral  |                    54 |                    53 |                  450 |
| SciFact   | Qwen     |                    54 |                     0 |                    0 |

There are 297 empty-evidence rows, 214 generation-cap rows, 900 invalid strict-label rows and 203 annotation-mapping rows. These are logical condition counts; the 203 mapping flags correspond to the same seven QASPER queries across 27 public conditions and two specialist conditions. Empty evidence follows the frozen allocation policy. Cap hits and invalid labels were preserved, not silently reparsed or regenerated.

Full row/action flags are in `flagged_queries.csv`. The separately saved `label_compliance_diagnostic_summary.csv` is post-hoc and does not replace the primary scoring, bootstrap or oracle tables.

## Cost and timing

850 physical generation calls served 4,150 logical rows: Mistral 350 (including the specialist), Qwen 250 and Llama 250. The other 3,300 rows reused compatible cached outputs. This does not make their conceptual inference cost zero. C2 needs no extraction LM; SFR embedding preprocessing/ranking still has a conceptual cost. Candidate text in memory is not counted as LM prompt input. Operational qualification calls are recorded separately from scientific rows.

Cached original retrieval/selection/generation timings do not measure full serving latency; model loading, IPC and some preparation are outside them. End-to-end latency remains unmeasured/null. No wall-clock efficiency superiority is claimed from these fields.

## Continuation

E remains frozen without changing any primary policy. F is the previously reviewed Qwen finalist panel under BM25/MiniLM/SFR; no post-hoc SARA promotion or reader swap is added. G retains Qwen/SFR to isolate deterministic hierarchy exposure. H/I use saved results only. The current 50-per-dataset development pilot needs a new, larger confirmation protocol before generalization claims.

## Complete 83-condition result table

| dataset   | reader   | reader_variant               | compressor   | evidence_granularity   | budget   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |   prompt_tokens |   generated_tokens |
|:----------|:---------|:-----------------------------|:-------------|:-----------------------|:---------|-------------:|-------------------:|---------:|---------:|-------------------------:|----------------:|-------------------:|
| HotpotQA  | Llama    | PUBLIC_TEXT_READER           | C0           | NONE                   | 0        |     0.094824 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       75.440000 |          18.780000 |
| HotpotQA  | Llama    | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 1024     |     0.424796 |           0.329972 | 0.700000 | 0.040000 |               762.960000 |      838.400000 |          13.920000 |
| HotpotQA  | Llama    | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 256      |     0.284481 |           0.189657 | 0.580000 | 0.100000 |               157.560000 |      233.000000 |          15.920000 |
| HotpotQA  | Llama    | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 512      |     0.386535 |           0.291711 | 0.660000 | 0.020000 |               370.380000 |      445.820000 |          13.860000 |
| HotpotQA  | Llama    | PUBLIC_TEXT_READER           | C0           | PASSAGE                | FULL     |     0.429692 |           0.334868 | 0.680000 | 0.060000 |              1293.580000 |     1369.020000 |          13.880000 |
| HotpotQA  | Llama    | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 1024     |     0.354328 |           0.259504 | 0.680000 | 0.060000 |               248.920000 |      324.360000 |          15.680000 |
| HotpotQA  | Llama    | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 256      |     0.388219 |           0.293395 | 0.640000 | 0.080000 |               184.620000 |      260.060000 |          14.940000 |
| HotpotQA  | Llama    | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 512      |     0.354328 |           0.259504 | 0.680000 | 0.060000 |               246.920000 |      322.360000 |          15.680000 |
| HotpotQA  | Llama    | PUBLIC_TEXT_READER           | C2           | SENTENCE               | FULL     |     0.354328 |           0.259504 | 0.680000 | 0.060000 |               248.920000 |      324.360000 |          15.680000 |
| HotpotQA  | Mistral  | PUBLIC_TEXT_READER           | C0           | NONE                   | 0        |     0.100453 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       53.360000 |          28.420000 |
| HotpotQA  | Mistral  | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 1024     |     0.309067 |           0.208614 | 0.560000 | 0.200000 |               903.240000 |      957.060000 |          24.780000 |
| HotpotQA  | Mistral  | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 256      |     0.238869 |           0.138417 | 0.440000 | 0.240000 |               185.200000 |      239.020000 |          25.280000 |
| HotpotQA  | Mistral  | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 512      |     0.324281 |           0.223828 | 0.580000 | 0.140000 |               439.180000 |      493.060000 |          23.000000 |
| HotpotQA  | Mistral  | PUBLIC_TEXT_READER           | C0           | PASSAGE                | FULL     |     0.320750 |           0.220297 | 0.560000 | 0.220000 |              1527.300000 |     1581.120000 |          26.980000 |
| HotpotQA  | Mistral  | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 1024     |     0.254890 |           0.154437 | 0.620000 | 0.180000 |               300.020000 |      353.600000 |          30.160000 |
| HotpotQA  | Mistral  | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 256      |     0.254523 |           0.154071 | 0.600000 | 0.200000 |               222.480000 |      276.100000 |          29.660000 |
| HotpotQA  | Mistral  | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 512      |     0.255692 |           0.155239 | 0.620000 | 0.180000 |               297.580000 |      351.160000 |          30.120000 |
| HotpotQA  | Mistral  | PUBLIC_TEXT_READER           | C2           | SENTENCE               | FULL     |     0.254890 |           0.154437 | 0.620000 | 0.180000 |               300.020000 |      353.600000 |          30.160000 |
| HotpotQA  | Qwen     | PUBLIC_TEXT_READER           | C0           | NONE                   | 0        |     0.150250 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       52.800000 |          13.820000 |
| HotpotQA  | Qwen     | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 1024     |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 |      851.160000 |          11.980000 |
| HotpotQA  | Qwen     | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 256      |     0.337510 |           0.187260 | 0.400000 | 0.120000 |               165.900000 |      218.700000 |          12.440000 |
| HotpotQA  | Qwen     | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 512      |     0.456280 |           0.306030 | 0.600000 | 0.040000 |               389.100000 |      441.920000 |          12.140000 |
| HotpotQA  | Qwen     | PUBLIC_TEXT_READER           | C0           | PASSAGE                | FULL     |     0.431145 |           0.280895 | 0.580000 | 0.060000 |              1351.060000 |     1403.860000 |          13.520000 |
| HotpotQA  | Qwen     | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 1024     |     0.420640 |           0.270391 | 0.540000 | 0.060000 |               262.260000 |      315.060000 |          11.800000 |
| HotpotQA  | Qwen     | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 256      |     0.437931 |           0.287682 | 0.560000 | 0.040000 |               194.920000 |      247.720000 |          11.880000 |
| HotpotQA  | Qwen     | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 512      |     0.425640 |           0.275391 | 0.540000 | 0.060000 |               260.040000 |      312.840000 |          11.800000 |
| HotpotQA  | Qwen     | PUBLIC_TEXT_READER           | C2           | SENTENCE               | FULL     |     0.420640 |           0.270391 | 0.540000 | 0.060000 |               262.260000 |      315.060000 |          11.800000 |
| QASPER    | Llama    | PUBLIC_TEXT_READER           | C0           | NONE                   | 0        |     0.071760 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       64.940000 |          17.720000 |
| QASPER    | Llama    | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 1024     |     0.198886 |           0.127126 | 0.580000 | 0.180000 |               759.700000 |      825.280000 |          30.800000 |
| QASPER    | Llama    | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 256      |     0.085587 |           0.013827 | 0.040000 | 0.020000 |                18.560000 |       83.540000 |          17.900000 |
| QASPER    | Llama    | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 512      |     0.131191 |           0.059432 | 0.440000 | 0.200000 |               262.660000 |      328.320000 |          21.800000 |
| QASPER    | Llama    | PUBLIC_TEXT_READER           | C0           | PASSAGE                | FULL     |     0.262209 |           0.190449 | 0.680000 | 0.100000 |              2351.320000 |     2416.720000 |          34.280000 |
| QASPER    | Llama    | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 1024     |     0.177285 |           0.105525 | 0.540000 | 0.160000 |               287.780000 |      352.800000 |          22.420000 |
| QASPER    | Llama    | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 256      |     0.184874 |           0.113115 | 0.560000 | 0.160000 |               195.920000 |      260.920000 |          20.220000 |
| QASPER    | Llama    | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 512      |     0.182285 |           0.110525 | 0.560000 | 0.160000 |               283.120000 |      348.140000 |          22.300000 |
| QASPER    | Llama    | PUBLIC_TEXT_READER           | C2           | SENTENCE               | FULL     |     0.177285 |           0.105525 | 0.540000 | 0.160000 |               287.780000 |      352.800000 |          22.420000 |
| QASPER    | Mistral  | PUBLIC_TEXT_READER           | C0           | NONE                   | 0        |     0.059805 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       41.660000 |          22.040000 |
| QASPER    | Mistral  | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 1024     |     0.192379 |           0.132573 | 0.700000 | 0.100000 |               848.440000 |      890.280000 |          41.020000 |
| QASPER    | Mistral  | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 256      |     0.064574 |           0.004769 | 0.060000 | 0.020000 |                19.660000 |       61.320000 |          22.500000 |
| QASPER    | Mistral  | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 512      |     0.140820 |           0.081014 | 0.540000 | 0.180000 |               289.480000 |      331.260000 |          33.440000 |
| QASPER    | Mistral  | PUBLIC_TEXT_READER           | C0           | PASSAGE                | FULL     |     0.187197 |           0.127392 | 0.580000 | 0.160000 |              2632.440000 |     2674.240000 |          60.300000 |
| QASPER    | Mistral  | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 1024     |     0.179171 |           0.119366 | 0.600000 | 0.180000 |               334.740000 |      376.580000 |          34.120000 |
| QASPER    | Mistral  | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 256      |     0.185692 |           0.125887 | 0.620000 | 0.160000 |               228.600000 |      270.440000 |          29.120000 |
| QASPER    | Mistral  | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 512      |     0.178653 |           0.118848 | 0.600000 | 0.180000 |               329.300000 |      371.140000 |          33.240000 |
| QASPER    | Mistral  | PUBLIC_TEXT_READER           | C2           | SENTENCE               | FULL     |     0.179171 |           0.119366 | 0.600000 | 0.180000 |               334.740000 |      376.580000 |          34.120000 |
| QASPER    | Mistral  | MISTRAL_ONLY_LATENT_BASELINE | C3           | NONE                   | 0        |     0.190914 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       70.660000 |          27.600000 |
| QASPER    | Mistral  | MISTRAL_ONLY_LATENT_BASELINE | C3           | PASSAGE                | FULL     |     0.403749 |           0.212834 | 0.500000 | 0.080000 |              1090.340000 |     1215.140000 |          21.680000 |
| QASPER    | Qwen     | PUBLIC_TEXT_READER           | C0           | NONE                   | 0        |     0.091228 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       41.980000 |          18.160000 |
| QASPER    | Qwen     | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 1024     |     0.199183 |           0.107956 | 0.600000 | 0.100000 |               773.900000 |      816.520000 |          20.380000 |
| QASPER    | Qwen     | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 256      |     0.096925 |           0.005698 | 0.060000 | 0.020000 |                18.720000 |       60.740000 |          17.960000 |
| QASPER    | Qwen     | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 512      |     0.158710 |           0.067482 | 0.460000 | 0.180000 |               267.100000 |      309.800000 |          19.320000 |
| QASPER    | Qwen     | PUBLIC_TEXT_READER           | C0           | PASSAGE                | FULL     |     0.218722 |           0.127494 | 0.660000 | 0.100000 |              2394.940000 |     2437.380000 |          21.500000 |
| QASPER    | Qwen     | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 1024     |     0.195984 |           0.104756 | 0.520000 | 0.180000 |               292.020000 |      334.080000 |          19.440000 |
| QASPER    | Qwen     | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 256      |     0.206276 |           0.115048 | 0.560000 | 0.180000 |               198.700000 |      240.740000 |          20.180000 |
| QASPER    | Qwen     | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 512      |     0.195583 |           0.104356 | 0.520000 | 0.180000 |               287.260000 |      329.320000 |          19.560000 |
| QASPER    | Qwen     | PUBLIC_TEXT_READER           | C2           | SENTENCE               | FULL     |     0.195984 |           0.104756 | 0.520000 | 0.180000 |               292.020000 |      334.080000 |          19.440000 |
| SciFact   | Llama    | PUBLIC_TEXT_READER           | C0           | NONE                   | 0        |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       81.220000 |         110.440000 |
| SciFact   | Llama    | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               653.920000 |      735.160000 |          99.740000 |
| SciFact   | Llama    | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                19.140000 |      100.360000 |         106.500000 |
| SciFact   | Llama    | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               234.680000 |      315.900000 |          86.740000 |
| SciFact   | Llama    | PUBLIC_TEXT_READER           | C0           | PASSAGE                | FULL     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |              3117.260000 |     3198.480000 |         112.760000 |
| SciFact   | Llama    | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               349.340000 |      430.600000 |          97.660000 |
| SciFact   | Llama    | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               183.780000 |      265.000000 |          88.640000 |
| SciFact   | Llama    | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               335.580000 |      416.840000 |          97.160000 |
| SciFact   | Llama    | PUBLIC_TEXT_READER           | C2           | SENTENCE               | FULL     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               349.340000 |      430.600000 |          97.660000 |
| SciFact   | Mistral  | PUBLIC_TEXT_READER           | C0           | NONE                   | 0        |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       66.780000 |          79.860000 |
| SciFact   | Mistral  | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               800.320000 |      867.260000 |          70.680000 |
| SciFact   | Mistral  | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                23.440000 |       90.200000 |          78.320000 |
| SciFact   | Mistral  | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               285.200000 |      351.980000 |          70.500000 |
| SciFact   | Mistral  | PUBLIC_TEXT_READER           | C0           | PASSAGE                | FULL     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |              3832.180000 |     3899.100000 |          88.240000 |
| SciFact   | Mistral  | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               431.820000 |      498.780000 |          74.020000 |
| SciFact   | Mistral  | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               226.640000 |      293.440000 |          69.500000 |
| SciFact   | Mistral  | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               414.880000 |      481.840000 |          74.920000 |
| SciFact   | Mistral  | PUBLIC_TEXT_READER           | C2           | SENTENCE               | FULL     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               431.820000 |      498.780000 |          74.020000 |
| SciFact   | Qwen     | PUBLIC_TEXT_READER           | C0           | NONE                   | 0        |     0.360000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       58.300000 |           3.720000 |
| SciFact   | Qwen     | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 1024     |     0.680000 |           0.320000 | 0.440000 | 0.120000 |               667.620000 |      725.940000 |           3.680000 |
| SciFact   | Qwen     | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 256      |     0.380000 |           0.020000 | 0.040000 | 0.020000 |                19.360000 |       77.660000 |           3.800000 |
| SciFact   | Qwen     | PUBLIC_TEXT_READER           | C0           | PASSAGE                | 512      |     0.660000 |           0.300000 | 0.400000 | 0.100000 |               237.700000 |      296.000000 |           3.840000 |
| SciFact   | Qwen     | PUBLIC_TEXT_READER           | C0           | PASSAGE                | FULL     |     0.640000 |           0.280000 | 0.420000 | 0.140000 |              3199.000000 |     3257.300000 |           3.640000 |
| SciFact   | Qwen     | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 1024     |     0.680000 |           0.320000 | 0.400000 | 0.080000 |               357.100000 |      415.440000 |           3.560000 |
| SciFact   | Qwen     | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 256      |     0.700000 |           0.340000 | 0.440000 | 0.100000 |               186.960000 |      245.260000 |           3.720000 |
| SciFact   | Qwen     | PUBLIC_TEXT_READER           | C2           | SENTENCE               | 512      |     0.680000 |           0.320000 | 0.400000 | 0.080000 |               342.240000 |      400.580000 |           3.560000 |
| SciFact   | Qwen     | PUBLIC_TEXT_READER           | C2           | SENTENCE               | FULL     |     0.680000 |           0.320000 | 0.400000 | 0.080000 |               357.100000 |      415.440000 |           3.560000 |
