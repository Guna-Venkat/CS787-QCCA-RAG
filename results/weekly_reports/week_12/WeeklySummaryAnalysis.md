# Week 12 — Summary Analysis

## Objective

Validate query-conditioned budgeted granularity fusion under frozen TRAIN/VAL splits, test its mechanisms, freeze a final fallback, then separate post-hoc attribution from reserved confirmation.

## Experiments performed

| Part   | Question                                                                                                     | Status       | Main_result                                                                                                                                                                                                                                                                                                                                                            |
|:-------|:-------------------------------------------------------------------------------------------------------------|:-------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0      | Can TRAIN, reused VAL and reserved confirmation roles be frozen without QASPER TEST reuse?                   | COMPLETED    | VAL counts are QASPER1005, HotpotQA7155 and SciFact177. Frozen confirmation is HotpotQA200, SciFact50, QASPER0. Historical QASPER TEST stays closed.                                                                                                                                                                                                                   |
| A      | Can query-conditioned budgeted mixed-granularity optimization be implemented with exact-source evidence?     | SMOKE-TESTED | Algorithm implementation supports exact-source sentences/passages, query-conditioned weights/mass and a hard additive token budget. Quality evidence belongs in C/D, not implementation tests.                                                                                                                                                                         |
| B      | Which controls are qualified before validation and what can qualification establish?                         | SMOKE-TESTED | The qualified optional LLMLingua2 checkpoint is microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank @5f0c82792b7ea14c6484e015b6a072009496b7f2, not the XLM-R-large checkpoint. Qualification does not establish quality.                                                                                                                                    |
| C      | Does QBGF improve over strong matched fixed evidence policies?                                               | COMPLETED    | QBGF lacks a robust universal VAL advantage. At256 QASPER QBGF0.235515 trails FIXED_SENTENCE0.263430; HotpotQA0.412068 slightly exceeds0.408247; SciFact0.674969 trails0.700062. At1024 passages are competitive or stronger in multiple tasks.                                                                                                                        |
| C-EXT  | How do BM25 and SFR passage controls compare on the same frozen VAL cohorts/budgets?                         | COMPLETED    | The extension has33348 completed logical rows. At1024 HotpotQA SFR RAW_SKIP0.452133 exceeds BM250.435345; SciFact BM250.710716 exceeds SFR0.698979. No universal SFR claim follows.                                                                                                                                                                                    |
| D      | Does query conditioning matter, and is mixed granularity necessary beyond sentence/passages-only allocation? | COMPLETED    | QBGF_FULL exceeds NO_QUERY in all six VAL cells by point estimate; e.g. HotpotQA2560.412068 vs0.308593. Sentence-only nearly duplicates FULL at256, so query conditioning evidence does not establish necessity of fusion.                                                                                                                                             |
| E      | Does the external token compressor improve matched quality/cost?                                             | SKIPPED      | Optional-phase decisions explicitly mark E NOT_RUN_BY_USER_DECISION. A qualified checkpoint exists, but no full LLMLingua quality outcome is established.                                                                                                                                                                                                              |
| F      | Does trained evidence-utilization adaptation improve the reader?                                             | SKIPPED      | Full F training was skipped by explicit recorded decision. Scheduler/coverage implementation and tiny training qualification are not completed final LoRA training.                                                                                                                                                                                                    |
| G      | Does a trained adapter improve the same frozen evidence?                                                     | SKIPPED      | Full G LoRA evaluation was not run; missing adapter results remain unestablished rather than assigned zero.                                                                                                                                                                                                                                                            |
| H      | Is QBGF sufficiently robust to promote as a universal final policy?                                          | COMPLETED    | All51 aggregate validation conditions reproduce from141729 unique C/D rows after16674 matching duplicates are removed. The Part-I freeze retains FIXED_SENTENCE without LoRA; QBGF universal promotion was not established.                                                                                                                                            |
| I      | Were method, reader, budgets and confirmation membership frozen before outcomes?                             | COMPLETED    | Both byte-identical Part-I freezes have SHA d0188a3582d6aead0a3a31f9f15bad9b51a88a6b475b05ac79ae38e58f05366b. FIXED_SENTENCE, no LoRA, five methods and fixed confirmation IDs were frozen.                                                                                                                                                                            |
| J      | Can post-hoc input ablation use a frozen VAL subset without changing the final system?                       | COMPLETED    | A separate150-query/600-cell VAL attribution freeze exists. It compares QBGF and a VAL-selected comparator at both budgets, not fresh confirmation.                                                                                                                                                                                                                    |
| K      | Which selected units change task score when removed or exposed alone?                                        | COMPLETED    | All600 cells completed, with16032 perturbation rows,9607 LOO units and13410 recorded physical calls. Across units687 are helpful,8311 neutral and609 harmful under the signed task-score criterion.                                                                                                                                                                    |
| L      | Does relevance predict utility, and do oracle-ranked subsets reveal redundancy?                              | COMPLETED    | Ten of12 relevance/importance CIs include zero. Corrected finite retention has only6–34 eligible queries per curve; original curve_summary propagates NaN for undefined ratios. Original outputs remain preserved.                                                                                                                                                     |
| M      | Do the frozen policy and validation conclusions generalize to the reserved cohort?                           | COMPLETED    | All2500 rows complete,2213 distinct cached generations. FIXED_SENTENCE exceeds NONE in all four cells after primary-family correction. SciFact256 sentence-minus-RAW_SKIP is+0.28762 [0.09709,0.47206]; at1024 passage point estimates are higher but paired intervals cross zero. QBGF wins HotpotQA1024 by0.03438 with secondary Holm p0.03160; no universal winner. |

## Key results

### Recorded validation counts

| dataset   |   n_queries |
|:----------|------------:|
| HotpotQA  |        7155 |
| QASPER    |        1005 |
| SciFact   |         177 |

Source: [results/week12/qbgf/v3/analysis/summary.csv](../../week12/qbgf/v3/analysis/summary.csv).

### Frozen confirmation membership counts

| dataset   |   confirmation_queries |
|:----------|-----------------------:|
| HotpotQA  |                    200 |
| QASPER    |                      0 |
| SciFact   |                     50 |

Source: [results/week12/W13_FINAL_METHOD_FREEZE.json](../../week12/W13_FINAL_METHOD_FREEZE.json).

### Qualification numeric fields; not benchmark scores

| field                                             |       value |
|:--------------------------------------------------|------------:|
| frozen_science_config_unchanged                   |    1.000000 |
| reason                                            |             |
| synthetic_budget_checks[0].budget                 |  256.000000 |
| synthetic_budget_checks[0].compressed_qwen_tokens |  250.000000 |
| synthetic_budget_checks[0].raw_qwen_tokens        | 2041.000000 |
| synthetic_budget_checks[0].retry_factor           |    0.800000 |
| synthetic_budget_checks[0].seconds                |    6.121564 |
| synthetic_budget_checks[1].budget                 | 1024.000000 |
| synthetic_budget_checks[1].compressed_qwen_tokens |  875.000000 |
| synthetic_budget_checks[1].raw_qwen_tokens        | 2041.000000 |
| synthetic_budget_checks[1].retry_factor           |    0.800000 |
| synthetic_budget_checks[1].seconds                |    6.137273 |

Source: [results/week12/qbgf/v3/protocol/llmlingua_qualification_lock.json](../../week12/qbgf/v3/protocol/llmlingua_qualification_lock.json).

### Main matched VAL outcomes

| dataset   | method         |   budget | reader    |   n_queries |   quality | metric   |   prompt_tokens |   evidence_tokens |     HELP |     HURT |   invalid_outputs |
|:----------|:---------------|---------:|:----------|------------:|----------:|:---------|----------------:|------------------:|---------:|---------:|------------------:|
| HotpotQA  | C2             |      256 | BASE_FP16 |        7155 |  0.411927 | f1       |      274.616492 |        220.700489 | 0.505521 | 0.094060 |                 0 |
| HotpotQA  | FIXED_SENTENCE |     1024 | BASE_FP16 |        7155 |  0.423310 | f1       |     1055.249057 |       1001.333054 | 0.590217 | 0.089727 |                 0 |
| HotpotQA  | FIXED_SENTENCE |      256 | BASE_FP16 |        7155 |  0.408247 | f1       |      308.606569 |        254.690566 | 0.516841 | 0.094200 |                 0 |
| HotpotQA  | H3             |     1024 | BASE_FP16 |        7155 |  0.436421 | f1       |     1041.308875 |        987.392872 | 0.588959 | 0.089727 |                 0 |
| HotpotQA  | H3             |      256 | BASE_FP16 |        7155 |  0.425311 | f1       |      291.398602 |        237.482600 | 0.525227 | 0.092662 |                 0 |
| HotpotQA  | NONE           |     1024 | BASE_FP16 |        7155 |  0.165324 | f1       |       53.916003 |          0.000000 | 0.000000 | 0.000000 |                 0 |
| HotpotQA  | NONE           |      256 | BASE_FP16 |        7155 |  0.165324 | f1       |       53.916003 |          0.000000 | 0.000000 | 0.000000 |                 0 |
| HotpotQA  | QBGF_FULL      |     1024 | BASE_FP16 |        7155 |  0.432344 | f1       |     1047.078127 |        993.162124 | 0.594549 | 0.087072 |                 0 |
| HotpotQA  | QBGF_FULL      |      256 | BASE_FP16 |        7155 |  0.412068 | f1       |      304.758211 |        250.842208 | 0.518658 | 0.096017 |                 0 |
| HotpotQA  | RAW_SKIP       |     1024 | BASE_FP16 |        7155 |  0.452133 | f1       |     1005.667365 |        951.751363 | 0.592453 | 0.082600 |                 0 |
| HotpotQA  | RAW_SKIP       |      256 | BASE_FP16 |        7155 |  0.406974 | f1       |      278.060936 |        224.144934 | 0.496436 | 0.099511 |                 0 |
| QASPER    | C2             |      256 | BASE_FP16 |        1005 |  0.258118 | f1       |      270.380100 |        227.702488 | 0.544279 | 0.103483 |                 0 |

Source: [results/week12/qbgf/v3/analysis/summary.csv](../../week12/qbgf/v3/analysis/summary.csv). First 12 of 33 rows shown; detailed report and source preserve the full comparison.

### Matched paired contrasts; joint family where recorded

| dataset   |   budget | left      | right          | metric   | status   |     point |    ci_low |   ci_high |   joint_holm_adjusted_p |
|:----------|---------:|:----------|:---------------|:---------|:---------|----------:|----------:|----------:|------------------------:|
| QASPER    |      256 | QBGF_FULL | RAW_SKIP       | f1       | COMPLETE |  0.066233 |  0.048862 |  0.083949 |                0.005099 |
| QASPER    |      256 | QBGF_FULL | FIXED_SENTENCE | f1       | COMPLETE | -0.027915 | -0.040996 | -0.014908 |                0.005099 |
| QASPER    |      256 | QBGF_FULL | H3             | f1       | COMPLETE | -0.018847 | -0.032123 | -0.005551 |                0.163184 |
| QASPER    |     1024 | QBGF_FULL | RAW_SKIP       | f1       | COMPLETE |  0.007098 | -0.003304 |  0.017170 |                1.000000 |
| QASPER    |     1024 | QBGF_FULL | FIXED_SENTENCE | f1       | COMPLETE | -0.003734 | -0.012490 |  0.005008 |                1.000000 |
| QASPER    |     1024 | QBGF_FULL | H3             | f1       | COMPLETE | -0.004432 | -0.012973 |  0.003865 |                1.000000 |
| HotpotQA  |      256 | QBGF_FULL | RAW_SKIP       | f1       | COMPLETE |  0.005094 | -0.002327 |  0.012384 |                1.000000 |
| HotpotQA  |      256 | QBGF_FULL | FIXED_SENTENCE | f1       | COMPLETE |  0.003821 | -0.003511 |  0.011013 |                1.000000 |
| HotpotQA  |      256 | QBGF_FULL | H3             | f1       | COMPLETE | -0.013243 | -0.020821 | -0.005900 |                0.034997 |
| HotpotQA  |     1024 | QBGF_FULL | RAW_SKIP       | f1       | COMPLETE | -0.019789 | -0.024994 | -0.014683 |                0.005099 |
| HotpotQA  |     1024 | QBGF_FULL | FIXED_SENTENCE | f1       | COMPLETE |  0.009033 |  0.004544 |  0.013509 |                0.022198 |
| HotpotQA  |     1024 | QBGF_FULL | H3             | f1       | COMPLETE | -0.004077 | -0.008862 |  0.000724 |                1.000000 |

Source: [results/week12/qbgf/v3/analysis/primary_comparisons.csv](../../week12/qbgf/v3/analysis/primary_comparisons.csv). First 12 of 18 rows shown; detailed report and source preserve the full comparison.

### Matched Qwen passage-RAG controls

| dataset   | method         |   budget | reader    |   n_queries |   quality |   prompt_tokens |   evidence_tokens | scope               |
|:----------|:---------------|---------:|:----------|------------:|----------:|----------------:|------------------:|:--------------------|
| HotpotQA  | BM25_PLAIN_RAG |     1024 | BASE_FP16 |        7155 |  0.435345 |     1013.819706 |        959.903704 | matched Qwen/budget |
| HotpotQA  | BM25_PLAIN_RAG |      256 | BASE_FP16 |        7155 |  0.379553 |      279.466806 |        225.550804 | matched Qwen/budget |
| HotpotQA  | SFR_PLAIN_RAG  |     1024 | BASE_FP16 |        7155 |  0.452133 |     1005.667365 |        951.751363 | matched Qwen/budget |
| HotpotQA  | SFR_PLAIN_RAG  |      256 | BASE_FP16 |        7155 |  0.406974 |      278.060936 |        224.144934 | matched Qwen/budget |
| QASPER    | BM25_PLAIN_RAG |     1024 | BASE_FP16 |        1005 |  0.255928 |      950.866667 |        908.189055 | matched Qwen/budget |
| QASPER    | BM25_PLAIN_RAG |      256 | BASE_FP16 |        1005 |  0.153388 |      280.466667 |        237.789055 | matched Qwen/budget |
| QASPER    | SFR_PLAIN_RAG  |     1024 | BASE_FP16 |        1005 |  0.287957 |      955.235821 |        912.558209 | matched Qwen/budget |
| QASPER    | SFR_PLAIN_RAG  |      256 | BASE_FP16 |        1005 |  0.169283 |      267.397015 |        224.719403 | matched Qwen/budget |
| SciFact   | BM25_PLAIN_RAG |     1024 | BASE_FP16 |         177 |  0.710716 |     1003.779661 |        938.841808 | matched Qwen/budget |
| SciFact   | BM25_PLAIN_RAG |      256 | BASE_FP16 |         177 |  0.437511 |      248.971751 |        184.033898 | matched Qwen/budget |
| SciFact   | SFR_PLAIN_RAG  |     1024 | BASE_FP16 |         177 |  0.698979 |      998.937853 |        934.000000 | matched Qwen/budget |
| SciFact   | SFR_PLAIN_RAG  |      256 | BASE_FP16 |         177 |  0.456378 |      251.073446 |        186.135593 | matched Qwen/budget |

Source: [results/week12/qbgf/v3/c_extension/v1/analysis/merged_validation_summary.csv](../../week12/qbgf/v3/c_extension/v1/analysis/merged_validation_summary.csv).

### Full CORE D ablations

| dataset   | method             |   budget | reader    |   n_queries |   quality | metric   |   prompt_tokens |   evidence_tokens |     HELP |     HURT |   invalid_outputs |
|:----------|:-------------------|---------:|:----------|------------:|----------:|:---------|----------------:|------------------:|---------:|---------:|------------------:|
| HotpotQA  | QBGF_FULL          |     1024 | BASE_FP16 |        7155 |  0.432344 | f1       |     1047.078127 |        993.162124 | 0.594549 | 0.087072 |                 0 |
| HotpotQA  | QBGF_FULL          |      256 | BASE_FP16 |        7155 |  0.412068 | f1       |      304.758211 |        250.842208 | 0.518658 | 0.096017 |                 0 |
| HotpotQA  | QBGF_NO_QUERY      |     1024 | BASE_FP16 |        7155 |  0.412252 | f1       |     1044.255625 |        990.339623 | 0.574563 | 0.096995 |                 0 |
| HotpotQA  | QBGF_NO_QUERY      |      256 | BASE_FP16 |        7155 |  0.308593 | f1       |      304.902166 |        250.986164 | 0.394969 | 0.137386 |                 0 |
| HotpotQA  | QBGF_PASSAGE_ONLY  |     1024 | BASE_FP16 |        7155 |  0.452507 | f1       |     1010.822642 |        956.906639 | 0.594829 | 0.083019 |                 0 |
| HotpotQA  | QBGF_PASSAGE_ONLY  |      256 | BASE_FP16 |        7155 |  0.393509 | f1       |      282.667226 |        228.751223 | 0.483578 | 0.106499 |                 0 |
| HotpotQA  | QBGF_SENTENCE_ONLY |     1024 | BASE_FP16 |        7155 |  0.432295 | f1       |     1046.459119 |        992.543117 | 0.594270 | 0.087072 |                 0 |
| HotpotQA  | QBGF_SENTENCE_ONLY |      256 | BASE_FP16 |        7155 |  0.412189 | f1       |      304.756674 |        250.840671 | 0.518658 | 0.096017 |                 0 |
| QASPER    | QBGF_FULL          |     1024 | BASE_FP16 |        1005 |  0.295055 | f1       |     1063.146269 |       1020.468657 | 0.615920 | 0.070647 |                 0 |
| QASPER    | QBGF_FULL          |      256 | BASE_FP16 |        1005 |  0.235515 | f1       |      295.353234 |        252.675622 | 0.519403 | 0.107463 |                 0 |
| QASPER    | QBGF_NO_QUERY      |     1024 | BASE_FP16 |        1005 |  0.228373 | f1       |     1062.479602 |       1019.801990 | 0.525373 | 0.113433 |                 0 |
| QASPER    | QBGF_NO_QUERY      |      256 | BASE_FP16 |        1005 |  0.147303 | f1       |      295.238806 |        252.561194 | 0.364179 | 0.164179 |                 0 |

Source: [results/week12/qbgf/v3/analysis/summary.csv](../../week12/qbgf/v3/analysis/summary.csv). First 12 of 24 rows shown; detailed report and source preserve the full comparison.

### Mixed versus single-granularity diagnostics

| dataset   |   budget | left      | right              |     point |    ci_low |   ci_high |   joint_holm_adjusted_p |
|:----------|---------:|:----------|:-------------------|----------:|----------:|----------:|------------------------:|
| QASPER    |      256 | QBGF_FULL | QBGF_SENTENCE_ONLY | -0.000069 | -0.000581 |  0.000367 |                1.000000 |
| QASPER    |      256 | QBGF_FULL | QBGF_PASSAGE_ONLY  |  0.066233 |  0.048862 |  0.083949 |                0.005099 |
| QASPER    |     1024 | QBGF_FULL | QBGF_SENTENCE_ONLY | -0.002685 | -0.010267 |  0.004915 |                1.000000 |
| QASPER    |     1024 | QBGF_FULL | QBGF_PASSAGE_ONLY  |  0.005474 | -0.005031 |  0.015607 |                1.000000 |
| HotpotQA  |      256 | QBGF_FULL | QBGF_SENTENCE_ONLY | -0.000121 | -0.000363 |  0.000000 |                1.000000 |
| HotpotQA  |      256 | QBGF_FULL | QBGF_PASSAGE_ONLY  |  0.018559 |  0.011475 |  0.025506 |                0.005099 |
| HotpotQA  |     1024 | QBGF_FULL | QBGF_SENTENCE_ONLY |  0.000049 | -0.001157 |  0.001270 |                1.000000 |
| HotpotQA  |     1024 | QBGF_FULL | QBGF_PASSAGE_ONLY  | -0.020163 | -0.025042 | -0.015425 |                0.005099 |
| SciFact   |      256 | QBGF_FULL | QBGF_SENTENCE_ONLY |  0.000000 |  0.000000 |  0.000000 |                1.000000 |
| SciFact   |      256 | QBGF_FULL | QBGF_PASSAGE_ONLY  |  0.217194 |  0.139681 |  0.298944 |                0.005099 |
| SciFact   |     1024 | QBGF_FULL | QBGF_SENTENCE_ONLY | -0.024280 | -0.051337 | -0.004812 |                1.000000 |
| SciFact   |     1024 | QBGF_FULL | QBGF_PASSAGE_ONLY  |  0.057305 | -0.009679 |  0.127139 |                1.000000 |

Source: [results/week12/qbgf/v3/analysis/fusion_diagnostic_comparisons.csv](../../week12/qbgf/v3/analysis/fusion_diagnostic_comparisons.csv).

### Offline raw-result recomputation audit

| dataset   | method         |   budget | reader    |    n |   recomputed_quality |   stored_quality |   difference |   recomputed_prompt_tokens |
|:----------|:---------------|---------:|:----------|-----:|---------------------:|-----------------:|-------------:|---------------------------:|
| HotpotQA  | NONE           |      256 | BASE_FP16 | 7155 |             0.165324 |         0.165324 |    -0.000000 |                  53.916003 |
| HotpotQA  | C2             |      256 | BASE_FP16 | 7155 |             0.411927 |         0.411927 |    -0.000000 |                 274.616492 |
| SciFact   | RAW_SKIP       |      256 | BASE_FP16 |  177 |             0.456378 |         0.456378 |     0.000000 |                 251.073446 |
| HotpotQA  | NONE           |     1024 | BASE_FP16 | 7155 |             0.165324 |         0.165324 |    -0.000000 |                  53.916003 |
| HotpotQA  | RAW_SKIP       |     1024 | BASE_FP16 | 7155 |             0.452133 |         0.452133 |    -0.000000 |                1005.667365 |
| HotpotQA  | H3             |     1024 | BASE_FP16 | 7155 |             0.436421 |         0.436421 |    -0.000000 |                1041.308875 |
| QASPER    | NONE           |     1024 | BASE_FP16 | 1005 |             0.082157 |         0.082157 |     0.000000 |                  42.677612 |
| HotpotQA  | H3             |      256 | BASE_FP16 | 7155 |             0.425311 |         0.425311 |    -0.000000 |                 291.398602 |
| QASPER    | RAW_SKIP       |     1024 | BASE_FP16 | 1005 |             0.287957 |         0.287957 |     0.000000 |                 955.235821 |
| HotpotQA  | QBGF_FULL      |     1024 | BASE_FP16 | 7155 |             0.432344 |         0.432344 |    -0.000000 |                1047.078127 |
| HotpotQA  | QBGF_FULL      |      256 | BASE_FP16 | 7155 |             0.412068 |         0.412068 |    -0.000000 |                 304.758211 |
| QASPER    | FIXED_SENTENCE |      256 | BASE_FP16 | 1005 |             0.263430 |         0.263430 |     0.000000 |                 297.893532 |

Source: [results/weekly_reports/week_12/OfflineValidationRecomputation.csv](OfflineValidationRecomputation.csv). First 12 of 51 rows shown; detailed report and source preserve the full comparison.

### Exact frozen confirmation registry

| method         | reader    | budgets   |
|:---------------|:----------|:----------|
| NONE           | BASE_FP16 | 256/1024  |
| RAW_SKIP       | BASE_FP16 | 256/1024  |
| FIXED_SENTENCE | BASE_FP16 | 256/1024  |
| H3             | BASE_FP16 | 256/1024  |
| QBGF_FULL      | BASE_FP16 | 256/1024  |

Source: [results/week12/W13_FINAL_METHOD_FREEZE.json](../../week12/W13_FINAL_METHOD_FREEZE.json).

### Attribution planned cost metadata

| field                                     |        value |
|:------------------------------------------|-------------:|
| causal_ONLY_top3_unknown_until_LOO        |     1.000000 |
| cells                                     |   600.000000 |
| estimated_additional_disk_GiB_upper       |     0.268601 |
| known_LOO_and_relevance_ONLY_unique       | 10505.000000 |
| known_action_cache_reuse_percent          |     0.514041 |
| known_actions_already_cached              |    54.000000 |
| minimum_free_disk_GiB                     |     2.000000 |
| no_evidence_reference_cache_reuse_percent |   100.000000 |
| optional_replacement                      |     0.000000 |
| optional_third_method                     |     0.000000 |
| original_reference_answers                |   600.000000 |
| original_reference_cache_reuse_percent    |   100.000000 |

Source: [results/week12/attribution/v1/cost_plan.json](../../week12/attribution/v1/cost_plan.json). First 12 of 16 rows shown; detailed report and source preserve the full comparison.

### unit summary

| dataset   |   budget | method            |   n_queries |   n_units |   mean_importance |   median_importance |   helpful |   neutral |   harmful |   mean_query_harmful |     top1 |   top1_tie_aware |     top3 |   top_relevance_negative |   top_relevance_zero |   within_query_rho |   within_query_valid |   exact_answer_flip |   correctness_flip |
|:----------|---------:|:------------------|------------:|----------:|------------------:|--------------------:|----------:|----------:|----------:|---------------------:|---------:|-----------------:|---------:|-------------------------:|---------------------:|-------------------:|---------------------:|--------------------:|-------------------:|
| HotpotQA  |      256 | H3                |          50 |       451 |          0.026659 |            0.000000 |  0.097561 |  0.853659 |  0.048780 |             0.050845 | 0.120000 |         0.600000 | 0.373333 |                 0.080000 |             0.820000 |          -0.015780 |                   32 |            0.268293 |           0.079823 |
| HotpotQA  |      256 | QBGF_FULL         |          50 |       587 |          0.007786 |            0.000000 |  0.088586 |  0.839864 |  0.071550 |             0.070172 | 0.160000 |         0.640000 | 0.280000 |                 0.100000 |             0.660000 |           0.091010 |                   31 |            0.282794 |           0.069847 |
| HotpotQA  |     1024 | QBGF_FULL         |          50 |      1914 |          0.004924 |            0.000000 |  0.050157 |  0.880878 |  0.068966 |             0.068345 | 0.060000 |         0.560000 | 0.120000 |                 0.120000 |             0.720000 |           0.105920 |                   37 |            0.192268 |           0.025601 |
| HotpotQA  |     1024 | QBGF_PASSAGE_ONLY |          50 |       395 |          0.050549 |            0.000000 |  0.131646 |  0.820253 |  0.048101 |             0.047143 | 0.240000 |         0.700000 | 0.453333 |                 0.060000 |             0.560000 |           0.265397 |                   35 |            0.301266 |           0.068354 |
| QASPER    |      256 | FIXED_SENTENCE    |          50 |       467 |          0.014623 |            0.000000 |  0.156317 |  0.764454 |  0.079229 |             0.081342 | 0.220000 |         0.600000 | 0.346667 |                 0.140000 |             0.660000 |           0.154813 |                   30 |            0.404711 |           0.002141 |
| QASPER    |      256 | QBGF_FULL         |          50 |       511 |          0.017406 |            0.000000 |  0.170254 |  0.694716 |  0.135029 |             0.124977 | 0.300000 |         0.600000 | 0.446667 |                 0.280000 |             0.480000 |           0.144183 |                   32 |            0.481409 |           0.003914 |
| QASPER    |     1024 | H3                |          50 |      1824 |          0.000568 |            0.000000 |  0.069627 |  0.870066 |  0.060307 |             0.060853 | 0.160000 |         0.560000 | 0.233333 |                 0.100000 |             0.720000 |           0.069541 |                   32 |            0.240680 |           0.000548 |
| QASPER    |     1024 | QBGF_FULL         |          50 |      1225 |          0.002085 |            0.000000 |  0.079184 |  0.817959 |  0.102857 |             0.090947 | 0.160000 |         0.560000 | 0.253333 |                 0.140000 |             0.660000 |           0.079713 |                   32 |            0.331429 |           0.004082 |
| SciFact   |      256 | C2                |          50 |       267 |          0.007491 |            0.000000 |  0.071161 |  0.865169 |  0.063670 |             0.062333 | 0.180000 |         0.780000 | 0.593333 |                 0.080000 |             0.800000 |           0.024034 |                   22 |            0.138577 |           0.134831 |
| SciFact   |      256 | QBGF_FULL         |          50 |       425 |         -0.021176 |            0.000000 |  0.028235 |  0.922353 |  0.049412 |             0.044968 | 0.200000 |         0.880000 | 0.393333 |                 0.040000 |             0.860000 |           0.131812 |                   16 |            0.082353 |           0.077647 |
| SciFact   |     1024 | BM25_PLAIN_RAG    |          50 |       156 |          0.096154 |            0.000000 |  0.134615 |  0.826923 |  0.038462 |             0.038333 | 0.440000 |         0.840000 | 0.900000 |                 0.080000 |             0.620000 |           0.281316 |                   23 |            0.192308 |           0.173077 |
| SciFact   |     1024 | QBGF_FULL         |          50 |      1385 |         -0.000722 |            0.000000 |  0.005054 |  0.989170 |  0.005776 |             0.005307 | 0.160000 |         0.920000 | 0.160000 |                 0.040000 |             0.880000 |           0.090958 |                   11 |            0.021661 |           0.010830 |

Source: [results/week12/attribution/v1/analysis/unit_summary.csv](../../week12/attribution/v1/analysis/unit_summary.csv).

### system quality

| dataset   |   budget | method            |   n_queries |   quality |   accuracy |
|:----------|---------:|:------------------|------------:|----------:|-----------:|
| HotpotQA  |      256 | H3                |          50 |  0.387697 | nan        |
| HotpotQA  |      256 | QBGF_FULL         |          50 |  0.404225 | nan        |
| HotpotQA  |     1024 | QBGF_FULL         |          50 |  0.402047 | nan        |
| HotpotQA  |     1024 | QBGF_PASSAGE_ONLY |          50 |  0.451415 | nan        |
| QASPER    |      256 | FIXED_SENTENCE    |          50 |  0.266794 | nan        |
| QASPER    |      256 | QBGF_FULL         |          50 |  0.264119 | nan        |
| QASPER    |     1024 | H3                |          50 |  0.281196 | nan        |
| QASPER    |     1024 | QBGF_FULL         |          50 |  0.296416 | nan        |
| SciFact   |      256 | C2                |          50 |  0.601190 |   0.620000 |
| SciFact   |      256 | QBGF_FULL         |          50 |  0.530994 |   0.540000 |
| SciFact   |     1024 | BM25_PLAIN_RAG    |          50 |  0.712121 |   0.740000 |
| SciFact   |     1024 | QBGF_FULL         |          50 |  0.588105 |   0.640000 |

Source: [results/week12/attribution/v1/analysis/system_quality.csv](../../week12/attribution/v1/analysis/system_quality.csv).

### relevance utility

| dataset   |   budget | method            |       rho |    ci_low |   ci_high |   pearson_secondary |   valid_bootstrap |   n_units |   n_clusters |   reason |
|:----------|---------:|:------------------|----------:|----------:|----------:|--------------------:|------------------:|----------:|-------------:|---------:|
| HotpotQA  |      256 | H3                |  0.023016 | -0.064932 |  0.109274 |            0.015805 |             10000 |       451 |           50 |      nan |
| HotpotQA  |      256 | QBGF_FULL         |  0.068114 | -0.015664 |  0.148245 |            0.025706 |             10000 |       587 |           50 |      nan |
| HotpotQA  |     1024 | QBGF_FULL         |  0.010000 | -0.089176 |  0.119896 |            0.035762 |             10000 |      1914 |           50 |      nan |
| HotpotQA  |     1024 | QBGF_PASSAGE_ONLY |  0.221220 |  0.138216 |  0.303377 |            0.205209 |             10000 |       395 |           50 |      nan |
| QASPER    |      256 | FIXED_SENTENCE    |  0.126160 | -0.021608 |  0.266764 |            0.050025 |             10000 |       467 |           50 |      nan |
| QASPER    |      256 | QBGF_FULL         |  0.104150 | -0.054446 |  0.246300 |            0.077742 |             10000 |       511 |           50 |      nan |
| QASPER    |     1024 | H3                | -0.027767 | -0.133377 |  0.081273 |            0.075036 |             10000 |      1824 |           50 |      nan |
| QASPER    |     1024 | QBGF_FULL         | -0.034423 | -0.125624 |  0.057194 |            0.034624 |             10000 |      1225 |           50 |      nan |
| SciFact   |      256 | C2                |  0.038219 | -0.129371 |  0.204550 |            0.054169 |             10000 |       267 |           50 |      nan |
| SciFact   |      256 | QBGF_FULL         |  0.134372 |  0.009403 |  0.244331 |            0.148799 |             10000 |       425 |           50 |      nan |
| SciFact   |     1024 | BM25_PLAIN_RAG    |  0.168433 | -0.013400 |  0.338062 |            0.263932 |             10000 |       156 |           50 |      nan |
| SciFact   |     1024 | QBGF_FULL         |  0.044106 | -0.028456 |  0.112701 |            0.063750 |             10000 |      1385 |           50 |      nan |

Source: [results/week12/attribution/v1/analysis/relevance_utility.csv](../../week12/attribution/v1/analysis/relevance_utility.csv).

### paired harmful effects

| dataset   |   budget | contrast                                                 |      mean |    ci_low |   ci_high |   n |
|:----------|---------:|:---------------------------------------------------------|----------:|----------:|----------:|----:|
| HotpotQA  |      256 | QBGF - comparator harmful unit fraction (query weighted) |  0.019327 | -0.017290 |  0.059550 |  50 |
| HotpotQA  |     1024 | QBGF - comparator harmful unit fraction (query weighted) |  0.021202 | -0.025015 |  0.076183 |  50 |
| QASPER    |      256 | QBGF - comparator harmful unit fraction (query weighted) |  0.043635 | -0.018040 |  0.113949 |  50 |
| QASPER    |     1024 | QBGF - comparator harmful unit fraction (query weighted) |  0.030094 | -0.032109 |  0.093112 |  50 |
| SciFact   |      256 | QBGF - comparator harmful unit fraction (query weighted) | -0.017365 | -0.076314 |  0.040501 |  50 |
| SciFact   |     1024 | QBGF - comparator harmful unit fraction (query weighted) | -0.033027 | -0.069873 | -0.003404 |  50 |

Source: [results/week12/attribution/v1/analysis/paired_harmful_effects.csv](../../week12/attribution/v1/analysis/paired_harmful_effects.csv).

## Main scientific conclusions

- VAL counts are QASPER1005, HotpotQA7155 and SciFact177. Frozen confirmation is HotpotQA200, SciFact50, QASPER0. Historical QASPER TEST stays closed. Source: [results/week12/W13_FINAL_METHOD_FREEZE.json](../../week12/W13_FINAL_METHOD_FREEZE.json), [results/week12/qbgf/v3/protocol/training_execution_patch.json](../../week12/qbgf/v3/protocol/training_execution_patch.json)
- Algorithm implementation supports exact-source sentences/passages, query-conditioned weights/mass and a hard additive token budget. Quality evidence belongs in C/D, not implementation tests. Source: [notebooks/week12/week12_A_qbgf_algorithm.ipynb](../../../notebooks/week12/week12_A_qbgf_algorithm.ipynb), [src/week12/algorithm.py](../../../src/week12/algorithm.py)
- The qualified optional LLMLingua2 checkpoint is microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank @5f0c82792b7ea14c6484e015b6a072009496b7f2, not the XLM-R-large checkpoint. Qualification does not establish quality. Source: [results/week12/qbgf/v3/qualification/baselines.json](../../week12/qbgf/v3/qualification/baselines.json), [results/week12/qbgf/v3/protocol/llmlingua_qualification_lock.json](../../week12/qbgf/v3/protocol/llmlingua_qualification_lock.json)
- QBGF lacks a robust universal VAL advantage. At256 QASPER QBGF0.235515 trails FIXED_SENTENCE0.263430; HotpotQA0.412068 slightly exceeds0.408247; SciFact0.674969 trails0.700062. At1024 passages are competitive or stronger in multiple tasks. Source: [results/week12/qbgf/v3/analysis/summary.csv](../../week12/qbgf/v3/analysis/summary.csv), [results/week12/qbgf/v3/validation/C/status.json](../../week12/qbgf/v3/validation/C/status.json)
- The extension has33348 completed logical rows. At1024 HotpotQA SFR RAW_SKIP0.452133 exceeds BM250.435345; SciFact BM250.710716 exceeds SFR0.698979. No universal SFR claim follows. Source: [results/week12/qbgf/v3/c_extension/v1/analysis/merged_validation_summary.csv](../../week12/qbgf/v3/c_extension/v1/analysis/merged_validation_summary.csv), [results/week12/qbgf/v3/c_extension/v1/status.json](../../week12/qbgf/v3/c_extension/v1/status.json)
- QBGF_FULL exceeds NO_QUERY in all six VAL cells by point estimate; e.g. HotpotQA2560.412068 vs0.308593. Sentence-only nearly duplicates FULL at256, so query conditioning evidence does not establish necessity of fusion. Source: [results/week12/qbgf/v3/analysis/summary.csv](../../week12/qbgf/v3/analysis/summary.csv), [results/week12/qbgf/v3/analysis/fusion_diagnostic_comparisons.csv](../../week12/qbgf/v3/analysis/fusion_diagnostic_comparisons.csv)
- Optional-phase decisions explicitly mark E NOT_RUN_BY_USER_DECISION. A qualified checkpoint exists, but no full LLMLingua quality outcome is established. Source: [results/week12/qbgf/v3/protocol/optional_phase_decisions.json](../../week12/qbgf/v3/protocol/optional_phase_decisions.json), [results/week12/qbgf/v3/protocol/llmlingua_qualification_lock.json](../../week12/qbgf/v3/protocol/llmlingua_qualification_lock.json)
- Full F training was skipped by explicit recorded decision. Scheduler/coverage implementation and tiny training qualification are not completed final LoRA training. Source: [results/week12/qbgf/v3/protocol/optional_phase_decisions.json](../../week12/qbgf/v3/protocol/optional_phase_decisions.json), [results/week12/qbgf/v3/protocol/training_execution_patch.json](../../week12/qbgf/v3/protocol/training_execution_patch.json)
- Full G LoRA evaluation was not run; missing adapter results remain unestablished rather than assigned zero. Source: [results/week12/qbgf/v3/protocol/optional_phase_decisions.json](../../week12/qbgf/v3/protocol/optional_phase_decisions.json)
- All51 aggregate validation conditions reproduce from141729 unique C/D rows after16674 matching duplicates are removed. The Part-I freeze retains FIXED_SENTENCE without LoRA; QBGF universal promotion was not established. Source: [results/week12/qbgf/v3/analysis/summary.csv](../../week12/qbgf/v3/analysis/summary.csv), [results/week12/qbgf/v3/analysis/primary_comparisons.csv](../../week12/qbgf/v3/analysis/primary_comparisons.csv)
- Both byte-identical Part-I freezes have SHA d0188a3582d6aead0a3a31f9f15bad9b51a88a6b475b05ac79ae38e58f05366b. FIXED_SENTENCE, no LoRA, five methods and fixed confirmation IDs were frozen. Source: [results/week12/W13_FINAL_METHOD_FREEZE.json](../../week12/W13_FINAL_METHOD_FREEZE.json), [results/week12/qbgf/v3/W13_FINAL_METHOD_FREEZE.json](../../week12/qbgf/v3/W13_FINAL_METHOD_FREEZE.json)
- A separate150-query/600-cell VAL attribution freeze exists. It compares QBGF and a VAL-selected comparator at both budgets, not fresh confirmation. Source: [results/week12/attribution/ATTRIBUTION_QUERY_FREEZE.json](../../week12/attribution/ATTRIBUTION_QUERY_FREEZE.json), [results/week12/attribution/v1/cost_plan.json](../../week12/attribution/v1/cost_plan.json)
- All600 cells completed, with16032 perturbation rows,9607 LOO units and13410 recorded physical calls. Across units687 are helpful,8311 neutral and609 harmful under the signed task-score criterion. Source: [results/week12/attribution/v1/status.json](../../week12/attribution/v1/status.json), [results/week12/attribution/v1/analysis/unit_importance.csv](../../week12/attribution/v1/analysis/unit_importance.csv)
- Ten of12 relevance/importance CIs include zero. Corrected finite retention has only6–34 eligible queries per curve; original curve_summary propagates NaN for undefined ratios. Original outputs remain preserved. Source: [results/week12/attribution/v1/analysis/relevance_utility.csv](../../week12/attribution/v1/analysis/relevance_utility.csv), [results/week12/attribution/v1/analysis/paired_harmful_effects.csv](../../week12/attribution/v1/analysis/paired_harmful_effects.csv)
- All2500 rows complete,2213 distinct cached generations. FIXED_SENTENCE exceeds NONE in all four cells after primary-family correction. SciFact256 sentence-minus-RAW_SKIP is+0.28762 [0.09709,0.47206]; at1024 passage point estimates are higher but paired intervals cross zero. QBGF wins HotpotQA1024 by0.03438 with secondary Holm p0.03160; no universal winner. Source: [results/week12/final_confirmation/v1/summary.csv](../../week12/final_confirmation/v1/summary.csv), [results/week12/final_confirmation/v1/contrasts.csv](../../week12/final_confirmation/v1/contrasts.csv)

## What did NOT work?

Fusion is not shown necessary beyond sentence-only at tight budgets; skipped LLMLingua/LoRA outcomes remain missing; numerous paired differences are uncertain.

## Limitations

VAL reused for model selection; post-hoc attribution;250-query two-task confirmation with43 SciFact groups and no fresh QASPER; no universal method, latency or internal-reasoning claim.

## Decision after this week

FIXED_SENTENCE remains frozen; QBGF universal promotion is not established. Confirmation wins in particular cells are reported without retroactive promotion.

## One-paragraph week takeaway

VAL counts are QASPER1005, HotpotQA7155 and SciFact177. Frozen confirmation is HotpotQA200, SciFact50, QASPER0. Historical QASPER TEST stays closed. Algorithm implementation supports exact-source sentences/passages, query-conditioned weights/mass and a hard additive token budget. Quality evidence belongs in C/D, not implementation tests. The qualified optional LLMLingua2 checkpoint is microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank @5f0c82792b7ea14c6484e015b6a072009496b7f2, not the XLM-R-large checkpoint. Qualification does not establish quality. QBGF lacks a robust universal VAL advantage. At256 QASPER QBGF0.235515 trails FIXED_SENTENCE0.263430; HotpotQA0.412068 slightly exceeds0.408247; SciFact0.674969 trails0.700062. At1024 passages are competitive or stronger in multiple tasks. The extension has33348 completed logical rows. At1024 HotpotQA SFR RAW_SKIP0.452133 exceeds BM250.435345; SciFact BM250.710716 exceeds SFR0.698979. No universal SFR claim follows. QBGF_FULL exceeds NO_QUERY in all six VAL cells by point estimate; e.g. HotpotQA2560.412068 vs0.308593. Sentence-only nearly duplicates FULL at256, so query conditioning evidence does not establish necessity of fusion. Optional-phase decisions explicitly mark E NOT_RUN_BY_USER_DECISION. A qualified checkpoint exists, but no full LLMLingua quality outcome is established. Full F training was skipped by explicit recorded decision. Scheduler/coverage implementation and tiny training qualification are not completed final LoRA training. Full G LoRA evaluation was not run; missing adapter results remain unestablished rather than assigned zero. All51 aggregate validation conditions reproduce from141729 unique C/D rows after16674 matching duplicates are removed. The Part-I freeze retains FIXED_SENTENCE without LoRA; QBGF universal promotion was not established. Both byte-identical Part-I freezes have SHA d0188a3582d6aead0a3a31f9f15bad9b51a88a6b475b05ac79ae38e58f05366b. FIXED_SENTENCE, no LoRA, five methods and fixed confirmation IDs were frozen. A separate150-query/600-cell VAL attribution freeze exists. It compares QBGF and a VAL-selected comparator at both budgets, not fresh confirmation. All600 cells completed, with16032 perturbation rows,9607 LOO units and13410 recorded physical calls. Across units687 are helpful,8311 neutral and609 harmful under the signed task-score criterion. Ten of12 relevance/importance CIs include zero. Corrected finite retention has only6–34 eligible queries per curve; original curve_summary propagates NaN for undefined ratios. Original outputs remain preserved. All2500 rows complete,2213 distinct cached generations. FIXED_SENTENCE exceeds NONE in all four cells after primary-family correction. SciFact256 sentence-minus-RAW_SKIP is+0.28762 [0.09709,0.47206]; at1024 passage point estimates are higher but paired intervals cross zero. QBGF wins HotpotQA1024 by0.03438 with secondary Holm p0.03160; no universal winner.

# Primary Sources Used

- [configs/week12/protocol.yaml](../../../configs/week12/protocol.yaml)
- [notebooks/week12/week12_0_protocol_and_split_audit.ipynb](../../../notebooks/week12/week12_0_protocol_and_split_audit.ipynb)
- [notebooks/week12/week12_A_qbgf_algorithm.ipynb](../../../notebooks/week12/week12_A_qbgf_algorithm.ipynb)
- [notebooks/week12/week12_B_baseline_qualification.ipynb](../../../notebooks/week12/week12_B_baseline_qualification.ipynb)
- [notebooks/week12/week12_C_EXT_baseline_extension.ipynb](../../../notebooks/week12/week12_C_EXT_baseline_extension.ipynb)
- [notebooks/week12/week12_C_validation_surface.ipynb](../../../notebooks/week12/week12_C_validation_surface.ipynb)
- [notebooks/week12/week12_D_qbgf_ablations.ipynb](../../../notebooks/week12/week12_D_qbgf_ablations.ipynb)
- [notebooks/week12/week12_E_llmlingua2.ipynb](../../../notebooks/week12/week12_E_llmlingua2.ipynb)
- [notebooks/week12/week12_F_lora_training.ipynb](../../../notebooks/week12/week12_F_lora_training.ipynb)
- [notebooks/week12/week12_G_lora_validation.ipynb](../../../notebooks/week12/week12_G_lora_validation.ipynb)
- [notebooks/week12/week12_H_validation_analysis.ipynb](../../../notebooks/week12/week12_H_validation_analysis.ipynb)
- [notebooks/week12/week12_I_week13_freeze.ipynb](../../../notebooks/week12/week12_I_week13_freeze.ipynb)
- [notebooks/week12/week12_J_attribution_setup.ipynb](../../../notebooks/week12/week12_J_attribution_setup.ipynb)
- [notebooks/week12/week12_K_causal_evidence_ablation.ipynb](../../../notebooks/week12/week12_K_causal_evidence_ablation.ipynb)
- [notebooks/week12/week12_L_attribution_analysis.ipynb](../../../notebooks/week12/week12_L_attribution_analysis.ipynb)
- [notebooks/week12/week12_M_final_confirmation.ipynb](../../../notebooks/week12/week12_M_final_confirmation.ipynb)
- [results/week12/W13_FINAL_METHOD_FREEZE.json](../../week12/W13_FINAL_METHOD_FREEZE.json)
- [results/week12/attribution/ATTRIBUTION_QUERY_FREEZE.json](../../week12/attribution/ATTRIBUTION_QUERY_FREEZE.json)
- [results/week12/attribution/v1/analysis/CLOSEOUT_AUDIT.json](../../week12/attribution/v1/analysis/CLOSEOUT_AUDIT.json)
- [results/week12/attribution/v1/analysis/curve_summary.csv](../../week12/attribution/v1/analysis/curve_summary.csv)
- [results/week12/attribution/v1/analysis/finite_retention_summary.csv](../../week12/attribution/v1/analysis/finite_retention_summary.csv)
- [results/week12/attribution/v1/analysis/paired_harmful_effects.csv](../../week12/attribution/v1/analysis/paired_harmful_effects.csv)
- [results/week12/attribution/v1/analysis/relevance_utility.csv](../../week12/attribution/v1/analysis/relevance_utility.csv)
- [results/week12/attribution/v1/analysis/system_quality.csv](../../week12/attribution/v1/analysis/system_quality.csv)
- [results/week12/attribution/v1/analysis/unit_importance.csv](../../week12/attribution/v1/analysis/unit_importance.csv)
- [results/week12/attribution/v1/analysis/unit_summary.csv](../../week12/attribution/v1/analysis/unit_summary.csv)
- [results/week12/attribution/v1/cost_plan.json](../../week12/attribution/v1/cost_plan.json)
- [results/week12/attribution/v1/status.json](../../week12/attribution/v1/status.json)
- [results/week12/final_confirmation/v1/FINAL_CLOSEOUT_AUDIT.json](../../week12/final_confirmation/v1/FINAL_CLOSEOUT_AUDIT.json)
- [results/week12/final_confirmation/v1/complete.json](../../week12/final_confirmation/v1/complete.json)
- [results/week12/final_confirmation/v1/contrasts.csv](../../week12/final_confirmation/v1/contrasts.csv)
- [results/week12/final_confirmation/v1/figures/paired_contrasts.png](../../week12/final_confirmation/v1/figures/paired_contrasts.png)
- [results/week12/final_confirmation/v1/figures/quality_by_budget.png](../../week12/final_confirmation/v1/figures/quality_by_budget.png)
- [results/week12/final_confirmation/v1/summary.csv](../../week12/final_confirmation/v1/summary.csv)
- [results/week12/final_confirmation/v1/validation_comparison.csv](../../week12/final_confirmation/v1/validation_comparison.csv)
- [results/week12/qbgf/v3/W13_FINAL_METHOD_FREEZE.json](../../week12/qbgf/v3/W13_FINAL_METHOD_FREEZE.json)
- [results/week12/qbgf/v3/analysis/fusion_diagnostic_comparisons.csv](../../week12/qbgf/v3/analysis/fusion_diagnostic_comparisons.csv)
- [results/week12/qbgf/v3/analysis/primary_comparisons.csv](../../week12/qbgf/v3/analysis/primary_comparisons.csv)
- [results/week12/qbgf/v3/analysis/summary.csv](../../week12/qbgf/v3/analysis/summary.csv)
- [results/week12/qbgf/v3/c_extension/v1/analysis/merged_validation_summary.csv](../../week12/qbgf/v3/c_extension/v1/analysis/merged_validation_summary.csv)
- [results/week12/qbgf/v3/c_extension/v1/status.json](../../week12/qbgf/v3/c_extension/v1/status.json)
- [results/week12/qbgf/v3/protocol/llmlingua_qualification_lock.json](../../week12/qbgf/v3/protocol/llmlingua_qualification_lock.json)
- [results/week12/qbgf/v3/protocol/optional_phase_decisions.json](../../week12/qbgf/v3/protocol/optional_phase_decisions.json)
- [results/week12/qbgf/v3/protocol/training_execution_patch.json](../../week12/qbgf/v3/protocol/training_execution_patch.json)
- [results/week12/qbgf/v3/qualification/baselines.json](../../week12/qbgf/v3/qualification/baselines.json)
- [results/week12/qbgf/v3/validation/C/status.json](../../week12/qbgf/v3/validation/C/status.json)
- [results/week12/qbgf/v3/validation/D/status.json](../../week12/qbgf/v3/validation/D/status.json)
- [results/weekly_reports/week_12/OfflineValidationRecomputation.csv](OfflineValidationRecomputation.csv)
- [results/weekly_reports/week_12/OfflineValidationRecomputation.json](OfflineValidationRecomputation.json)
- [src/week12/algorithm.py](../../../src/week12/algorithm.py)

# Existing Markdown Used as Evidence

NONE
