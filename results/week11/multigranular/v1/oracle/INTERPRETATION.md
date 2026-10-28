# Week 11 H — saved-result analysis

Verified 31,900 source rows by their immutable store checksums and aggregate status hash. No generation started in this phase.

## Retrospective oracle ceilings

| dataset   | reader                  | best_fixed           |   best_fixed_score |   exact_oracle |   epsilon_oracle |   oracle_headroom |   epsilon |
|:----------|:------------------------|:---------------------|-------------------:|---------------:|-----------------:|------------------:|----------:|
| HotpotQA  | ALL_READERS             | 0962fbdcf76305eedb48 |           0.487169 |       0.716332 |         0.716332 |          0.229163 |  0.010000 |
| HotpotQA  | Llama                   | 12dd48b33d6d3fb8d217 |           0.443019 |       0.587990 |         0.587990 |          0.144972 |  0.010000 |
| HotpotQA  | Mistral                 | 642251784cef20d8cbf0 |           0.324281 |       0.436404 |         0.436257 |          0.112123 |  0.010000 |
| HotpotQA  | Qwen                    | 0962fbdcf76305eedb48 |           0.487169 |       0.612285 |         0.612285 |          0.125116 |  0.010000 |
| QASPER    | ALL_READERS             | 4c4a8a869121679a3aa9 |           0.403749 |       0.623349 |         0.623055 |          0.219601 |  0.010000 |
| QASPER    | Llama                   | 2554cee82fe23d813c59 |           0.289146 |       0.410300 |         0.410146 |          0.121154 |  0.010000 |
| QASPER    | Mistral                 | e177fe617c51719a0d13 |           0.200750 |       0.364319 |         0.364234 |          0.163570 |  0.010000 |
| QASPER    | Mistral/SARA-specialist | 4c4a8a869121679a3aa9 |           0.403749 |       0.427414 |         0.427414 |          0.023665 |  0.010000 |
| QASPER    | Qwen                    | 0dc37be7e1b23dacb4e6 |           0.249211 |       0.373358 |         0.373223 |          0.124146 |  0.010000 |
| SciFact   | ALL_READERS             | ee2068a1bcf2d92d6dd1 |           0.780000 |       0.960000 |         0.960000 |          0.180000 |  0.000000 |
| SciFact   | Llama                   | 56b4550963a50e30655b |           0.040000 |       0.040000 |         0.040000 |          0.000000 |  0.000000 |
| SciFact   | Mistral                 | 0201e568dc562f50954e |           0.000000 |       0.000000 |         0.000000 |          0.000000 |  0.000000 |
| SciFact   | Qwen                    | ee2068a1bcf2d92d6dd1 |           0.780000 |       0.960000 |         0.960000 |          0.180000 |  0.000000 |

These ceilings select actions using observed outcomes on the same development pilot. They do not train or evaluate a deployable controller. Exact/epsilon choice is cost-aware only within the retrospective action surface. SciFact uses exact classification utility; QA uses epsilon 0.01.

## Interpretation limits

Use the C/D/E/F/G phase interpretations for controlled comparisons and flagged rows. Do not pool raw QA F1 and verification accuracy. ALL_READERS includes heterogeneous checkpoints and is a retrospective ceiling; SARA-specialist results stay separate from portable public-reader conclusions. C and D share cached Qwen observations; duplicate action/query records are deduplicated in aggregate analyses and are not independent replications. Marginal factorial summaries have unequal exposure and are descriptive, not causal effects. All point-estimate best policies were selected retrospectively on 50 development queries per dataset.

LLMLingua token conditions remain pending and are not implied by completion of the available text study. Historical trajectory records remain unspecified rather than inventing comparable scores across protocols. A larger preregistered cohort/new output version is needed for confirmation.

## Explicit exploratory interaction contrasts

| name                               | dataset   | family      |     point |    ci_low |   ci_high |   resamples |   seed | cluster           |
|:-----------------------------------|:----------|:------------|----------:|----------:|----------:|------------:|-------:|:------------------|
| QASPER_reader_by_granularity_512   | QASPER    | exploratory | -0.012382 | -0.049047 |  0.026008 |       10000 |     42 | paper_id          |
| QASPER_granularity_by_budget       | QASPER    | exploratory | -0.108726 | -0.171182 | -0.048261 |       10000 |     42 | paper_id          |
| QASPER_compressor_by_budget        | QASPER    | exploratory | -0.112550 | -0.168389 | -0.056251 |       10000 |     42 | paper_id          |
| QASPER_retriever_by_granularity    | QASPER    | exploratory |  0.004613 | -0.000114 |  0.011691 |       10000 |     42 | paper_id          |
| HotpotQA_reader_by_granularity_512 | HotpotQA  | exploratory | -0.009865 | -0.078200 |  0.062618 |       10000 |     42 | query_or_claim_id |
| HotpotQA_granularity_by_budget     | HotpotQA  | exploratory | -0.091934 | -0.214655 |  0.024745 |       10000 |     42 | query_or_claim_id |
| HotpotQA_compressor_by_budget      | HotpotQA  | exploratory | -0.160053 | -0.272444 | -0.056200 |       10000 |     42 | query_or_claim_id |
| HotpotQA_retriever_by_granularity  | HotpotQA  | exploratory |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact_reader_by_granularity_512  | SciFact   | exploratory |  0.020000 | -0.120000 |  0.160000 |       10000 |     42 | query_or_claim_id |
| SciFact_granularity_by_budget      | SciFact   | exploratory | -0.360000 | -0.560000 | -0.140000 |       10000 |     42 | query_or_claim_id |
| SciFact_compressor_by_budget       | SciFact   | exploratory | -0.320000 | -0.520000 | -0.120000 |       10000 |     42 | query_or_claim_id |
| SciFact_retriever_by_granularity   | SciFact   | exploratory |  0.040000 | -0.060000 |  0.140000 |       10000 |     42 | query_or_claim_id |

The action order for each named contrast is recorded in the configuration. Twelve difference-in-differences contrasts were specified before D/E/F finished. Intervals are nominal, not corrected for multiplicity.
