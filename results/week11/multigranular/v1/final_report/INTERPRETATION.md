# Week 11 I — saved-result analysis

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

## Retrospective best by dataset

| dataset   | reader   | reader_variant               | retriever   | evidence_granularity   | selector   | budget   | compressor   |   task_score |   prompt_tokens |
|:----------|:---------|:-----------------------------|:------------|:-----------------------|:-----------|:---------|:-------------|-------------:|----------------:|
| HotpotQA  | Qwen     | PUBLIC_TEXT_READER           | SFR         | HIERARCHICAL           | semantic   | 1024     | C0           |     0.487169 |      917.060000 |
| QASPER    | Mistral  | MISTRAL_ONLY_LATENT_BASELINE | SFR         | PASSAGE                | semantic   | FULL     | C3           |     0.403749 |     1215.140000 |
| SciFact   | Qwen     | PUBLIC_TEXT_READER           | SFR         | PASSAGE                | hybrid     | 1024     | C0           |     0.780000 |      689.020000 |

## Next research step

Review the matched evidence-budget and reader/compressor contrasts before choosing a prospective policy. Freeze any candidate in a new confirmation protocol, retain no-evidence/raw-passage controls, and evaluate a larger untouched cohort. The present pilot cannot establish general RAG superiority or validate an adaptive controller.
