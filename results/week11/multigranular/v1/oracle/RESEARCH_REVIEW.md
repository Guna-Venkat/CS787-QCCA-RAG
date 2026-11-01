# Week 11 H — oracle and interactions: research interpretation

H is complete and frozen. It reads 31,900 immutable generation-result rows, deduplicated into 25,300 query/action observations and 506 dataset/condition summaries. No new model generation occurs. Supplemental cross-phase duplicate checks found no disagreement in prediction, utility, evidence hash, model hash or source hash.

## Retrospective action ceilings

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

The controlled Qwen action surface has oracle headroom of 0.125116 F1 on HotpotQA, 0.124146 on QASPER, and 0.18 strict accuracy on SciFact. This reveals outcome heterogeneity across the tested actions. It does not demonstrate learnability from inference-time features, sequential stopping, or a deployable controller. QA epsilon is 0.01; SciFact uses epsilon 0. The ALL_READERS QASPER ceiling includes the distinct trained SARA specialist, so it cannot be attributed to portable text representations.

## Named interaction formula and controls

Every difference-in-differences is (A − B) − (C − D). Action identities are decoded here to prevent a sign-only interpretation.

| name                               | cell_A                     | cell_B                      | cell_C                   | cell_D                    | formula           |
|:-----------------------------------|:---------------------------|:----------------------------|:-------------------------|:--------------------------|:------------------|
| QASPER_reader_by_granularity_512   | Mistral/SFR/PASSAGE/C0/512 | Mistral/SFR/SENTENCE/C0/512 | Qwen/SFR/PASSAGE/C0/512  | Qwen/SFR/SENTENCE/C0/512  | (A - B) - (C - D) |
| QASPER_granularity_by_budget       | Qwen/SFR/PASSAGE/C0/256    | Qwen/SFR/PASSAGE/C0/1024    | Qwen/SFR/SENTENCE/C0/256 | Qwen/SFR/SENTENCE/C0/1024 | (A - B) - (C - D) |
| QASPER_compressor_by_budget        | Qwen/SFR/PASSAGE/C0/256    | Qwen/SFR/PASSAGE/C0/1024    | Qwen/SFR/SENTENCE/C2/256 | Qwen/SFR/SENTENCE/C2/1024 | (A - B) - (C - D) |
| QASPER_retriever_by_granularity    | Qwen/BM25/PASSAGE/C0/1024  | Qwen/BM25/SENTENCE/C0/1024  | Qwen/SFR/PASSAGE/C0/1024 | Qwen/SFR/SENTENCE/C0/1024 | (A - B) - (C - D) |
| HotpotQA_reader_by_granularity_512 | Mistral/SFR/PASSAGE/C0/512 | Mistral/SFR/SENTENCE/C0/512 | Qwen/SFR/PASSAGE/C0/512  | Qwen/SFR/SENTENCE/C0/512  | (A - B) - (C - D) |
| HotpotQA_granularity_by_budget     | Qwen/SFR/PASSAGE/C0/256    | Qwen/SFR/PASSAGE/C0/1024    | Qwen/SFR/SENTENCE/C0/256 | Qwen/SFR/SENTENCE/C0/1024 | (A - B) - (C - D) |
| HotpotQA_compressor_by_budget      | Qwen/SFR/PASSAGE/C0/256    | Qwen/SFR/PASSAGE/C0/1024    | Qwen/SFR/SENTENCE/C2/256 | Qwen/SFR/SENTENCE/C2/1024 | (A - B) - (C - D) |
| HotpotQA_retriever_by_granularity  | Qwen/BM25/PASSAGE/C0/1024  | Qwen/BM25/SENTENCE/C0/1024  | Qwen/SFR/PASSAGE/C0/1024 | Qwen/SFR/SENTENCE/C0/1024 | (A - B) - (C - D) |
| SciFact_reader_by_granularity_512  | Mistral/SFR/PASSAGE/C0/512 | Mistral/SFR/SENTENCE/C0/512 | Qwen/SFR/PASSAGE/C0/512  | Qwen/SFR/SENTENCE/C0/512  | (A - B) - (C - D) |
| SciFact_granularity_by_budget      | Qwen/SFR/PASSAGE/C0/256    | Qwen/SFR/PASSAGE/C0/1024    | Qwen/SFR/SENTENCE/C0/256 | Qwen/SFR/SENTENCE/C0/1024 | (A - B) - (C - D) |
| SciFact_compressor_by_budget       | Qwen/SFR/PASSAGE/C0/256    | Qwen/SFR/PASSAGE/C0/1024    | Qwen/SFR/SENTENCE/C2/256 | Qwen/SFR/SENTENCE/C2/1024 | (A - B) - (C - D) |
| SciFact_retriever_by_granularity   | Qwen/BM25/PASSAGE/C0/1024  | Qwen/BM25/SENTENCE/C0/1024  | Qwen/SFR/PASSAGE/C0/1024 | Qwen/SFR/SENTENCE/C0/1024 | (A - B) - (C - D) |

## Paired interaction results

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

The negative granularity-by-budget contrasts mean the passage 256-to-1024 utility gain exceeds the sentence gain (QASPER and SciFact nominal intervals exclude zero; HotpotQA is inconclusive). Similarly, all three negative compressor-by-budget contrasts mean raw C0 benefits more from increasing budget than compact C2 does. This supports budget-dependent representation behavior and is consistent with raw low-budget prefix failures. It does not isolate irrelevant-text suppression from better budget fit.

Reader-by-granularity intervals at 512 all include zero. Retriever-by-granularity intervals also include or touch zero. Therefore this panel does not establish either interaction. HotpotQA retriever interaction is exactly zero because the capped provided-context pool makes the compared evidence identical, not because open-corpus retriever choice is irrelevant.

All 12 contrasts use paired 10,000-resample bootstrap, seed 42; QASPER paper clusters and other query/claim clusters. They were explicitly configured before D/E/F completed, but after exploratory C inspection, and remain nominal exploratory analyses without multiplicity correction. Descriptive factorial means from the unequal blocks are not causal main effects. Format-confounded SciFact Mistral/Llama results remain strict primary outcomes.

## Limits and next step

Oracles over hundreds of tried actions are optimistic retrospective ceilings. A learned stopping or routing policy requires a new protocol, query/paper-disjoint training and confirmation, inference-time features only, and comparison with a fixed sentence baseline. No dynamic controller was trained by H. The W10 sampling-stability hook is a separate protocol, not three-reader Week 11 replication. LLMLingua remains pending; historical trajectory comparisons remain unspecified.
