# Week 4 — Summary Analysis

## Objective

Test whether observable query/retrieval/content features can choose SARA representation budgets, then diagnose direct and sequential allocation failures.

## Experiments performed

| Part               | Question                                                                       | Status    | Main_result                                                                                                                                                                   |
|:-------------------|:-------------------------------------------------------------------------------|:----------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FEATURE-DISCOVERY  | Which pregeneration feature families associate with budget/transition targets? | COMPLETED | All feature-family maximum absolute Spearman correlations are below 0.19; observed individual associations are weak.                                                          |
| FEATURE-VALIDATION | Do associations survive redundancy and multiple-comparison checks?             | COMPLETED | The six-family validation summary reports zero FDR-significant relationships at q0.10 in every family. Provisional feature retention is exploratory, not confirmed discovery. |
| QCCA-V1            | Can direct budget prediction recover retrospective headroom?                   | COMPLETED | QCCA-LogReg has F1 0.3034 and mean k2.03, below static k8 0.3950; high target-label agreement does not imply downstream quality.                                              |
| MODEL-BENCHMARK    | Do transition classifiers improve the quality/cost decision?                   | COMPLETED | Balanced logistic transition classification at threshold0.5 reaches recorded policy F1 0.3842; model-family/threshold tuning reuses DEV and is exploratory.                   |
| V2-MODELING        | Does sequential expansion avoid direct-classification collapse?                | COMPLETED | Unbalanced sequential logistic policy collapses to k2 (F1 0.3038). Balanced logistic reaches 0.3318; ridge reaches 0.3675, still below static k8.                             |
| V2-DIAGNOSTIC      | What do calibration and transition errors reveal about failures?               | COMPLETED | Stored balanced/ridge policy F1 values are 0.3318/0.3675; transition quality and policy quality are distinct quantities.                                                      |
| V2-FINAL           | Does the selected C1 transition policy retain quality while reducing context?  | COMPLETED | Final C1 F1 is 0.3842 at 1115.9 tokens, versus k6 0.385699 at1359.1 and k8 0.394995 at1758.5; paired F1 intervals versus all three static references cross zero.              |

## Key results

### feature family summary

| feature_family                  |   n_features |   max_abs_spearman_rho |   median_abs_spearman_rho |   mean_abs_spearman_rho |   n_features_fold_stable | strongest_feature           | strongest_target   |   strongest_rho |
|:--------------------------------|-------------:|-----------------------:|--------------------------:|------------------------:|-------------------------:|:----------------------------|:-------------------|----------------:|
| B: Query Complexity             |           12 |               0.188800 |                  0.054300 |                0.062000 |                        9 | query_conjunction_count     | oracle_k           |        0.188800 |
| A: BM25/Retrieval               |           17 |               0.174300 |                  0.049700 |                0.056500 |                       17 | bm25_mean_score             | G_4_to_6           |        0.174300 |
| E: Lexical Coverage             |            8 |               0.173500 |                  0.074800 |                0.075700 |                        8 | lex_coverage_top1           | oracle_k           |        0.173500 |
| C: Semantic Structure           |           11 |               0.172100 |                  0.066100 |                0.072100 |                       11 | sem_sim_mean_top6           | G_4_to_6           |        0.172100 |
| D: Passage Redundancy/Diversity |           10 |               0.156900 |                  0.036300 |                0.043400 |                       10 | passage_sim_std             | G_6_to_8           |        0.156900 |
| F: Evidence Structure           |            6 |               0.152200 |                  0.046500 |                0.052600 |                        6 | evidence_query_cluster_span | G_4_to_6           |        0.152200 |

Source: [results/week4/feature_discovery/feature_family_summary.csv](../../week4/feature_discovery/feature_family_summary.csv).

### feature family validation summary

| feature_family                  |   original_features |   fdr_sig_relationships_q10 |   bootstrap_ci_excludes_zero |   median_abs_rho |   max_abs_rho |   median_fold_stability | representative_features                                                                |
|:--------------------------------|--------------------:|----------------------------:|-----------------------------:|-----------------:|--------------:|------------------------:|:---------------------------------------------------------------------------------------|
| A: BM25/Retrieval               |                  17 |                           0 |                            6 |         0.049700 |      0.174300 |                0.800000 | bm25_mean_score, bm25_top1_score, bm25_first_gap_ratio                                 |
| B: Query Complexity             |                  12 |                           0 |                            6 |         0.054300 |      0.188800 |                0.600000 | query_conjunction_count, query_is_what_which, query_is_factoid                         |
| C: Semantic Structure           |                  11 |                           0 |                           10 |         0.066100 |      0.172100 |                0.600000 | sem_sim_mean_top6, sem_sim_top2, sem_sim_gap_12                                        |
| D: Passage Redundancy/Diversity |                  10 |                           0 |                            1 |         0.036300 |      0.156900 |                0.600000 | passage_sim_std, passage_sim_min, passage_sim_mean                                     |
| E: Lexical Coverage             |                   8 |                           0 |                            9 |         0.074800 |      0.173500 |                0.800000 | lex_coverage_top1, lex_coverage_top2, lex_coverage_gain_1_to_4                         |
| F: Evidence Structure           |                   6 |                           0 |                            1 |         0.046500 |      0.152200 |                0.600000 | evidence_query_cluster_span, evidence_lexical_overlap_mean, evidence_unique_word_count |

Source: [results/week4/feature_validation/feature_family_validation_summary.csv](../../week4/feature_validation/feature_family_validation_summary.csv).

### Retained exploratory features

| feature                  | family                          |   spearman_rho |   q_value_fdr |   bootstrap_ci_low |   bootstrap_ci_high |
|:-------------------------|:--------------------------------|---------------:|--------------:|-------------------:|--------------------:|
| query_conjunction_count  | B: Query Complexity             |       0.188800 |      0.410400 |           0.049400 |            0.329100 |
| query_is_what_which      | B: Query Complexity             |       0.175200 |      0.410400 |           0.041400 |            0.301300 |
| query_is_numerical       | B: Query Complexity             |      -0.123000 |      0.496400 |          -0.174300 |           -0.053500 |
| lex_coverage_top1        | E: Lexical Coverage             |       0.173500 |      0.410400 |           0.044400 |            0.316400 |
| lex_coverage_top2        | E: Lexical Coverage             |      -0.170100 |      0.410400 |          -0.300200 |           -0.040500 |
| lex_coverage_gain_1_to_4 | E: Lexical Coverage             |      -0.147900 |      0.455200 |          -0.256600 |           -0.018700 |
| sem_sim_mean_top6        | C: Semantic Structure           |       0.172100 |      0.410400 |           0.026300 |            0.319600 |
| sem_sim_top1             | C: Semantic Structure           |       0.135400 |      0.496400 |           0.001900 |            0.264900 |
| bm25_mean_score          | A: BM25/Retrieval               |       0.174300 |      0.410400 |           0.046000 |            0.295800 |
| bm25_first_gap_ratio     | A: BM25/Retrieval               |       0.147800 |      0.455200 |           0.011700 |            0.275700 |
| passage_sim_std          | D: Passage Redundancy/Diversity |       0.156900 |      0.455200 |           0.015400 |            0.283100 |
| passage_cluster_count    | D: Passage Redundancy/Diversity |       0.076600 |      0.710400 |          -0.053400 |            0.202500 |

Source: [results/week4/feature_validation/provisional_feature_set.csv](../../week4/feature_validation/provisional_feature_set.csv). First 12 of 14 rows shown; detailed report and source preserve the full comparison.

### model comparison

| system                  |   mean_f1 |   mean_k |   mean_context_tokens |   context_reduction_pct_vs_k8 |   mean_epsilon_regret |   mean_oracle_regret |   target_accuracy |   oracle_headroom_recovered_pct |
|:------------------------|----------:|---------:|----------------------:|------------------------------:|----------------------:|---------------------:|------------------:|--------------------------------:|
| Static k=5              |  0.362900 | 5.000000 |           1157.000000 |                     34.210000 |              0.126600 |             0.126600 |          0.073600 |                      -34.000000 |
| Static k=8              |  0.395000 | 8.000000 |           1758.500000 |                      0.000000 |              0.094500 |             0.094500 |          0.099600 |                        0.000000 |
| Best Static SARA (k=8)  |  0.395000 | 8.000000 |           1758.500000 |                      0.000000 |              0.094500 |             0.094500 |          0.099600 |                        0.000000 |
| Random Allocation       |  0.370900 | 4.900000 |           1131.000000 |                     35.680000 |              0.118500 |             0.118600 |          0.203500 |                      -25.450000 |
| QCCA-Rule               |  0.328600 | 3.560000 |            854.400000 |                     51.410000 |              0.160800 |             0.160900 |          0.324700 |                      -70.260000 |
| QCCA-LogReg             |  0.303400 | 2.030000 |            539.800000 |                     69.300000 |              0.186000 |             0.186200 |          0.567100 |                      -96.960000 |
| QCCA-Tree               |  0.328200 | 2.660000 |            671.000000 |                     61.840000 |              0.161200 |             0.161300 |          0.489200 |                      -70.700000 |
| Epsilon Oracle (ε=0.01) |  0.489400 | 3.490000 |            841.300000 |                     52.160000 |              0.000000 |             0.000100 |          1.000000 |                       99.880000 |
| Quality Oracle          |  0.489500 | 3.530000 |            848.400000 |                     51.750000 |              0.000000 |             0.000000 |          0.982700 |                      100.000000 |

Source: [results/week4/processed/model_comparison.csv](../../week4/processed/model_comparison.csv).

### policy comparison

| system                                                                |   mean_f1 |   mean_k |   mean_context_tokens |   context_reduction_pct_vs_k8 |   mean_regret | system_type             |
|:----------------------------------------------------------------------|----------:|---------:|----------------------:|------------------------------:|--------------:|:------------------------|
| Static SARA k=2                                                       |  0.303835 | 2.000000 |            534.500000 |                     69.600000 |      0.185672 | Static Baseline         |
| Static SARA k=4                                                       |  0.359601 | 4.000000 |            946.100000 |                     46.200000 |      0.129906 | Static Baseline         |
| Static SARA k=5                                                       |  0.362858 | 5.000000 |           1157.000000 |                     34.210000 |      0.126649 | Static Baseline         |
| Static SARA k=6                                                       |  0.385699 | 6.000000 |           1359.100000 |                     22.710000 |      0.103808 | Static Baseline         |
| Static SARA k=8                                                       |  0.394995 | 8.000000 |           1758.500000 |                      0.000000 |      0.094512 | Static Baseline         |
| Random Allocation                                                     |  0.370900 | 4.900000 |           1131.000000 |                     35.680000 |      0.118500 | Unlearned Baseline      |
| Oracle Sequential Ceiling                                             |  0.406900 | 2.560000 |            650.300000 |                     63.020000 |      0.082500 | Theoretical Bound       |
| Policy C1 [Balanced Logistic Regression (5 feats, Baseline)] (t=0.50) |  0.384200 | 4.820000 |           1115.900000 |                     36.540000 |      0.105300 | Learned Policy (t=0.50) |
| Policy C1 [Balanced Logistic Regression (2 feats: BM25+Lex)] (t=0.50) |  0.380000 | 4.830000 |           1117.700000 |                     36.440000 |      0.109500 | Learned Policy (t=0.50) |
| Policy C1 [Shallow Decision Tree (5 feats)] (t=0.50)                  |  0.367100 | 4.640000 |           1078.400000 |                     38.670000 |      0.122400 | Learned Policy (t=0.50) |
| Policy C1 [Random Forest (5 feats)] (t=0.50)                          |  0.371700 | 4.730000 |           1096.300000 |                     37.660000 |      0.117800 | Learned Policy (t=0.50) |
| Policy C1 [XGBoost (5 feats)] (t=0.50)                                |  0.372500 | 4.560000 |           1062.300000 |                     39.590000 |      0.117000 | Learned Policy (t=0.50) |

Source: [results/week4/model_benchmark/policy_comparison.csv](../../week4/model_benchmark/policy_comparison.csv). First 12 of 19 rows shown; detailed report and source preserve the full comparison.

### Stored bootstrap contrasts; interpret named metric/difference exactly

| candidate                                        | baseline                                         |   mean_delta_f1 |   ci_95_f1_low |   ci_95_f1_high | statistically_detectable_f1   |   mean_delta_tokens |   ci_95_tokens_low |   ci_95_tokens_high | statistically_detectable_tokens   |   mean_delta_regret |   ci_95_regret_low |   ci_95_regret_high |
|:-------------------------------------------------|:-------------------------------------------------|----------------:|---------------:|----------------:|:------------------------------|--------------------:|-------------------:|--------------------:|:----------------------------------|--------------------:|-------------------:|--------------------:|
| Balanced Logistic Regression (5 feats, Baseline) | Static SARA k=8                                  |       -0.011200 |      -0.044200 |        0.021200 | False                         |         -642.400000 |        -676.100000 |         -609.400000 | True                              |            0.011200 |          -0.021200 |            0.044200 |
| Balanced Logistic Regression (5 feats, Baseline) | Static SARA k=6                                  |       -0.001600 |      -0.022300 |        0.018200 | False                         |         -243.000000 |        -276.700000 |         -210.000000 | True                              |            0.001600 |          -0.018200 |            0.022300 |
| Balanced Logistic Regression (2 feats: BM25+Lex) | Balanced Logistic Regression (5 feats, Baseline) |       -0.004200 |      -0.017700 |        0.009100 | False                         |            1.900000 |         -16.300000 |           20.800000 | False                             |            0.004200 |          -0.009100 |            0.017700 |
| Balanced Logistic Regression (2 feats: BM25+Lex) | Static SARA k=8                                  |       -0.015400 |      -0.047000 |        0.015500 | False                         |         -640.600000 |        -672.100000 |         -608.500000 | True                              |            0.015400 |          -0.015500 |            0.047000 |
| Balanced Logistic Regression (2 feats: BM25+Lex) | Static SARA k=6                                  |       -0.005800 |      -0.027700 |        0.016400 | False                         |         -241.200000 |        -272.700000 |         -209.100000 | True                              |            0.005800 |          -0.016400 |            0.027700 |
| Shallow Decision Tree (5 feats)                  | Balanced Logistic Regression (5 feats, Baseline) |       -0.016800 |      -0.034800 |        0.000100 | False                         |          -37.600000 |         -66.500000 |           -9.300000 | True                              |            0.016800 |          -0.000100 |            0.034800 |
| Shallow Decision Tree (5 feats)                  | Static SARA k=8                                  |       -0.028000 |      -0.060200 |        0.003900 | False                         |         -680.000000 |        -706.800000 |         -653.800000 | True                              |            0.028000 |          -0.003900 |            0.060200 |
| Shallow Decision Tree (5 feats)                  | Static SARA k=6                                  |       -0.018400 |      -0.043700 |        0.007200 | False                         |         -280.600000 |        -307.400000 |         -254.400000 | True                              |            0.018400 |          -0.007200 |            0.043700 |
| Random Forest (5 feats)                          | Balanced Logistic Regression (5 feats, Baseline) |       -0.012400 |      -0.030000 |        0.003800 | False                         |          -19.300000 |         -42.600000 |            5.600000 | False                             |            0.012400 |          -0.003800 |            0.030000 |
| Random Forest (5 feats)                          | Static SARA k=8                                  |       -0.023600 |      -0.054600 |        0.007000 | False                         |         -661.800000 |        -691.900000 |         -630.900000 | True                              |            0.023600 |          -0.007000 |            0.054600 |
| Random Forest (5 feats)                          | Static SARA k=6                                  |       -0.014000 |      -0.041100 |        0.011600 | False                         |         -262.400000 |        -292.500000 |         -231.500000 | True                              |            0.014000 |          -0.011600 |            0.041100 |
| XGBoost (5 feats)                                | Balanced Logistic Regression (5 feats, Baseline) |       -0.011500 |      -0.031300 |        0.007700 | False                         |          -53.400000 |         -77.800000 |          -29.800000 | True                              |            0.011500 |          -0.007700 |            0.031300 |

Source: [results/week4/model_benchmark/bootstrap_comparison.csv](../../week4/model_benchmark/bootstrap_comparison.csv). First 12 of 35 rows shown; detailed report and source preserve the full comparison.

### qcca v2 policy summary

| policy                        |   mean_f1 |   mean_k |   mean_context_tokens |   context_reduction_pct_vs_k8 |   f1_per_1k_tokens |   mean_regret |   median_regret |     pct_k2 |    pct_k4 |    pct_k6 |    pct_k8 |   stage1_expansion_prec |   stage1_expansion_rec |   stage2_expansion_prec |   stage2_expansion_rec |   stage3_expansion_prec |   stage3_expansion_rec |
|:------------------------------|----------:|---------:|----------------------:|------------------------------:|-------------------:|--------------:|----------------:|-----------:|----------:|----------:|----------:|------------------------:|-----------------------:|------------------------:|-----------------------:|------------------------:|-----------------------:|
| QCCA-V2 (Logistic Regression) |  0.303800 | 2.000000 |            534.500000 |                     69.600000 |           0.568400 |      0.185600 |        0.000000 | 100.000000 |  0.000000 |  0.000000 |  0.000000 |                0.000000 |               0.000000 |                0.000000 |               0.000000 |                0.000000 |               0.000000 |
| QCCA-V2 (Balanced LogReg)     |  0.331800 | 3.500000 |            842.600000 |                     52.080000 |           0.393700 |      0.157700 |        0.000000 |  55.000000 | 19.000000 | 22.100000 |  3.900000 |                0.201900 |               0.381800 |                0.316700 |               0.791700 |                0.222200 |               0.200000 |
| QCCA-V2 (Decision Tree d=2)   |  0.306700 | 2.080000 |            550.500000 |                     68.690000 |           0.557100 |      0.182700 |        0.000000 |  96.500000 |  3.000000 |  0.400000 |  0.000000 |                0.125000 |               0.018200 |                0.000000 |               0.000000 |                0.000000 |               0.000000 |
| QCCA-V2 (Ridge Regression)    |  0.367500 | 5.450000 |           1241.600000 |                     29.390000 |           0.296000 |      0.122000 |        0.000000 |  12.600000 | 26.400000 | 37.200000 | 23.800000 |                0.232700 |               0.854500 |                0.205700 |               0.725000 |                0.109100 |               0.333300 |

Source: [results/week4/qcca_v2/qcca_v2_policy_summary.csv](../../week4/qcca_v2/qcca_v2_policy_summary.csv).

## Main scientific conclusions

- All feature-family maximum absolute Spearman correlations are below 0.19; observed individual associations are weak. Source: [results/week4/feature_discovery/feature_family_summary.csv](../../week4/feature_discovery/feature_family_summary.csv)
- The six-family validation summary reports zero FDR-significant relationships at q0.10 in every family. Provisional feature retention is exploratory, not confirmed discovery. Source: [results/week4/feature_validation/feature_family_validation_summary.csv](../../week4/feature_validation/feature_family_validation_summary.csv)
- QCCA-LogReg has F1 0.3034 and mean k2.03, below static k8 0.3950; high target-label agreement does not imply downstream quality. Source: [results/week4/processed/model_comparison.csv](../../week4/processed/model_comparison.csv)
- Balanced logistic transition classification at threshold0.5 reaches recorded policy F1 0.3842; model-family/threshold tuning reuses DEV and is exploratory. Source: [results/week4/model_benchmark/policy_comparison.csv](../../week4/model_benchmark/policy_comparison.csv)
- Unbalanced sequential logistic policy collapses to k2 (F1 0.3038). Balanced logistic reaches 0.3318; ridge reaches 0.3675, still below static k8. Source: [results/week4/qcca_v2/qcca_v2_policy_summary.csv](../../week4/qcca_v2/qcca_v2_policy_summary.csv)
- Stored balanced/ridge policy F1 values are 0.3318/0.3675; transition quality and policy quality are distinct quantities. Source: [results/week4/qcca_v2_diagnostic/transition_model_comparison.csv](../../week4/qcca_v2_diagnostic/transition_model_comparison.csv)
- Final C1 F1 is 0.3842 at 1115.9 tokens, versus k6 0.385699 at1359.1 and k8 0.394995 at1758.5; paired F1 intervals versus all three static references cross zero. Source: [results/week4/qcca_v2_final_validation/policy_pareto_points.csv](../../week4/qcca_v2_final_validation/policy_pareto_points.csv)

## What did NOT work?

Several learned policies underperform static k8; no feature relationships survive family-reported q0.10 FDR; final F1 intervals cross zero.

## Limitations

OOF predictions do not erase exploratory feature/model/threshold selection on the same DEV cohort.

## Decision after this week

Final validation table identifies C1; no superiority claim follows. Later Week5 tests content representations and target formulations.

## One-paragraph week takeaway

All feature-family maximum absolute Spearman correlations are below 0.19; observed individual associations are weak. The six-family validation summary reports zero FDR-significant relationships at q0.10 in every family. Provisional feature retention is exploratory, not confirmed discovery. QCCA-LogReg has F1 0.3034 and mean k2.03, below static k8 0.3950; high target-label agreement does not imply downstream quality. Balanced logistic transition classification at threshold0.5 reaches recorded policy F1 0.3842; model-family/threshold tuning reuses DEV and is exploratory. Unbalanced sequential logistic policy collapses to k2 (F1 0.3038). Balanced logistic reaches 0.3318; ridge reaches 0.3675, still below static k8. Stored balanced/ridge policy F1 values are 0.3318/0.3675; transition quality and policy quality are distinct quantities. Final C1 F1 is 0.3842 at 1115.9 tokens, versus k6 0.385699 at1359.1 and k8 0.394995 at1758.5; paired F1 intervals versus all three static references cross zero.

# Primary Sources Used

- [configs/week4/week4_config.yaml](../../../configs/week4/week4_config.yaml)
- [notebooks/week4/week4_feature_discovery.ipynb](../../../notebooks/week4/week4_feature_discovery.ipynb)
- [notebooks/week4/week4_feature_validation.ipynb](../../../notebooks/week4/week4_feature_validation.ipynb)
- [notebooks/week4/week4_model_benchmark.ipynb](../../../notebooks/week4/week4_model_benchmark.ipynb)
- [notebooks/week4/week4_qcca_development.ipynb](../../../notebooks/week4/week4_qcca_development.ipynb)
- [notebooks/week4/week4_qcca_v2_diagnostic.ipynb](../../../notebooks/week4/week4_qcca_v2_diagnostic.ipynb)
- [notebooks/week4/week4_qcca_v2_final_validation.ipynb](../../../notebooks/week4/week4_qcca_v2_final_validation.ipynb)
- [notebooks/week4/week4_qcca_v2_modeling.ipynb](../../../notebooks/week4/week4_qcca_v2_modeling.ipynb)
- [results/week4/feature_discovery/feature_family_summary.csv](../../week4/feature_discovery/feature_family_summary.csv)
- [results/week4/feature_validation/feature_family_validation_summary.csv](../../week4/feature_validation/feature_family_validation_summary.csv)
- [results/week4/feature_validation/provisional_feature_set.csv](../../week4/feature_validation/provisional_feature_set.csv)
- [results/week4/model_benchmark/bootstrap_comparison.csv](../../week4/model_benchmark/bootstrap_comparison.csv)
- [results/week4/model_benchmark/policy_comparison.csv](../../week4/model_benchmark/policy_comparison.csv)
- [results/week4/processed/model_comparison.csv](../../week4/processed/model_comparison.csv)
- [results/week4/qcca_v2/bootstrap_policy_comparison.csv](../../week4/qcca_v2/bootstrap_policy_comparison.csv)
- [results/week4/qcca_v2/qcca_v2_policy_summary.csv](../../week4/qcca_v2/qcca_v2_policy_summary.csv)
- [results/week4/qcca_v2_diagnostic/figures/policy_pareto_frontier.png](../../week4/qcca_v2_diagnostic/figures/policy_pareto_frontier.png)
- [results/week4/qcca_v2_diagnostic/transition_model_comparison.csv](../../week4/qcca_v2_diagnostic/transition_model_comparison.csv)
- [results/week4/qcca_v2_final_validation/policy_pareto_points.csv](../../week4/qcca_v2_final_validation/policy_pareto_points.csv)
- [results/week4/qcca_v2_final_validation/statistical_comparisons.csv](../../week4/qcca_v2_final_validation/statistical_comparisons.csv)

# Existing Markdown Used as Evidence

NONE
