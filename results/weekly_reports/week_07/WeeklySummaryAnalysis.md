# Week 7 — Summary Analysis

## Objective

Test whether semantic routing plus adaptive budget decisions or medium-fidelity representations can improve static semantic context policies.

## Experiments performed

| Part   | Question                                                                  | Status    | Main_result                                                                                                                                                                                |
|:-------|:--------------------------------------------------------------------------|:----------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A      | Can the old W4 allocator benefit from semantic routing?                   | COMPLETED | Old allocator + semantic routing F1 is0.378150, below the same allocator with BM250.384172 and static semantic k40.391333.                                                                 |
| B      | Can retraining on a semantic response surface improve allocation?         | COMPLETED | New semantic-aware allocator F1 is0.3522 versus semantic k40.3909 and k80.4010; its retrospective epsilon oracle0.4869 is not deployable.                                                  |
| C      | Do richer routing-aware features identify the smallest sufficient budget? | COMPLETED | Selected sufficiency allocator F1 is0.378323 at mean k6.8485, below static semantic k40.390929.                                                                                            |
| C2     | Does direct utility regression/ranking improve the quality/cost tradeoff? | COMPLETED | Selected utility-ranking model F1 is0.392233 at mean k7.3853 and1622.10 tokens, below static k80.401041.                                                                                   |
| C3     | Can safe down/up shifts around semantic k4 improve the anchor?            | COMPLETED | Anchor-and-deviate F1 is0.380107 at903.10 tokens versus anchor k40.390929 at921.06 tokens; limited savings come with lower point quality.                                                  |
| D      | Do stronger learned representations make safer allocation changes?        | COMPLETED | Hierarchical frozen/unfrozen DEV policies collapse to static k4 (F10.390929); listwise7B LoRA reaches0.387714. Internal-VAL table values are different cohorts, not competing DEV scores.  |
| E      | Does adding medium extractive evidence recover k6 quality cheaply?        | COMPLETED | Repaired hybrid E1-H F10.3861 exceeds corrected extract-only0.3755, but trails static k40.3909 while using more tokens. Medium representation does not establish net quality/cost benefit. |

## Key results

### w7a main results

| policy                | routing               |   mean_k |   answer_f1_mean |   answer_f1_std |   context_tokens_mean |   latency_mean_sec |
|:----------------------|:----------------------|---------:|-----------------:|----------------:|----------------------:|-------------------:|
| Static k=2            | BM25 Prefix           | 2.000000 |         0.303835 |        0.379664 |            526.523810 |           0.000000 |
| Static k=4            | BM25 Prefix           | 4.000000 |         0.359601 |        0.393989 |            938.121212 |           0.000000 |
| Static k=6            | BM25 Prefix           | 6.000000 |         0.385699 |        0.390981 |           1351.116883 |           0.000000 |
| Static k=8            | BM25 Prefix           | 8.000000 |         0.395000 |        0.392000 |           1750.500000 |           4.800000 |
| Static k=2            | Semantic Top-k        | 2.000000 |         0.312678 |        0.381688 |            509.636364 |           1.836474 |
| Static k=4            | Semantic Top-k        | 4.000000 |         0.391333 |        0.400429 |            921.073593 |           2.684737 |
| Static k=6            | Semantic Top-k        | 6.000000 |         0.395345 |        0.393080 |           1325.090909 |           3.629630 |
| W4 Adaptive Allocator | BM25 Prefix           | 4.822511 |         0.384172 |        0.389792 |           1114.229437 |           3.065000 |
| W4 Adaptive Allocator | Semantic Top-k (W7-A) | 4.822511 |         0.378150 |        0.395410 |           1095.030303 |           3.094185 |

Source: [results/week7/w7a_budget_semantic_composition/tables/w7a_main_results.csv](../../week7/w7a_budget_semantic_composition/tables/w7a_main_results.csv).

### Stored bootstrap contrasts; interpret named metric/difference exactly

| comparison             | policy_A    | policy_B   |   delta_f1 |   f1_ci_lower |   f1_ci_upper |   p_value | significant   |   delta_tokens |   tokens_ci_lower |   tokens_ci_upper |
|:-----------------------|:------------|:-----------|-----------:|--------------:|--------------:|----------:|:--------------|---------------:|------------------:|------------------:|
| W4+Semantic_vs_W4+BM25 | W4+Semantic | W4+BM25    |  -0.006022 |     -0.049531 |      0.037156 |  0.814000 | False         |     -19.199134 |        -38.426170 |         -0.282759 |

Source: [results/week7/w7a_budget_semantic_composition/bootstrap/w7a_bootstrap_comparisons.csv](../../week7/w7a_budget_semantic_composition/bootstrap/w7a_bootstrap_comparisons.csv).

### w7b main dev results

| policy                                | routing                 |   mean_k |   answer_f1_mean |   answer_f1_std |   context_tokens_mean |   token_reduction_pct_vs_k8 |   f1_per_1k_tokens |
|:--------------------------------------|:------------------------|---------:|-----------------:|----------------:|----------------------:|----------------------------:|-------------------:|
| Static semantic k=2                   | Semantic Top-k          | 2.000000 |         0.311800 |        0.383300 |            509.600000 |                   70.670000 |           0.611900 |
| Static semantic k=4                   | Semantic Top-k          | 4.000000 |         0.390900 |        0.402000 |            921.100000 |                   46.990000 |           0.424400 |
| Static semantic k=5                   | Semantic Top-k          | 5.000000 |         0.393100 |        0.397700 |           1123.000000 |                   35.370000 |           0.350100 |
| Static semantic k=6                   | Semantic Top-k          | 6.000000 |         0.395700 |        0.394700 |           1325.100000 |                   23.730000 |           0.298600 |
| Static semantic k=8                   | Semantic Top-k          | 8.000000 |         0.401000 |        0.396300 |           1737.400000 |                    0.000000 |           0.230800 |
| W4 old allocator + semantic routing   | Semantic Top-k (W7-A)   | 4.820000 |         0.378200 |        0.395400 |           1095.000000 |                   36.980000 |           0.345400 |
| NEW semantic-aware adaptive allocator | Semantic Top-k (W7-B)   | 3.480000 |         0.352200 |        0.393200 |            815.400000 |                   53.070000 |           0.432000 |
| Semantic epsilon-oracle (eps=0.01)    | Semantic Top-k (Oracle) | 3.300000 |         0.486900 |        0.392300 |            775.200000 |                   55.380000 |           0.628200 |

Source: [results/week7/w7b_semantic_budget/tables/w7b_main_dev_results.csv](../../week7/w7b_semantic_budget/tables/w7b_main_dev_results.csv).

### Stored bootstrap contrasts; interpret named metric/difference exactly

| comparison                            |   delta_f1 | ci_95_f1           |   p_value | significant_at_05   |   delta_tokens | ci_95_tokens      |
|:--------------------------------------|-----------:|:-------------------|----------:|:--------------------|---------------:|:------------------|
| W7B_Adaptive_vs_W7A_Composite         |  -0.026000 | [-0.0802, +0.0303] |  0.320000 | False               |    -279.600000 | [-340.4, -223.2]  |
| W7B_Adaptive_vs_Static_Semantic_k4    |  -0.038700 | [-0.0751, -0.0077] |  0.007000 | True                |    -105.700000 | [-157.1, -57.8]   |
| W7B_Adaptive_vs_Static_Semantic_k6    |  -0.043500 | [-0.0873, -0.0038] |  0.034000 | True                |    -509.700000 | [-559.5, -462.3]  |
| W7B_Adaptive_vs_Static_Semantic_k8    |  -0.048800 | [-0.0906, -0.0091] |  0.017000 | True                |    -922.000000 | [-972.8, -870.5]  |
| Semantic_Oracle_vs_Static_Semantic_k8 |   0.085900 | [+0.0610, +0.1142] |  0.000500 | True                |    -962.200000 | [-1015.7, -902.2] |

Source: [results/week7/w7b_semantic_budget/bootstrap/w7b_bootstrap_comparisons.csv](../../week7/w7b_semantic_budget/bootstrap/w7b_bootstrap_comparisons.csv).

### w7c main dev results

| method                                                  |   mean_k |   mean_f1 |      tokens | notes                      |   ROH_sem |
|:--------------------------------------------------------|---------:|----------:|------------:|:---------------------------|----------:|
| B0: Static Semantic k=2                                 | 2.000000 |  0.311819 |  509.619048 | Static baseline            | -1.039176 |
| B1: Static Semantic k=4                                 | 4.000000 |  0.390929 |  921.056277 | Static baseline            | -0.117776 |
| B2: Static Semantic k=6                                 | 6.000000 |  0.395731 | 1325.073593 | Static baseline            | -0.061852 |
| B3: Static Semantic k=8                                 | 8.000000 |  0.401041 | 1737.411255 | Best static reference      |  0.000000 |
| B4: W7-A (Old W4 + Semantic Routing)                    | 4.820000 |  0.378200 | 1095.000000 | Frozen W4 composition      | -0.266036 |
| B5: W7-B (Semantic-Aware Old W4 Features)               | 3.480000 |  0.352200 |  815.400000 | W4 retrained               | -0.568860 |
| B6: Semantic Epsilon-Oracle                             | 3.300000 |  0.486900 |  775.200000 | Theoretical upper bound    |  1.000000 |
| B7: W7-C Retrieval-Aware Semantic Sufficiency Allocator | 6.848485 |  0.378323 | 1503.800866 | Winning F4 GBDT (tau=0.80) | -0.264598 |

Source: [results/week7/w7c_retrieval_aware_allocator/tables/w7c_main_dev_results.csv](../../week7/w7c_retrieval_aware_allocator/tables/w7c_main_dev_results.csv).

### Stored bootstrap contrasts; interpret named metric/difference exactly

| comparison                  |   delta_f1 |   ci_lower_f1 |   ci_upper_f1 |   p_value_f1 |   delta_tokens |   ci_lower_tokens |   ci_upper_tokens |
|:----------------------------|-----------:|--------------:|--------------:|-------------:|---------------:|------------------:|------------------:|
| W7-C vs Static Semantic k=4 |  -0.012606 |     -0.049099 |      0.019829 |     0.475000 |     582.744589 |        503.886497 |        659.507719 |
| W7-C vs Static Semantic k=6 |  -0.017408 |     -0.051682 |      0.013709 |     0.279000 |     178.727273 |        107.462990 |        248.906385 |
| W7-C vs Static Semantic k=8 |  -0.022718 |     -0.053577 |      0.003620 |     0.110000 |    -233.610390 |       -301.625401 |       -167.691813 |
| W7-C vs W7-B Allocator      |   0.026487 |     -0.006464 |      0.057484 |     0.125000 |     688.400866 |        596.713808 |        774.317959 |

Source: [results/week7/w7c_retrieval_aware_allocator/bootstrap/w7c_bootstrap_comparisons.csv](../../week7/w7c_retrieval_aware_allocator/bootstrap/w7c_bootstrap_comparisons.csv).

### w7c2 main dev results

| method                               |   mean_k |   mean_f1 |      tokens | notes                                      |   ROH_sem |
|:-------------------------------------|---------:|----------:|------------:|:-------------------------------------------|----------:|
| B0: Static Semantic k=2              | 2.000000 |  0.311819 |  509.619048 | Static baseline                            | -1.039176 |
| B1: Static Semantic k=4              | 4.000000 |  0.390929 |  921.056277 | Static baseline (Dominant)                 | -0.117776 |
| B2: Static Semantic k=6              | 6.000000 |  0.395731 | 1325.073593 | Static baseline                            | -0.061852 |
| B3: Static Semantic k=8              | 8.000000 |  0.401041 | 1737.411255 | Best static reference                      |  0.000000 |
| B4: W7-A (Old W4 + Semantic Routing) | 4.820000 |  0.378200 | 1095.000000 | W4 composition                             | -0.266036 |
| B5: W7-B (Old Features Retrained)    | 3.480000 |  0.352200 |  815.400000 | W4 retrained                               | -0.568860 |
| B6: W7-C1 (Sufficiency Classifier)   | 6.850000 |  0.378300 | 1503.800000 | Fallback k8 failure                        | -0.264871 |
| B7: Semantic Epsilon-Oracle          | 3.300000 |  0.486900 |  775.200000 | Theoretical upper bound                    |  1.000000 |
| B8: W7-C2 Winner (M1_Reg_delta_0.0)  | 7.385281 |  0.392233 | 1622.099567 | Winning W7-C2 model (M1_Direct_Regression) | -0.102593 |

Source: [results/week7/w7c2_utility_ranking/tables/w7c2_main_dev_results.csv](../../week7/w7c2_utility_ranking/tables/w7c2_main_dev_results.csv).

## Main scientific conclusions

- Old allocator + semantic routing F1 is0.378150, below the same allocator with BM250.384172 and static semantic k40.391333. Source: [results/week7/w7a_budget_semantic_composition/tables/w7a_main_results.csv](../../week7/w7a_budget_semantic_composition/tables/w7a_main_results.csv)
- New semantic-aware allocator F1 is0.3522 versus semantic k40.3909 and k80.4010; its retrospective epsilon oracle0.4869 is not deployable. Source: [results/week7/w7b_semantic_budget/tables/w7b_main_dev_results.csv](../../week7/w7b_semantic_budget/tables/w7b_main_dev_results.csv)
- Selected sufficiency allocator F1 is0.378323 at mean k6.8485, below static semantic k40.390929. Source: [results/week7/w7c_retrieval_aware_allocator/tables/w7c_main_dev_results.csv](../../week7/w7c_retrieval_aware_allocator/tables/w7c_main_dev_results.csv)
- Selected utility-ranking model F1 is0.392233 at mean k7.3853 and1622.10 tokens, below static k80.401041. Source: [results/week7/w7c2_utility_ranking/tables/w7c2_main_dev_results.csv](../../week7/w7c2_utility_ranking/tables/w7c2_main_dev_results.csv)
- Anchor-and-deviate F1 is0.380107 at903.10 tokens versus anchor k40.390929 at921.06 tokens; limited savings come with lower point quality. Source: [results/week7/w7c3_anchor_deviate/tables/w7c3_main_dev_results.csv](../../week7/w7c3_anchor_deviate/tables/w7c3_main_dev_results.csv)
- Hierarchical frozen/unfrozen DEV policies collapse to static k4 (F10.390929); listwise7B LoRA reaches0.387714. Internal-VAL table values are different cohorts, not competing DEV scores. Source: [results/week7/w7d_cross_encoder_router/tables/w7d_main_dev_results.csv](../../week7/w7d_cross_encoder_router/tables/w7d_main_dev_results.csv)
- Repaired hybrid E1-H F10.3861 exceeds corrected extract-only0.3755, but trails static k40.3909 while using more tokens. Medium representation does not establish net quality/cost benefit. Source: [results/week7/w7e_multifidelity/tables/w7e_main_dev_results.csv](../../week7/w7e_multifidelity/tables/w7e_main_dev_results.csv)

## What did NOT work?

Retrained, sufficiency, utility and anchor policies generally fail to improve static quality; multifidelity medium extraction adds cost without clear net gain.

## Limitations

Repeated exploratory DEV exposure, outcome-derived oracle targets, semantic-encoder limitations, baseline drift and separate train/VAL/DEV scopes.

## Decision after this week

Simple fixed semantic budgets remain strong references; learned global-k and medium-fidelity choices are not universally promoted.

## One-paragraph week takeaway

Old allocator + semantic routing F1 is0.378150, below the same allocator with BM250.384172 and static semantic k40.391333. New semantic-aware allocator F1 is0.3522 versus semantic k40.3909 and k80.4010; its retrospective epsilon oracle0.4869 is not deployable. Selected sufficiency allocator F1 is0.378323 at mean k6.8485, below static semantic k40.390929. Selected utility-ranking model F1 is0.392233 at mean k7.3853 and1622.10 tokens, below static k80.401041. Anchor-and-deviate F1 is0.380107 at903.10 tokens versus anchor k40.390929 at921.06 tokens; limited savings come with lower point quality. Hierarchical frozen/unfrozen DEV policies collapse to static k4 (F10.390929); listwise7B LoRA reaches0.387714. Internal-VAL table values are different cohorts, not competing DEV scores. Repaired hybrid E1-H F10.3861 exceeds corrected extract-only0.3755, but trails static k40.3909 while using more tokens. Medium representation does not establish net quality/cost benefit.

# Primary Sources Used

- [configs/week7/w7a_budget_semantic_composition.yaml](../../../configs/week7/w7a_budget_semantic_composition.yaml)
- [configs/week7/w7b_semantic_budget_relearning.yaml](../../../configs/week7/w7b_semantic_budget_relearning.yaml)
- [configs/week7/w7c2_utility_aware_budget_ranking.yaml](../../../configs/week7/w7c2_utility_aware_budget_ranking.yaml)
- [configs/week7/w7c3_anchor_deviate.yaml](../../../configs/week7/w7c3_anchor_deviate.yaml)
- [configs/week7/w7c_retrieval_aware_allocator.yaml](../../../configs/week7/w7c_retrieval_aware_allocator.yaml)
- [configs/week7/w7d_cross_encoder_fidelity_router.yaml](../../../configs/week7/w7d_cross_encoder_fidelity_router.yaml)
- [configs/week7/w7e_multifidelity.yaml](../../../configs/week7/w7e_multifidelity.yaml)
- [notebooks/week7/week7a_budget_semantic_composition.ipynb](../../../notebooks/week7/week7a_budget_semantic_composition.ipynb)
- [notebooks/week7/week7b_semantic_budget_relearning.ipynb](../../../notebooks/week7/week7b_semantic_budget_relearning.ipynb)
- [notebooks/week7/week7c2_utility_aware_budget_ranking.ipynb](../../../notebooks/week7/week7c2_utility_aware_budget_ranking.ipynb)
- [notebooks/week7/week7c3_anchor_and_deviate_policy.ipynb](../../../notebooks/week7/week7c3_anchor_and_deviate_policy.ipynb)
- [notebooks/week7/week7c_retrieval_aware_semantic_allocator.ipynb](../../../notebooks/week7/week7c_retrieval_aware_semantic_allocator.ipynb)
- [notebooks/week7/week7d_cross_encoder_fidelity_router.ipynb](../../../notebooks/week7/week7d_cross_encoder_fidelity_router.ipynb)
- [notebooks/week7/week7e_multifidelity_representation.ipynb](../../../notebooks/week7/week7e_multifidelity_representation.ipynb)
- [results/week7/w7a_budget_semantic_composition/bootstrap/w7a_bootstrap_comparisons.csv](../../week7/w7a_budget_semantic_composition/bootstrap/w7a_bootstrap_comparisons.csv)
- [results/week7/w7a_budget_semantic_composition/figures/w7a_quality_vs_tokens.png](../../week7/w7a_budget_semantic_composition/figures/w7a_quality_vs_tokens.png)
- [results/week7/w7a_budget_semantic_composition/tables/w7a_main_results.csv](../../week7/w7a_budget_semantic_composition/tables/w7a_main_results.csv)
- [results/week7/w7b_semantic_budget/bootstrap/w7b_bootstrap_comparisons.csv](../../week7/w7b_semantic_budget/bootstrap/w7b_bootstrap_comparisons.csv)
- [results/week7/w7b_semantic_budget/figures/w7b_quality_vs_tokens.png](../../week7/w7b_semantic_budget/figures/w7b_quality_vs_tokens.png)
- [results/week7/w7b_semantic_budget/tables/w7b_main_dev_results.csv](../../week7/w7b_semantic_budget/tables/w7b_main_dev_results.csv)
- [results/week7/w7c2_utility_ranking/bootstrap/w7c2_bootstrap_comparisons.csv](../../week7/w7c2_utility_ranking/bootstrap/w7c2_bootstrap_comparisons.csv)
- [results/week7/w7c2_utility_ranking/figures/w7c2_quality_vs_tokens.png](../../week7/w7c2_utility_ranking/figures/w7c2_quality_vs_tokens.png)
- [results/week7/w7c2_utility_ranking/tables/w7c2_main_dev_results.csv](../../week7/w7c2_utility_ranking/tables/w7c2_main_dev_results.csv)
- [results/week7/w7c3_anchor_deviate/bootstrap/w7c3_bootstrap_comparisons.csv](../../week7/w7c3_anchor_deviate/bootstrap/w7c3_bootstrap_comparisons.csv)
- [results/week7/w7c3_anchor_deviate/figures/w7c3_oof_quality_vs_tokens.png](../../week7/w7c3_anchor_deviate/figures/w7c3_oof_quality_vs_tokens.png)
- [results/week7/w7c3_anchor_deviate/tables/w7c3_main_dev_results.csv](../../week7/w7c3_anchor_deviate/tables/w7c3_main_dev_results.csv)
- [results/week7/w7c_retrieval_aware_allocator/bootstrap/w7c_bootstrap_comparisons.csv](../../week7/w7c_retrieval_aware_allocator/bootstrap/w7c_bootstrap_comparisons.csv)
- [results/week7/w7c_retrieval_aware_allocator/figures/w7c_quality_vs_tokens.png](../../week7/w7c_retrieval_aware_allocator/figures/w7c_quality_vs_tokens.png)
- [results/week7/w7c_retrieval_aware_allocator/tables/w7c_main_dev_results.csv](../../week7/w7c_retrieval_aware_allocator/tables/w7c_main_dev_results.csv)
- [results/week7/w7d_cross_encoder_router/bootstrap/w7d_bootstrap_comparisons.csv](../../week7/w7d_cross_encoder_router/bootstrap/w7d_bootstrap_comparisons.csv)
- [results/week7/w7d_cross_encoder_router/figures/w7d_quality_vs_tokens.png](../../week7/w7d_cross_encoder_router/figures/w7d_quality_vs_tokens.png)
- [results/week7/w7d_cross_encoder_router/tables/w7d_main_dev_results.csv](../../week7/w7d_cross_encoder_router/tables/w7d_main_dev_results.csv)
- [results/week7/w7e_multifidelity/tables/w7e_main_dev_results.csv](../../week7/w7e_multifidelity/tables/w7e_main_dev_results.csv)

# Existing Markdown Used as Evidence

NONE
