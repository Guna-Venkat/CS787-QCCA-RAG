# Week 0 — Summary Analysis

## Objective

The week probes dataset/evidence characteristics and retrieval-to-reader sensitivity before learning a context allocator. Source code defines dataset analysis, layer analysis and utilization analysis; separately named GPU studies and a later Week3 bridge coexist. The generic layer/utilization tables are deterministic mock outputs and are excluded from empirical model conclusions. The notebook has no executed code outputs; it supplies no Tier-2 outcome evidence.

## Experiments performed

| part                    | question                                                                                                                         | status                  | main_result                                                                                                                                                                                                   |
|:------------------------|:---------------------------------------------------------------------------------------------------------------------------------|:------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A                       | How much annotated evidence is present in the local QASPER splits, and how often does the existing retrieved pool recover it?    | COMPLETED               | The local dev pool has any-gold recall 0.870130 and all-gold recall 0.800866 at 10; additional context can recover evidence that shallower retrieval misses.                                                  |
| B                       | Do query/passage representations evolve across layers?                                                                           | HISTORICAL / SUPERSEDED | The legacy export exists, but it is deterministic simulation output, not evidence about real model representations.                                                                                           |
| C                       | Does retrieval relevance agree with attention or passage-removal utility?                                                        | HISTORICAL / SUPERSEDED | The deterministic simulation is reproducible. Its 0.9928 ablation/gold Pearson correlation is an artifact of the construction.                                                                                |
| GPU-PILOT               | Do real forward passes exhibit the same attention/ablation behavior as the generic simulated exports?                            | COMPLETED               | 96/100 removal deltas are negative, with mean −0.823364. This differs fundamentally from the generic nonnegative mock ablation export.                                                                        |
| INTERVENTION-VALIDATION | Could removing a passage change prompt structure/length rather than isolate its information content?                             | COMPLETED               | Both pilot interventions mostly have negative deltas; replacement is approximately, not perfectly, length matched.                                                                                            |
| FINAL40                 | Do annotated-evidence categories differ in real replacement sensitivity across a broader, deliberately heterogeneous dev sample? | COMPLETED               | STRONG-match passages have mean replacement delta 0.618710 versus about 0.02 for WEAK/NONE. Overall replacement mean is +0.098325, but 146/400 deltas are negative and 19 are zero. Utility is heterogeneous. |
| RETROSPECTIVE-BRIDGE    | Does the final40 sensitivity profile relate retrospectively to best-tested context budgets?                                      | COMPLETED               | A 40-row retrospective join exists and matches its Week3 sources; mean selected oracle F1 is 0.547600 on this subset.                                                                                         |

Source: per-part primary paths in the source list below.

## Key results

|         k |   any_gold_recall |   all_gold_recall |    queries |
|----------:|------------------:|------------------:|-----------:|
|  2.000000 |          0.515152 |          0.372294 | 231.000000 |
|  4.000000 |          0.701299 |          0.558442 | 231.000000 |
|  6.000000 |          0.809524 |          0.705628 | 231.000000 |
| 10.000000 |          0.870130 |          0.800866 | 231.000000 |

Source: [results/week0/dataset_analysis/retrieval_statistics.csv](../../week0/dataset_analysis/retrieval_statistics.csv).

| study               |   rows |   queries |   mean_delta |   median_delta |   positive |   negative |   zero |
|:--------------------|-------:|----------:|-------------:|---------------:|-----------:|-----------:|-------:|
| pilot removal       |    100 |        10 |    -0.823364 |      -0.386718 |          4 |         96 |      0 |
| pilot replacement   |    100 |        10 |    -0.820215 |      -0.375000 |          6 |         94 |      0 |
| final40 removal     |    400 |        40 |     0.089741 |       0.003906 |        217 |        170 |     13 |
| final40 replacement |    400 |        40 |     0.098325 |       0.007812 |        235 |        146 |     19 |

Sources: pilot/removal/replacement and final40 intervention CSVs.

| match_category   |   n |   mean_replacement_delta |   median_replacement_delta |   positive_fraction |
|:-----------------|----:|-------------------------:|---------------------------:|--------------------:|
| NONE             | 108 |                 0.018283 |                   0.007812 |            0.629630 |
| PARTIAL          |  47 |                 0.031660 |                   0.007812 |            0.617021 |
| STRONG           |  51 |                 0.618710 |                   0.125000 |            0.784314 |
| WEAK             | 194 |                 0.022233 |                   0.000977 |            0.505155 |

Source: [results/week0/final40/replacement_results.csv](../../week0/final40/replacement_results.csv).

## Main scientific conclusions

The local split inventory and retrieval recall are reproducible. Real sensitivity studies show substantial heterogeneity, not uniform evidence use. Final40 strong-match passages have higher teacher-forced replacement impact, but that is not generated answer F1 or a latent-versus-text result. Generic layer/utilization tables are not empirical: every tested value matches seeded mock formulas.

Sources: [results/weekly_reports/week_00/PrimaryRecomputationChecks.json](PrimaryRecomputationChecks.json), [results/week0/final40/execution_metadata.json](../../week0/final40/execution_metadata.json), [results/week0/final40/replacement_results.csv](../../week0/final40/replacement_results.csv).

## What did NOT work?

The mock export cannot support retrieval/attention/utility correlations as a model discovery. Most pilot intervention deltas are negative. The larger study also contains negative and zero effects. No deployable budget controller or generation-quality superiority was evaluated.

## Limitations

Forty selected dev papers at most; known gold-answer teacher forcing; approximate evidence matching and replacement length; nested passage observations; pilot/final40 sample differences. The paper bootstrap interval is a newly reconstructed exploratory calculation, not a surviving original outcome file. No notebook executed outputs were found.

## Decision after this week

Current runner code avoids writing mocks to empirical paths and marks generic GPU work pending; separate pilot/final40 execution exists. An original frozen method-selection decision is not established from available primary artifacts. Exclude generic mock exports from empirical claims. Preserve all source evidence unchanged.

## One-paragraph week takeaway

Week0 supplies reproducible local QASPER split/retrieval descriptions and separate real teacher-forced sensitivity studies. Its generic model tables exactly reproduce mock generators and must be quarantined. Real final40 data associate strong annotation matching with greater positive replacement sensitivity, while preserving negative/null interventions and sampling limitations. The later joined Week3 oracle is retrospective. **EVIDENCE GAP — manual review required** before continuing the sequential reconstruction past this provenance conflict.

# Primary Sources Used

- [configs/week0/week0.yaml](../../../configs/week0/week0.yaml)
- [notebooks/week0/week0_analysis.ipynb](../../../notebooks/week0/week0_analysis.ipynb)
- [results/week0/dataset_analysis/dataset_summary.csv](../../week0/dataset_analysis/dataset_summary.csv)
- [results/week0/dataset_analysis/evidence_statistics.csv](../../week0/dataset_analysis/evidence_statistics.csv)
- [results/week0/dataset_analysis/paper_level_statistics.csv](../../week0/dataset_analysis/paper_level_statistics.csv)
- [results/week0/dataset_analysis/question_level_statistics.csv](../../week0/dataset_analysis/question_level_statistics.csv)
- [results/week0/dataset_analysis/retrieval_statistics.csv](../../week0/dataset_analysis/retrieval_statistics.csv)
- [results/week0/evidence_analysis/ablation_passage_scores.csv](../../week0/evidence_analysis/ablation_passage_scores.csv)
- [results/week0/evidence_analysis/attention_passage_scores.csv](../../week0/evidence_analysis/attention_passage_scores.csv)
- [results/week0/evidence_analysis/signal_correlations.csv](../../week0/evidence_analysis/signal_correlations.csv)
- [results/week0/figures/retrieval_diagnostics.png](../../week0/figures/retrieval_diagnostics.png)
- [results/week0/final40/attention_results.csv](../../week0/final40/attention_results.csv)
- [results/week0/final40/evidence_match_results.csv](../../week0/final40/evidence_match_results.csv)
- [results/week0/final40/execution_metadata.json](../../week0/final40/execution_metadata.json)
- [results/week0/final40/figures/delta_replace_by_evidence_category.png](../../week0/final40/figures/delta_replace_by_evidence_category.png)
- [results/week0/final40/figures/per_query_intervention_heterogeneity.png](../../week0/final40/figures/per_query_intervention_heterogeneity.png)
- [results/week0/final40/figures/week0_vs_week3_oracle.png](../../week0/final40/figures/week0_vs_week3_oracle.png)
- [results/week0/final40/layer_results.csv](../../week0/final40/layer_results.csv)
- [results/week0/final40/question_level_features.csv](../../week0/final40/question_level_features.csv)
- [results/week0/final40/removal_results.csv](../../week0/final40/removal_results.csv)
- [results/week0/final40/replacement_results.csv](../../week0/final40/replacement_results.csv)
- [results/week0/final40/sample_manifest.csv](../../week0/final40/sample_manifest.csv)
- [results/week0/final40/week3_bridge.csv](../../week0/final40/week3_bridge.csv)
- [results/week0/gpu_pilot/ablation_results.csv](../../week0/gpu_pilot/ablation_results.csv)
- [results/week0/gpu_pilot/attention_results.csv](../../week0/gpu_pilot/attention_results.csv)
- [results/week0/gpu_pilot/execution_metadata.json](../../week0/gpu_pilot/execution_metadata.json)
- [results/week0/gpu_pilot/figures/attention_vs_intervention.png](../../week0/gpu_pilot/figures/attention_vs_intervention.png)
- [results/week0/gpu_pilot/intervention_validation/evidence_match_audit.csv](../../week0/gpu_pilot/intervention_validation/evidence_match_audit.csv)
- [results/week0/gpu_pilot/intervention_validation/figures/removal_vs_replacement_scatter.png](../../week0/gpu_pilot/intervention_validation/figures/removal_vs_replacement_scatter.png)
- [results/week0/gpu_pilot/intervention_validation/removal_vs_replacement.csv](../../week0/gpu_pilot/intervention_validation/removal_vs_replacement.csv)
- [results/week0/gpu_pilot/intervention_validation/replacement_results.csv](../../week0/gpu_pilot/intervention_validation/replacement_results.csv)
- [results/week0/gpu_pilot/layer_results.csv](../../week0/gpu_pilot/layer_results.csv)
- [results/week0/gpu_pilot/pilot_questions.csv](../../week0/gpu_pilot/pilot_questions.csv)
- [results/week0/gpu_status.json](../../week0/gpu_status.json)
- [results/week0/layer_analysis/layer_similarity_long.csv](../../week0/layer_analysis/layer_similarity_long.csv)
- [results/week3/processed/dev_quality_oracle.csv](../../week3/processed/dev_quality_oracle.csv)
- [results/week3/processed/epsilon_oracle_all.csv](../../week3/processed/epsilon_oracle_all.csv)
- [results/weekly_reports/week_00/OfflineBootstrapReconstruction.json](OfflineBootstrapReconstruction.json)
- [results/weekly_reports/week_00/PrimaryRecomputationChecks.json](PrimaryRecomputationChecks.json)
- [scripts/week0/generate_final40_figures_and_report.py](../../../scripts/week0/generate_final40_figures_and_report.py)
- [scripts/week0/run_final40_study.py](../../../scripts/week0/run_final40_study.py)
- [scripts/week0/run_gpu_pilot.py](../../../scripts/week0/run_gpu_pilot.py)
- [scripts/week0/run_intervention_validation.py](../../../scripts/week0/run_intervention_validation.py)
- [scripts/week0/run_week0_analysis.py](../../../scripts/week0/run_week0_analysis.py)
- [scripts/week0/select_final40_sample.py](../../../scripts/week0/select_final40_sample.py)
- [scripts/weekly_reports/check_week0_primary.py](../../../scripts/weekly_reports/check_week0_primary.py)
- [src/week0/dataset_adapter.py](../../../src/week0/dataset_adapter.py)
- [src/week0/evidence_utilization.py](../../../src/week0/evidence_utilization.py)
- [src/week0/layer_analyzer.py](../../../src/week0/layer_analyzer.py)

# Existing Markdown Used as Evidence

NONE
