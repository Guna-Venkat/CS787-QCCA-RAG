# Week 11 phase I

Saved-result characterization only; no winner or deployable oracle is declared.

Scores and statistical comparisons are dataset-specific. No mixed raw QA/verification overall F1 is computed.

## Tables

- `master_summary_df.csv`: 506 rows
- `best_by_dataset_df.csv`: 3 rows
- `best_by_reader_df.csv`: 10 rows
- `best_by_budget_df.csv`: 15 rows
- `best_by_granularity_df.csv`: 15 rows
- `pareto_front_df.csv`: 504 rows
- `statistical_comparisons_df.csv`: 1654 rows
- `specialist_baseline_df.csv`: 2 rows
- `portable_reader_summary_df.csv`: 504 rows
- `answer_type_breakdown_df.csv`: 1350 rows
- `oracle_summary_df.csv`: 13 rows
- `per_query_oracle_df.csv`: 650 rows
- `oracle_action_distribution_df.csv`: 213 rows
- `oracle_gain_concentration_df.csv`: 52 rows
- `oracle_gain_curve_df.csv`: 650 rows
- `oracle_status_df.csv`: 0 rows
- `oracle_capture_df.csv`: 96 rows
- `retrieval_saved_df.csv`: 11700 rows
- `project_trajectory_df.csv`: 0 rows
- `main_granularity_summary_df.csv`: 111 rows
- `matched_reader_summary_df.csv`: 444 rows
- `matched_compressor_summary_df.csv`: 81 rows
- `finalist_retriever_summary_df.csv`: 48 rows
- `hierarchy_fixed_comparison_df.csv`: 75 rows
- `cross_dataset_action_ranks_df.csv`: 506 rows

## Figures

- fig01_retrieval: `HotpotQA_PASSAGE_any_gold_recall_fig01_retrieval.png` (PNG, PDF, and source CSV)
- fig01_retrieval: `HotpotQA_PROPOSITION_any_gold_recall_fig01_retrieval.png` (PNG, PDF, and source CSV)
- fig01_retrieval: `HotpotQA_SENTENCE_any_gold_recall_fig01_retrieval.png` (PNG, PDF, and source CSV)
- fig01_retrieval: `QASPER_PASSAGE_any_gold_recall_fig01_retrieval.png` (PNG, PDF, and source CSV)
- fig01_retrieval: `QASPER_PROPOSITION_any_gold_recall_fig01_retrieval.png` (PNG, PDF, and source CSV)
- fig01_retrieval: `QASPER_SENTENCE_any_gold_recall_fig01_retrieval.png` (PNG, PDF, and source CSV)
- fig01_retrieval: `SciFact_PASSAGE_any_gold_recall_fig01_retrieval.png` (PNG, PDF, and source CSV)
- fig01_retrieval: `SciFact_PROPOSITION_any_gold_recall_fig01_retrieval.png` (PNG, PDF, and source CSV)
- fig01_retrieval: `SciFact_SENTENCE_any_gold_recall_fig01_retrieval.png` (PNG, PDF, and source CSV)
- fig02_budget: `HotpotQA_task_score_fig02_budget.png` (PNG, PDF, and source CSV)
- fig02_budget: `QASPER_task_score_fig02_budget.png` (PNG, PDF, and source CSV)
- fig02_budget: `SciFact_task_score_fig02_budget.png` (PNG, PDF, and source CSV)
- fig03_reader_granularity: `HotpotQA_task_score_fig03_reader_granularity.png` (PNG, PDF, and source CSV)
- fig03_reader_granularity: `QASPER_task_score_fig03_reader_granularity.png` (PNG, PDF, and source CSV)
- fig03_reader_granularity: `SciFact_task_score_fig03_reader_granularity.png` (PNG, PDF, and source CSV)
- fig04_dataset_granularity_rank: `cross_dataset_Within-dataset_rank_lower_is_better__fig04_dataset_granularity_rank.png` (PNG, PDF, and source CSV)
- fig05_pareto: `HotpotQA_task_score_fig05_pareto.png` (PNG, PDF, and source CSV)
- fig05_pareto: `QASPER_task_score_fig05_pareto.png` (PNG, PDF, and source CSV)
- fig05_pareto: `SciFact_task_score_fig05_pareto.png` (PNG, PDF, and source CSV)
- fig06_help_hurt: `HotpotQA_HELP_fig06_help_hurt.png` (PNG, PDF, and source CSV)
- fig06_help_hurt: `QASPER_HELP_fig06_help_hurt.png` (PNG, PDF, and source CSV)
- fig06_help_hurt: `SciFact_HELP_fig06_help_hurt.png` (PNG, PDF, and source CSV)
- fig07_retriever_granularity: `HotpotQA_task_score_fig07_retriever_granularity.png` (PNG, PDF, and source CSV)
- fig07_retriever_granularity: `QASPER_task_score_fig07_retriever_granularity.png` (PNG, PDF, and source CSV)
- fig07_retriever_granularity: `SciFact_task_score_fig07_retriever_granularity.png` (PNG, PDF, and source CSV)
- fig08_compressor_reader: `HotpotQA_task_score_fig08_compressor_reader.png` (PNG, PDF, and source CSV)
- fig08_compressor_reader: `QASPER_task_score_fig08_compressor_reader.png` (PNG, PDF, and source CSV)
- fig08_compressor_reader: `SciFact_task_score_fig08_compressor_reader.png` (PNG, PDF, and source CSV)
- fig09_oracle_actions: `HotpotQA_count_fig09_oracle_actions.png` (PNG, PDF, and source CSV)
- fig09_oracle_actions: `QASPER_count_fig09_oracle_actions.png` (PNG, PDF, and source CSV)
- fig09_oracle_actions: `SciFact_count_fig09_oracle_actions.png` (PNG, PDF, and source CSV)
- fig10_oracle_concentration: `HotpotQA_cumulative_gain_fraction_fig10_oracle_concentration.png` (PNG, PDF, and source CSV)
- fig10_oracle_concentration: `QASPER_cumulative_gain_fraction_fig10_oracle_concentration.png` (PNG, PDF, and source CSV)
- fig10_oracle_concentration: `SciFact_cumulative_gain_fraction_fig10_oracle_concentration.png` (PNG, PDF, and source CSV)
- fig11_hierarchy_fixed: `HotpotQA_task_score_fig11_hierarchy_fixed.png` (PNG, PDF, and source CSV)
- fig11_hierarchy_fixed: `QASPER_task_score_fig11_hierarchy_fixed.png` (PNG, PDF, and source CSV)
- fig11_hierarchy_fixed: `SciFact_task_score_fig11_hierarchy_fixed.png` (PNG, PDF, and source CSV)
