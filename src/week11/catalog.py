"""Single notebook/table/figure inventory, shared by notebooks and reporting."""
NOTEBOOKS={
 '0':'week11_0_setup_and_data.ipynb','A':'week11_A_retrieval_granularity.ipynb',
 'B':'week11_B_passage_representation.ipynb','C':'week11_C_granularity_budget.ipynb',
 'D':'week11_D_reader_granularity.ipynb','E':'week11_E_compressor_reader.ipynb',
 'F':'week11_F_retriever_robustness.ipynb','G':'week11_G_hierarchical_fidelity.ipynb',
 'H':'week11_H_oracle_and_interactions.ipynb','I':'week11_I_final_report.ipynb'}
TABLES={
 '0':'dataset_summary_df answer_type_df document_length_df hierarchy_stats_df model_registry_df cache_status_df',
 'A':'retrieval_summary_df recall_at_k_df evidence_coverage_df retrieval_cost_df retrieval_pairwise_df',
 'B':'passage_representation_summary_df ranking_agreement_df recall_comparison_df topk_overlap_df per_query_representation_df',
 'C':'c_summary_df c_quality_df c_efficiency_df c_help_hurt_df c_pareto_df c_pairwise_bootstrap_df c_per_query_df',
 'D':'reader_granularity_summary_df reader_quality_df reader_help_hurt_df reader_efficiency_df reader_pairwise_df reader_interaction_df reader_per_query_df',
 'E':'compressor_reader_summary_df compressor_quality_df compressor_efficiency_df compressor_pairwise_df compressor_interaction_df',
 'F':'retriever_robustness_summary_df retriever_interaction_df retriever_pairwise_df',
 'G':'hierarchy_summary_df hierarchy_budget_allocation_df hierarchy_quality_df hierarchy_help_hurt_df hierarchy_pairwise_df hierarchy_oracle_df',
 'H':'oracle_summary_df oracle_action_distribution_df oracle_gain_concentration_df per_query_oracle_df interaction_summary_df factorial_effects_df',
 'I':'master_summary_df best_by_dataset_df best_by_reader_df best_by_budget_df best_by_granularity_df pareto_front_df statistical_comparisons_df'}
TABLES={k:v.split() for k,v in TABLES.items()}

# name, source table, visual, x, y, group. Every saved figure has a saved source CSV.
PLOTS={
'0':[
 ('document_length','document_length_df','hist','tokens',None,None),
 ('answer_types','answer_type_df','bar','answer_type','count',None),
 ('unit_lengths','hierarchy_stats_df','hist','token_count',None,'unit_type'),
 ('units_per_document','hierarchy_stats_df','bar','unit_type','units_per_document',None)],
'A':[
 ('recall_curves','recall_at_k_df','line','k','any_gold_recall','retriever'),
 ('any_gold','retrieval_summary_df','bar','evidence_granularity','any_gold_recall','retriever'),
 ('coverage_tokens','evidence_coverage_df','scatter','retrieved_tokens','evidence_coverage','retriever'),
 ('retriever_granularity','retrieval_summary_df','heatmap','retriever','any_gold_recall','evidence_granularity'),
 ('first_rank','first_rank_df','hist','first_relevant_rank',None,'retriever'),
 ('redundancy','retrieval_summary_df','bar','evidence_granularity','redundancy','retriever')],
'B':[
 ('pooling_recall','recall_comparison_df','bar','passage_representation','any_gold_recall',None),
 ('pooling_mrr','passage_representation_summary_df','bar','passage_representation','mrr',None),
 ('pooling_overlap','topk_overlap_df','heatmap','representation_a','overlap','representation_b'),
 ('weighted_minus_direct','weighted_improvement_df','hist','delta_vs_direct',None,None),
 ('length_pooling_gain','per_query_representation_df','scatter','mean_passage_tokens','delta_vs_direct','passage_representation')],
'C':[
 ('quality_budget','c_summary_df','line','budget','task_score','evidence_granularity'),
 ('quality_evidence','c_summary_df','scatter','actual_evidence_tokens','task_score','evidence_granularity'),
 ('quality_prompt','c_summary_df','scatter','prompt_tokens','task_score','evidence_granularity'),
 ('granularity_budget','c_summary_df','heatmap','evidence_granularity','task_score','budget'),
 ('granularity_selector','c_summary_df','heatmap','evidence_granularity','task_score','selector'),
 ('help_hurt','c_help_hurt_df','help_hurt','evidence_granularity','HELP','HURT'),
 ('pareto','c_pareto_df','scatter','prompt_tokens','task_score','evidence_granularity'),
 ('quality_per_1000','c_efficiency_df','bar','evidence_granularity','quality_per_1000_tokens',None),
 ('actual_requested','c_efficiency_df','scatter','requested_budget_numeric','actual_evidence_tokens','evidence_granularity'),
 ('dataset_quality','c_quality_df','line','budget','task_score','selector')],
'D':[
 ('reader_granularity','reader_quality_df','heatmap','reader','task_score','evidence_granularity'),
 ('reader_budget','reader_quality_df','line','budget','task_score','reader'),
 ('granularity_budget','reader_quality_df','line','budget','task_score','evidence_granularity'),
 ('reader_help_hurt','reader_help_hurt_df','help_hurt','reader','HELP','HURT'),
 ('reader_efficiency','reader_efficiency_df','scatter','prompt_tokens','task_score','reader'),
 ('reader_ranking','reader_quality_df','bar','reader','task_score',None),
 ('reader_dataset_rank','reader_quality_df','rank_heatmap','reader','task_score','dataset'),
 ('difference_from_mistral','reader_quality_df','difference_heatmap','reader','task_score','evidence_granularity')],
'E':[
 ('compressor_reader','compressor_quality_df','heatmap','compressor','task_score','reader'),
 ('compression_quality','compressor_efficiency_df','scatter','compression_ratio','task_score','reader'),
 ('compression_prompt','compressor_efficiency_df','scatter','prompt_tokens','task_score','compressor'),
 ('compression_hurt','compressor_efficiency_df','scatter','compression_ratio','HURT','reader'),
 ('reader_compressor_rank','compressor_quality_df','bar','compressor','task_score','reader'),
 ('compressor_pareto','compressor_efficiency_df','scatter','actual_evidence_tokens','task_score','compressor')],
'F':[
 ('retriever_finalist','retriever_robustness_summary_df','heatmap','retriever','task_score','action_id'),
 ('recall_downstream','retriever_robustness_summary_df','scatter','evidence_coverage','task_score','retriever'),
 ('retriever_pareto','retriever_robustness_summary_df','scatter','prompt_tokens','task_score','retriever'),
 ('relative_bm25','retriever_robustness_summary_df','difference_heatmap','retriever','task_score','evidence_granularity')],
'G':[
 ('hierarchy_fixed','hierarchy_fixed_comparison_df','bar','strategy','task_score',None),
 ('hierarchy_allocation','hierarchy_budget_allocation_df','bar','unit_type','tokens','strategy'),
 ('hierarchy_context','hierarchy_quality_df','scatter','actual_evidence_tokens','task_score','strategy'),
 ('hierarchy_help_hurt','hierarchy_help_hurt_df','help_hurt','strategy','HELP','HURT'),
 ('hierarchy_pareto','hierarchy_quality_df','scatter','prompt_tokens','task_score','strategy'),
 ('hierarchy_dataset','hierarchy_quality_df','rank_heatmap','strategy','task_score','dataset')],
'H':[
 ('oracle_fixed','oracle_summary_df','bar','reader','exact_oracle',None),
 ('oracle_actions','oracle_action_distribution_df','bar','action','count',None),
 ('gain_concentration','oracle_gain_curve_df','line','fraction','cumulative_gain_fraction','reader'),
 ('dataset_oracle','per_query_oracle_df','count_heatmap','dataset',None,'granularity'),
 ('reader_oracle','per_query_oracle_df','count_heatmap','reader',None,'granularity'),
 ('dataset_granularity','per_query_oracle_df','count_heatmap','dataset',None,'granularity'),
 ('reader_granularity','per_query_oracle_df','count_heatmap','reader',None,'granularity'),
 ('factorial_effects','factorial_effects_df','bar','level','task_score','axis')],
'I':[
 ('fig01_retrieval','retrieval_saved_df','line','k','any_gold_recall','retriever'),
 ('fig02_budget','main_granularity_summary_df','line','budget','task_score','evidence_granularity'),
 ('fig03_reader_granularity','matched_reader_summary_df','heatmap','reader','task_score','evidence_granularity'),
 ('fig04_dataset_granularity_rank','main_granularity_summary_df','rank_heatmap','dataset','task_score','evidence_granularity'),
 ('fig05_pareto','pareto_front_df','scatter','prompt_tokens','task_score','evidence_granularity'),
 ('fig06_help_hurt','main_granularity_summary_df','help_hurt','evidence_granularity','HELP','HURT'),
 ('fig07_retriever_granularity','finalist_retriever_summary_df','heatmap','retriever','task_score','evidence_granularity'),
 ('fig08_compressor_reader','matched_compressor_summary_df','heatmap','compressor','task_score','reader'),
 ('fig09_oracle_actions','oracle_action_distribution_df','bar','action','count',None),
 ('fig10_oracle_concentration','oracle_gain_curve_df','line','fraction','cumulative_gain_fraction','reader'),
 ('fig11_hierarchy_fixed','hierarchy_fixed_comparison_df','bar','strategy','task_score',None),
 ('fig12_project_trajectory','project_trajectory_df','trajectory','week','value','protocol')]
}
