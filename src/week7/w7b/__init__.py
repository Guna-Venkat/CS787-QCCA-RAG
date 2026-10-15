"""Week 7-B: Semantic-Routing-Aware Budget Response Surface and Adaptive Allocation package."""

from src.week7.w7b.data import (
    ALLOWED_BUDGETS,
    assert_test_disabled,
    load_retrieval_records,
    load_w7b_dataset_bundle,
    load_w7b_fit_val_splits,
)
from src.week7.w7b.response_surface import (
    build_semantic_routing_assignments,
    execute_resumable_semantic_response_surface,
    load_existing_response_surface_cache,
    build_semantic_fixed_k_matrix,
)
from src.week7.w7b.oracle import (
    compute_semantic_oracle_dataframe,
    summarize_semantic_oracle,
    compute_bm25_vs_semantic_oracle_shift,
    compute_recovered_oracle_headroom,
)
from src.week7.w7b.features import (
    STAGE_2_CORE_FEATURES,
    extract_single_query_w4_features,
    extract_features_for_bundle,
    standardize_features_train_fit_only,
)
from src.week7.w7b.training import (
    train_candidate_budget_models,
    predict_budgets,
    select_best_model_on_internal_val,
    save_frozen_w7b_model,
)
from src.week7.w7b.evaluation import (
    evaluate_policy_by_matrix_lookup,
    build_w7b_master_results_table,
    compute_conditional_quality_analysis,
)
from src.week7.w7b.bootstrap import (
    run_paper_clustered_bootstrap,
    run_all_w7b_bootstrap_comparisons,
)

__all__ = [
    "ALLOWED_BUDGETS",
    "assert_test_disabled",
    "load_retrieval_records",
    "load_w7b_dataset_bundle",
    "load_w7b_fit_val_splits",
    "build_semantic_routing_assignments",
    "execute_resumable_semantic_response_surface",
    "load_existing_response_surface_cache",
    "build_semantic_fixed_k_matrix",
    "compute_semantic_oracle_dataframe",
    "summarize_semantic_oracle",
    "compute_bm25_vs_semantic_oracle_shift",
    "compute_recovered_oracle_headroom",
    "STAGE_2_CORE_FEATURES",
    "extract_single_query_w4_features",
    "extract_features_for_bundle",
    "standardize_features_train_fit_only",
    "train_candidate_budget_models",
    "predict_budgets",
    "select_best_model_on_internal_val",
    "save_frozen_w7b_model",
    "evaluate_policy_by_matrix_lookup",
    "build_w7b_master_results_table",
    "compute_conditional_quality_analysis",
    "run_paper_clustered_bootstrap",
    "run_all_w7b_bootstrap_comparisons",
]
