"""Week 7-C2: Utility-Aware Budget Ranking and Continuation Policy Package."""

from src.week7.w7c2.data import (
    ALLOWED_BUDGETS,
    assert_test_disabled,
    build_w7c2_ranking_dataset,
    load_w7c2_splits,
)
from src.week7.w7c2.diagnostics import (
    compute_adjacent_marginal_gains,
    compute_admissible_set_distribution,
    compute_best_vs_second_margin,
    compute_monotonicity_analysis,
)
from src.week7.w7c2.targets import (
    compare_budgets_lexicographic,
    compute_continuation_target,
    compute_lexicographic_relevance_grades,
)
from src.week7.w7c2.regression import (
    DirectF1Regressor,
)
from src.week7.w7c2.ranking import (
    QueryGroupedBudgetRanker,
)
from src.week7.w7c2.continuation import (
    StagewiseContinuationPolicy,
)
from src.week7.w7c2.policy import (
    apply_m1_regression_decision_rule,
    apply_m2_ranking_decision_rule,
    evaluate_downstream_policy,
    select_best_model_internal_val,
)
from src.week7.w7c2.evaluation import (
    compute_corrected_error_taxonomy,
    compute_recovered_oracle_headroom,
    run_paper_clustered_bootstrap,
)
from src.week7.w7c2.importance import (
    assign_feature_family,
    compute_grouped_permutation_importance,
)

__all__ = [
    "ALLOWED_BUDGETS",
    "assert_test_disabled",
    "build_w7c2_ranking_dataset",
    "load_w7c2_splits",
    "compute_adjacent_marginal_gains",
    "compute_admissible_set_distribution",
    "compute_best_vs_second_margin",
    "compute_monotonicity_analysis",
    "compare_budgets_lexicographic",
    "compute_continuation_target",
    "compute_lexicographic_relevance_grades",
    "DirectF1Regressor",
    "QueryGroupedBudgetRanker",
    "StagewiseContinuationPolicy",
    "apply_m1_regression_decision_rule",
    "apply_m2_ranking_decision_rule",
    "evaluate_downstream_policy",
    "select_best_model_internal_val",
    "compute_corrected_error_taxonomy",
    "compute_recovered_oracle_headroom",
    "run_paper_clustered_bootstrap",
    "assign_feature_family",
    "compute_grouped_permutation_importance",
]
