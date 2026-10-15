"""Week 7-C3 Anchor-and-Deviate Semantic Budget Policy package."""

from src.week7.w7c3.data import (
    ALLOWED_BUDGETS,
    ANCHOR_K,
    assert_test_disabled,
    assert_zero_feature_leakage,
    build_w7c3_query_dataset,
    get_feature_subsets,
    load_w7c3_splits,
)
from src.week7.w7c3.delta_models import MarginalGainDeltaRegressors
from src.week7.w7c3.diagnostics import compute_4to6_gain_analysis, compute_anchor_oracle_diagnostics
from src.week7.w7c3.downshift import SafeDownshiftClassifier
from src.week7.w7c3.evaluation import (
    compute_downshift_safety_metrics,
    compute_recovered_oracle_headroom,
    compute_upshift_precision_metrics,
    compute_w7c3_error_taxonomy,
    run_paper_clustered_bootstrap,
)
from src.week7.w7c3.oof import evaluate_oof_policy_grid, run_group_kfold_oof
from src.week7.w7c3.policy import (
    apply_anchor_deviate_decision_rule,
    evaluate_downstream_anchor_policy,
    select_best_anchor_policy,
)
from src.week7.w7c3.targets import (
    compute_4to6_gain_target,
    compute_delta_regression_targets,
    compute_minimum_cost_oracle_k,
    compute_oracle_admissible_set,
    compute_safe_downshift_target,
    compute_selective_upshift_target,
)

__all__ = [
    "ALLOWED_BUDGETS",
    "ANCHOR_K",
    "assert_test_disabled",
    "assert_zero_feature_leakage",
    "build_w7c3_query_dataset",
    "get_feature_subsets",
    "load_w7c3_splits",
    "SafeDownshiftClassifier",
    "SelectiveUpshiftClassifier",
    "MarginalGainDeltaRegressors",
    "compute_anchor_oracle_diagnostics",
    "compute_4to6_gain_analysis",
    "compute_w7c3_error_taxonomy",
    "compute_downshift_safety_metrics",
    "compute_upshift_precision_metrics",
    "compute_recovered_oracle_headroom",
    "run_paper_clustered_bootstrap",
    "run_group_kfold_oof",
    "evaluate_oof_policy_grid",
    "apply_anchor_deviate_decision_rule",
    "evaluate_downstream_anchor_policy",
    "select_best_anchor_policy",
    "compute_oracle_admissible_set",
    "compute_minimum_cost_oracle_k",
    "compute_safe_downshift_target",
    "compute_selective_upshift_target",
    "compute_delta_regression_targets",
    "compute_4to6_gain_target",
]
