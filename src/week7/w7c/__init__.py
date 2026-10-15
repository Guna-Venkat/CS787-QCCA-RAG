"""Week 7-C: Retrieval-Aware Semantic Sufficiency Allocation Package."""

from src.week7.w7c.data import (
    ALLOWED_BUDGETS,
    BUDGET_LIST,
    assert_test_disabled,
    build_candidate_budget_dataframe,
    load_retrieval_records,
    load_w7c_dataset_bundle,
    load_w7c_fit_val_splits,
)
from src.week7.w7c.features_query import (
    compute_single_query_features,
    extract_query_complexity_features,
)
from src.week7.w7c.features_retrieval import (
    compute_bm25_qpp_features,
    compute_cross_retriever_features,
    compute_semantic_budget_geometry,
    compute_semantic_pool_geometry,
)
from src.week7.w7c.features_set import (
    compute_set_diversity_features,
    compute_tail_threat_features,
)
from src.week7.w7c.features_coverage import (
    compute_cost_features,
    compute_query_coverage_features,
)
from src.week7.w7c.features_crossencoder import (
    OptionalFrozenCrossEncoderScorer,
    compute_cross_encoder_budget_features,
)
from src.week7.w7c.leakage import (
    assert_zero_feature_leakage,
    audit_feature_leakage,
    sanitize_feature_matrix,
)
from src.week7.w7c.models_tree import (
    RegretGBDTRegressor,
    SufficiencyGBDTClassifier,
)
from src.week7.w7c.models_mlp import (
    MultitaskMLPNetwork,
    MultitaskMLPWrapper,
)
from src.week7.w7c.models_set import (
    PassageSetTransformer,
    RetrievalSetEncoderWrapper,
)
from src.week7.w7c.generator_probe import (
    GeneratorSufficiencyProbe,
)
from src.week7.w7c.training import (
    ABLATION_FAMILIES,
    PreprocessingPipeline,
    build_candidate_feature_bank,
    get_ablation_feature_names,
)
from src.week7.w7c.policy import (
    apply_risk_aware_decision_rule,
    apply_sufficiency_decision_rule,
    calibrate_tau_on_internal_val,
    evaluate_policy_on_response_surface,
)
from src.week7.w7c.evaluation import (
    compute_recovered_oracle_headroom,
    compute_under_overallocation_breakdown,
    run_paper_clustered_bootstrap,
)

__all__ = [
    "ALLOWED_BUDGETS",
    "BUDGET_LIST",
    "assert_test_disabled",
    "build_candidate_budget_dataframe",
    "load_retrieval_records",
    "load_w7c_dataset_bundle",
    "load_w7c_fit_val_splits",
    "compute_single_query_features",
    "extract_query_complexity_features",
    "compute_bm25_qpp_features",
    "compute_cross_retriever_features",
    "compute_semantic_budget_geometry",
    "compute_semantic_pool_geometry",
    "compute_set_diversity_features",
    "compute_tail_threat_features",
    "compute_cost_features",
    "compute_query_coverage_features",
    "OptionalFrozenCrossEncoderScorer",
    "compute_cross_encoder_budget_features",
    "assert_zero_feature_leakage",
    "audit_feature_leakage",
    "sanitize_feature_matrix",
    "RegretGBDTRegressor",
    "SufficiencyGBDTClassifier",
    "MultitaskMLPNetwork",
    "MultitaskMLPWrapper",
    "PassageSetTransformer",
    "RetrievalSetEncoderWrapper",
    "GeneratorSufficiencyProbe",
    "ABLATION_FAMILIES",
    "PreprocessingPipeline",
    "build_candidate_feature_bank",
    "get_ablation_feature_names",
    "apply_risk_aware_decision_rule",
    "apply_sufficiency_decision_rule",
    "calibrate_tau_on_internal_val",
    "evaluate_policy_on_response_surface",
    "compute_recovered_oracle_headroom",
    "compute_under_overallocation_breakdown",
    "run_paper_clustered_bootstrap",
]
