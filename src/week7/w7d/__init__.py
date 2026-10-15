"""Week 7-D Generator-Aligned Cross-Encoder Fidelity Router package."""

from src.week7.w7d.bootstrap import run_paper_clustered_bootstrap
from src.week7.w7d.cross_encoder import CrossEncoderScorer
from src.week7.w7d.cross_encoder_features import (
    build_d0_feature_dataframe,
    extract_query_cross_encoder_features,
)
from src.week7.w7d.data import (
    ALLOWED_BUDGETS,
    ANCHOR_K,
    assert_test_disabled,
    assert_zero_feature_leakage,
    audit_input_lengths,
    load_retrieval_passages,
    load_w7d_splits,
    truncate_document_deterministic,
)
from src.week7.w7d.evaluation import (
    compute_downshift_safety_metrics,
    compute_recovered_oracle_headroom,
    compute_upshift_precision_metrics,
    compute_w7d_error_taxonomy,
)
from src.week7.w7d.hierarchical_model import HierarchicalCrossEncoderRouter
from src.week7.w7d.listwise_llm import ListwiseLLMRouter, format_listwise_causal_prompt
from src.week7.w7d.lora_utils import audit_d2_hardware_feasibility, get_peft_lora_config
from src.week7.w7d.multitask_heads import MultiTaskUtilityHeads
from src.week7.w7d.policy import (
    apply_w7d_anchor_deviate_policy,
    apply_w7d_multilabel_policy,
    evaluate_downstream_fidelity_policy,
    select_best_fidelity_policy,
)
from src.week7.w7d.set_aggregator import SetLevelAggregator
from src.week7.w7d.targets import (
    build_w7d_targets_dataframe,
    compute_sara_multitask_targets,
)
from src.week7.w7d.training import (
    CachedEmbeddingDataset,
    D0TabularRouter,
    HierarchicalTokenDataset,
    ListwisePromptDataset,
    train_d1_aggregator,
    train_d1b_end_to_end,
    train_d2_listwise_router,
)

__all__ = [
    "ALLOWED_BUDGETS",
    "ANCHOR_K",
    "assert_test_disabled",
    "assert_zero_feature_leakage",
    "load_w7d_splits",
    "load_retrieval_passages",
    "audit_input_lengths",
    "truncate_document_deterministic",
    "compute_sara_multitask_targets",
    "build_w7d_targets_dataframe",
    "CrossEncoderScorer",
    "extract_query_cross_encoder_features",
    "build_d0_feature_dataframe",
    "SetLevelAggregator",
    "MultiTaskUtilityHeads",
    "HierarchicalCrossEncoderRouter",
    "ListwiseLLMRouter",
    "format_listwise_causal_prompt",
    "audit_d2_hardware_feasibility",
    "get_peft_lora_config",
    "apply_w7d_anchor_deviate_policy",
    "apply_w7d_multilabel_policy",
    "evaluate_downstream_fidelity_policy",
    "select_best_fidelity_policy",
    "D0TabularRouter",
    "CachedEmbeddingDataset",
    "HierarchicalTokenDataset",
    "ListwisePromptDataset",
    "train_d1_aggregator",
    "train_d1b_end_to_end",
    "train_d2_listwise_router",
    "compute_w7d_error_taxonomy",
    "compute_downshift_safety_metrics",
    "compute_upshift_precision_metrics",
    "compute_recovered_oracle_headroom",
    "run_paper_clustered_bootstrap",
]

