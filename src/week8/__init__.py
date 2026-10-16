"""DAFiR: Draft-Aware Fidelity Routing for Compressed RAG (Week 8).

Main research package for:
- Counterfactual single-passage promotion surface (W8-A)
- Generator draft state and uncertainty diagnostics (W8-B)
- Draft-conditioned passage utility modeling (W8-C)
- Generator interaction and marginal information gain features (W8-D)
- Draft-aware selective fidelity controller (W8-E)
- Query-conditioned latent adapter (W8-F)
- Hard-case outcome-weighted generator adaptation (W8-G)
- Full hybrid DAFiR system (W8-H)
- Dual representation-compute frontier accounting (W8-I / W8-J)
"""

from src.week8.attribution import (
    compute_masking_attribution_fallback,
    extract_compressed_slot_attention_features,
)
from src.week8.bootstrap import (
    run_all_week8_bootstrap_comparisons,
    run_paper_clustered_bootstrap,
)
from src.week8.controller import DAFiRController
from src.week8.cost_accounting import GPUTracker, QueryCostRecord, summarize_cost_records
from src.week8.data import (
    assert_paper_disjoint,
    get_week8_manifest_path,
    load_week8_split,
)
from src.week8.draft_generation import (
    extract_draft_uncertainty_features,
    generate_draft_with_diagnostics,
)
from src.week8.evaluation import (
    analyze_promotion_rank_distribution,
    compute_safety_and_controller_metrics,
    stratify_by_query_type,
)
from src.week8.generation import run_dafir_query
from src.week8.hard_case_dataset import (
    HardCasePromptDataset,
    build_hard_case_dataset,
    categorize_hard_cases,
)
from src.week8.hard_case_lora import (
    evaluate_generator_lora,
    setup_lora_generator,
    train_generator_lora_epoch,
)
from src.week8.listwise_utility_model import (
    MistralListwiseUtilityRouter,
    format_listwise_input_prompt,
)
from src.week8.marginal_information import (
    compute_candidate_marginal_information,
    tune_mig_hyperparameters,
)
from src.week8.outcome_weighted_training import run_ow_lora_training
from src.week8.plotting import (
    plot_promotion_gain_distribution,
    plot_quality_efficiency_frontiers,
)
from src.week8.promotion_oracle import (
    calculate_greedy_oracle2,
    calculate_oracle1,
    calculate_oracle_selective1,
)
from src.week8.promotion_surface import (
    generate_resumable_promotion_surface,
    load_existing_promotion_cache,
)
from src.week8.qcla import (
    BaseLatentAdapter,
    QCLA0_Baseline,
    QCLA1_ResidualAdapter,
    QCLA2_DraftConditionedAdapter,
    build_qcla_adapter,
)
from src.week8.qcla_training import (
    QCLATrainingDataset,
    evaluate_qcla,
    train_qcla_epoch,
)
from src.week8.self_consistency import (
    generate_self_consistency_samples,
    measure_self_consistency_agreement,
)
from src.week8.uncertainty import (
    UncertaintyClassifier,
    compute_expected_calibration_error,
    train_uncertainty_sufficiency_models,
)
from src.week8.utility_dataset import (
    assemble_multitask_utility_dataset,
    extract_utility_targets,
)
from src.week8.utility_models import (
    FullHybridPassageUtilityModel_U4,
    PassageCrossEncoder_U1,
    TabularUtilityModel_U0,
)

__all__ = [
    "load_week8_split",
    "assert_paper_disjoint",
    "get_week8_manifest_path",
    "generate_resumable_promotion_surface",
    "load_existing_promotion_cache",
    "calculate_oracle1",
    "calculate_oracle_selective1",
    "calculate_greedy_oracle2",
    "generate_draft_with_diagnostics",
    "extract_draft_uncertainty_features",
    "UncertaintyClassifier",
    "compute_expected_calibration_error",
    "train_uncertainty_sufficiency_models",
    "generate_self_consistency_samples",
    "measure_self_consistency_agreement",
    "assemble_multitask_utility_dataset",
    "extract_utility_targets",
    "compute_candidate_marginal_information",
    "tune_mig_hyperparameters",
    "extract_compressed_slot_attention_features",
    "compute_masking_attribution_fallback",
    "TabularUtilityModel_U0",
    "PassageCrossEncoder_U1",
    "FullHybridPassageUtilityModel_U4",
    "MistralListwiseUtilityRouter",
    "format_listwise_input_prompt",
    "DAFiRController",
    "BaseLatentAdapter",
    "QCLA0_Baseline",
    "QCLA1_ResidualAdapter",
    "QCLA2_DraftConditionedAdapter",
    "build_qcla_adapter",
    "QCLATrainingDataset",
    "train_qcla_epoch",
    "evaluate_qcla",
    "categorize_hard_cases",
    "HardCasePromptDataset",
    "build_hard_case_dataset",
    "setup_lora_generator",
    "train_generator_lora_epoch",
    "evaluate_generator_lora",
    "run_ow_lora_training",
    "run_dafir_query",
    "QueryCostRecord",
    "summarize_cost_records",
    "GPUTracker",
    "compute_safety_and_controller_metrics",
    "analyze_promotion_rank_distribution",
    "stratify_by_query_type",
    "run_paper_clustered_bootstrap",
    "run_all_week8_bootstrap_comparisons",
    "plot_promotion_gain_distribution",
    "plot_quality_efficiency_frontiers",
]
