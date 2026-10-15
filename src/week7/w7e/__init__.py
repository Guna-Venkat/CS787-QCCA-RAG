"""Week 7-E Multi-Fidelity Evidence Representation package."""

from src.week7.w7e.analysis import (
    compute_stratified_4to6_analysis,
    run_multifidelity_paper_clustered_bootstrap,
)
from src.week7.w7e.evaluation import (
    compute_latency_comparison_table,
    compute_policy_metrics_summary,
)
from src.week7.w7e.fidelity_allocator import (
    allocate_ce_full_ranking,
    allocate_e0_k4,
    allocate_e1_primary,
    allocate_e1x_cross_encoder,
    allocate_e2_k6,
    allocate_e3_compact,
    allocate_e4_extended,
    validate_multifidelity_assignment,
)
from src.week7.w7e.generation import (
    assert_pipeline_invariants,
    execute_resumable_multifidelity_generation,
    load_existing_generation_cache,
)
from src.week7.w7e.medium_compressor import (
    MediumExtractionCache,
    count_tokens,
    extract_medium_passage,
    truncate_tokens_deterministic,
)
from src.week7.w7e.prompt_builder import (
    build_mixed_fidelity_sara_prompt,
    extract_mixed_fidelity_sara_embeddings,
)
from src.week7.w7e.sentence_scorer import SentenceScorer
from src.week7.w7e.sentence_splitter import split_into_sentences
from src.week7.w7e.token_accounting import (
    compute_compression_statistics,
    compute_medium_compression_ratio,
)

__all__ = [
    "split_into_sentences",
    "SentenceScorer",
    "count_tokens",
    "truncate_tokens_deterministic",
    "extract_medium_passage",
    "MediumExtractionCache",
    "validate_multifidelity_assignment",
    "allocate_e0_k4",
    "allocate_e1_primary",
    "allocate_e2_k6",
    "allocate_e3_compact",
    "allocate_e4_extended",
    "allocate_e1x_cross_encoder",
    "allocate_ce_full_ranking",
    "build_mixed_fidelity_sara_prompt",
    "extract_mixed_fidelity_sara_embeddings",
    "compute_compression_statistics",
    "compute_medium_compression_ratio",
    "load_existing_generation_cache",
    "execute_resumable_multifidelity_generation",
    "assert_pipeline_invariants",
    "compute_policy_metrics_summary",
    "compute_latency_comparison_table",
    "compute_stratified_4to6_analysis",
    "run_multifidelity_paper_clustered_bootstrap",
]
