"""Week 6: Query-Conditioned Passage Fidelity Routing.

Investigates whether selecting WHICH passages remain in full natural language
at fixed fidelity budget k improves SARA answer quality over default BM25 prefix.
Week 6-B investigates hybrid lexical-semantic passage routing.
"""

from src.week6.hybrid_routing import (
    normalize_scores_minmax,
    normalize_scores_zscore,
    fuse_bm25_semantic_scores,
    route_r3_fixed_fusion,
    compute_rrf_scores,
    route_r4_rrf,
    compute_rank_sum_scores,
    route_r5_rank_sum,
    select_best_alpha_from_internal_val,
)
from src.week6.disagreement import (
    compute_jaccard_overlap,
    compute_differing_passages_count,
    compute_rank_correlation,
    extract_future_gate_features_per_query,
    build_disagreement_analysis_dataframe,
)

__all__ = [
    "normalize_scores_minmax",
    "normalize_scores_zscore",
    "fuse_bm25_semantic_scores",
    "route_r3_fixed_fusion",
    "compute_rrf_scores",
    "route_r4_rrf",
    "compute_rank_sum_scores",
    "route_r5_rank_sum",
    "select_best_alpha_from_internal_val",
    "compute_jaccard_overlap",
    "compute_differing_passages_count",
    "compute_rank_correlation",
    "extract_future_gate_features_per_query",
    "build_disagreement_analysis_dataframe",
]
