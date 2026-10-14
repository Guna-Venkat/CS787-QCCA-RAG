"""Week 7-A: Modular Composition of Adaptive Budget Allocation and Semantic Passage Routing.
"""

from src.week7.w7a.data import (
    load_frozen_w4_predictions,
    load_w7a_dev_bundle,
    verify_sara_checkpoint,
    audit_w7a_split_isolation,
)
from src.week7.w7a.routing import (
    construct_w7a_routing_assignments,
    compute_w7a_routing_change_records,
)
from src.week7.w7a.evaluation import (
    compute_safe_qa_metrics,
    compute_conditional_results_by_k,
    compute_w7a_win_tie_loss,
)
from src.week7.w7a.bootstrap import run_w7a_bootstrap_analysis

__all__ = [
    "load_frozen_w4_predictions",
    "load_w7a_dev_bundle",
    "verify_sara_checkpoint",
    "audit_w7a_split_isolation",
    "construct_w7a_routing_assignments",
    "compute_w7a_routing_change_records",
    "compute_safe_qa_metrics",
    "compute_conditional_results_by_k",
    "compute_w7a_win_tie_loss",
    "run_w7a_bootstrap_analysis",
]
