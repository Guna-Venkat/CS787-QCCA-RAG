"""Week 0: Dataset + RAG Evidence Understanding / Motivation Study.

Provides modular dataset adapters, characterization tools, hidden-state representation
analysis, attention diagnostics, and evidence-utilization profiling for SARA.
"""

from src.week0.dataset_adapter import DatasetAdapter, QASPERAdapter

__all__ = [
    "DatasetAdapter",
    "QASPERAdapter",
]
