"""Week 5-C: Semantic Capacity and Encoder Adaptation Study.

Exposes:
- Architecture audit and adaptation freezing policies (configure_encoder_adaptation, AdaptedContentAwareAllocator)
- Differential learning rate optimization and gradient health monitoring (train_adapted_allocator)
- Evaluation, threshold calibration, and paper-clustered bootstrap (evaluate_w6_model_on_dev as evaluate_w5c_model_on_dev)
- Representation shift diagnostics (compute_representation_shift)
"""

from __future__ import annotations
