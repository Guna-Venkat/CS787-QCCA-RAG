"""LoRA / QLoRA parameter-efficient fine-tuning and GPU VRAM audit utilities for D2.

Audits:
- Bitsandbytes availability
- PEFT compatibility
- GPU device memory (TITAN RTX 24GB)
- Safe fallback when hardware/quantization libraries are unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def audit_d2_hardware_feasibility(
    device_id: int = 0,
    required_vram_gb: float = 14.0,
) -> Dict[str, Any]:
    """Audit GPU hardware and library environment for running D2 7B LLM router."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(device_id) if cuda_available else "CPU"
    total_vram_gb = (
        torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        if cuda_available else 0.0
    )

    bnb_available = False
    try:
        import bitsandbytes as bnb
        bnb_available = True
    except ImportError:
        bnb_available = False

    peft_available = False
    try:
        import peft
        peft_available = True
    except ImportError:
        peft_available = False

    is_feasible = cuda_available and total_vram_gb >= required_vram_gb and peft_available
    status = "FEASIBLE" if is_feasible else "BLOCKED_BY_HARDWARE"

    report = {
        "status": status,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "total_vram_gb": total_vram_gb,
        "required_vram_gb": required_vram_gb,
        "bitsandbytes_available": bnb_available,
        "peft_available": peft_available,
    }
    logger.info("D2 Hardware Feasibility Report: %s (VRAM: %.1f GB, bnb: %s, peft: %s)",
                status, total_vram_gb, bnb_available, peft_available)
    return report


def get_peft_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[Sequence[str]] = None,
) -> Any:
    """Create PEFT LoraConfig for Mistral-7B."""
    from peft import LoraConfig, TaskType

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
