"""Plotting and Visualization Utilities for Week 8 DAFiR.

Generates:
1. figures/w8a_promotion_gain_distribution.png: Distribution of counterfactual promotion utility Delta_i.
2. figures/w8a_oracle_quality_tokens.png: Oracle-1, Oracle-Selective-1, and Greedy Oracle-2 vs k4, k6, k8.
3. Quality-Context Frontier: Representation efficiency (Final Context Tokens vs F1).
4. Quality-Compute Frontier: Compute efficiency (Cumulative Prompt Tokens vs F1).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_promotion_gain_distribution(
    deltas: Sequence[float],
    output_path: Union[str, Path],
    epsilon_help: float = 0.01,
    epsilon_hurt: float = -0.01,
) -> None:
    """Plot distribution of counterfactual promotion gains Delta_i."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.array(deltas, dtype=float)
    help_pct = 100.0 * np.mean(arr > epsilon_help)
    hurt_pct = 100.0 * np.mean(arr < epsilon_hurt)
    neutral_pct = 100.0 * np.mean((arr >= epsilon_hurt) & (arr <= epsilon_help))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr, bins=50, color="#2b5c8f", edgecolor="black", alpha=0.8)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.axvline(epsilon_help, color="green", linestyle=":", linewidth=1.5, label=f"HELP (>+{epsilon_help}): {help_pct:.1f}%")
    ax.axvline(epsilon_hurt, color="red", linestyle=":", linewidth=1.5, label=f"HURT (<{epsilon_hurt}): {hurt_pct:.1f}%")

    ax.set_title("W8-A: Counterfactual Single-Passage Promotion Gain Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Marginal Promotion Utility $\Delta_i = F_i - F_0$", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved promotion gain distribution to %s", out_path)


def plot_quality_efficiency_frontiers(
    systems_data: Sequence[Dict[str, Any]],
    output_dir: Union[str, Path],
) -> Tuple[Path, Path]:
    """Plot both Representation Efficiency and Compute Efficiency Frontiers.

    systems_data items must contain:
    - name: str
    - f1: float
    - final_tokens: float
    - cum_tokens: float
    - marker: str (optional)
    - color: str (optional)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Frontier 1: Representation Efficiency (Final Context Tokens vs F1)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for s in systems_data:
        name = s["name"]
        f1 = s["f1"]
        tokens = s["final_tokens"]
        color = s.get("color", "#1f77b4")
        marker = s.get("marker", "o")
        ax1.scatter(tokens, f1, color=color, marker=marker, s=80, label=name, zorder=5)
        ax1.annotate(name, (tokens, f1), textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax1.set_title("Frontier 1: Representation Efficiency (Final Context Tokens vs F1)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Mean Final Context Tokens", fontsize=11)
    ax1.set_ylabel("Answer Token F1", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p1 = out_dir / "frontier1_representation_efficiency.png"
    plt.savefig(p1, dpi=300)
    plt.close(fig1)

    # Frontier 2: Compute Efficiency (Cumulative Prompt Tokens vs F1)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for s in systems_data:
        name = s["name"]
        f1 = s["f1"]
        cum_tokens = s["cum_tokens"]
        color = s.get("color", "#1f77b4")
        marker = s.get("marker", "o")
        ax2.scatter(cum_tokens, f1, color=color, marker=marker, s=80, label=name, zorder=5)
        ax2.annotate(name, (cum_tokens, f1), textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax2.set_title("Frontier 2: Compute Efficiency (Cumulative Prompt Tokens vs F1)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Mean Cumulative Prompt Tokens (Across All Passes)", fontsize=11)
    ax2.set_ylabel("Answer Token F1", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p2 = out_dir / "frontier2_compute_efficiency.png"
    plt.savefig(p2, dpi=300)
    plt.close(fig2)

    logger.info("Saved frontiers to %s and %s", p1, p2)
    return p1, p2
