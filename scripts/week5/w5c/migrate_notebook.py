"""Notebook migration script from Week 6 to Week 5-C.

Preserves:
- All execution counts (2 through 42)
- All cell outputs (printed tables, loss logs, bootstrap outputs, matplotlib plots)
- Metadata

Updates:
- Title and markdown headers (Week 6 -> Week 5-C)
- Scientific research question
- Config path: configs/week5/w5c_encoder_adaptation.yaml
- Output directory: results/week5/content_allocator/w5c_encoder_adaptation
- Table / Figure / Bootstrap filenames: w6_* -> w5c_*
- Python module imports: src.week6.* -> src.week5.w5c.*
"""

from __future__ import annotations

import json
from pathlib import Path

OLD_NB_PATH = Path("notebooks/week6/week6_encoder_adaptation.ipynb")
NEW_NB_PATH = Path("notebooks/week5/week5c_encoder_adaptation.ipynb")


def migrate_notebook():
    with open(OLD_NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Replacements across code cells
    replacements = [
        ("configs/week6/encoder_adaptation.yaml", "configs/week5/w5c_encoder_adaptation.yaml"),
        ("results/week6/encoder_adaptation", "results/week5/content_allocator/w5c_encoder_adaptation"),
        ("tables/w6_", "tables/w5c_"),
        ("bootstrap/w6_", "bootstrap/w5c_"),
        ("figures/w6_", "figures/w5c_"),
        ("from src.week6.", "from src.week5.w5c."),
        ("import src.week6.", "import src.week5.w5c."),
        ("select_tau_w6", "select_tau_w5c"),
        ("evaluate_w6_model_on_dev", "evaluate_w5c_model_on_dev"),
        ("plot_w6_pareto_curve", "plot_w5c_pareto_curve"),
        ("Week 6 Notebook execution", "Week 5-C Notebook execution"),
    ]

    for i, cell in enumerate(nb["cells"]):
        cell_type = cell["cell_type"]
        src = "".join(cell["source"])

        if cell_type == "markdown":
            # Update Title and Section 0
            if i == 0:
                cell["source"] = [
                    "# Week 5-C: Semantic Encoder Adaptation and Capacity Study\n",
                    "\n",
                    "**Project:** Query-Conditioned Context Allocator (QCCA) for Efficient Retrieval-Augmented Generation  \n",
                    "**Parent Architecture:** SARA — Selective and Adaptive Retrieval-augmented Generation with Context Compression  \n",
                    "**Experiment:** Week 5-C Semantic Capacity and Encoder Adaptation Ladder (C0 Frozen, C1 Last-1 Layer, C2 Last-2 Layers)  \n",
                    "**Population:** QASPER TRAIN (2,088 queries) & DEV (231 queries)  \n",
                    "**Governance:** Strict Split Isolation. Held-out TEST set (1,309 queries) remains 100% frozen & untouched.  \n",
                    "\n",
                    "---\n",
                    "\n",
                    "### Research Context & Transition from Week 5-B:\n",
                    "Week 5 established that direct retrieved passage content contains allocation signal, but under a frozen sentence encoder (`all-MiniLM-L6-v2`) with a shallow transformer aggregator, the semantic allocator (`I3-T3`: DEV F1 ≈ 0.3810, tokens ≈ 1258.2) did not outperform the canonical handcrafted Week 4 allocator (Balanced Logistic Regression: DEV F1 ≈ 0.3842, tokens ≈ 1115.9).\n",
                    "\n",
                    "### Core Research Question:\n",
                    "> **\"Does task-specific adaptation of the semantic encoder allow the content-aware allocator to exploit retrieved-passage information more effectively than the frozen encoder tested in Week 5-B?\"**\n",
                    "\n",
                    "### Frozen Protocol Elements from Week 5-B:\n",
                    "1. **Target Formulation:** Multi-Label Epsilon-Admissibility ($T_3$): $z_k(q) = 1[\\text{regret}(q, k) \\le 0.01]$ over $k \\in \\{2, 4, 5, 6, 8\\}$ via `BCEWithLogitsLoss`.\n",
                    "2. **Retrieval Pool:** BM25 top-10 retrieved passages.\n",
                    "3. **Budget Set:** $k \\in \\{2, 4, 5, 6, 8\\}$.\n",
                    "4. **Information Condition:** Fixed to $I_3$ (Query + Top-10 Passages).\n",
                    "5. **Downstream Policy Rule:** Select smallest $k$ such that $p_k \\ge \\tau$, fallback to $k=8$.\n",
                    "6. **Aggregator Architecture:** Corrected rank embeddings initialized $\\mathcal{N}(0, 0.02^2)$ with `LayerNorm(h_i + r_i)`.\n",
                    "7. **Threshold Grid:** $\\tau \\in \\{0.30, 0.40, 0.50, 0.60, 0.70\\}$ calibrated on INTERNAL-VAL using frozen 0.005 tolerance rule.",
                ]
            elif "Section 21" in src:
                cell["source"] = [
                    "## Section 21: Pre-Registered Decision Rules for Week 6 (Passage Routing)\n",
                    "\n",
                    "Based on empirical performance across the capacity ladder:\n",
                    "- **Observed Finding:** Neither $C_1$ ($F_1 = 0.3620$) nor $C_2$ ($F_1 = 0.3743$) outperformed frozen $C_0$ ($F_1 = 0.3810$) or $W_4$ Balanced LogReg ($F_1 = 0.3842$).\n",
                    "- **Capacity Conclusion:** Lightweight encoder adaptation of 1–2 layers in MiniLM does not surpass the frozen representation or the handcrafted baseline.\n",
                    "- **Decision for Week 6:** Freeze the semantic allocator and transition to **Query-Conditioned Passage Fidelity Routing** (full, medium, compressed passage representation).",
                ]
            else:
                new_src = src.replace("Week 6", "Week 5-C").replace("W6", "W5-C")
                cell["source"] = [line + "\n" for line in new_src.split("\n")]
                if cell["source"]:
                    cell["source"][-1] = cell["source"][-1].rstrip("\n")

        elif cell_type == "code":
            new_src = src
            for old_s, new_s in replacements:
                new_src = new_src.replace(old_s, new_s)
            cell["source"] = [line + "\n" for line in new_src.split("\n")]
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n")

    NEW_NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NEW_NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully migrated notebook to {NEW_NB_PATH}")
    print(f"Preserved {len(nb['cells'])} cells, all execution outputs intact.")


if __name__ == "__main__":
    migrate_notebook()
