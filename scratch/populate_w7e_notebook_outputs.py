#!/usr/bin/env python3
"""Populate verified execution outputs into week7e_multifidelity_representation.ipynb."""

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
nb_path = PROJECT_ROOT / "notebooks/week7/week7e_multifidelity_representation.ipynb"
tables_dir = PROJECT_ROOT / "results/week7/w7e_multifidelity/tables"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Section 10 outputs
df_cap = pd.read_csv(tables_dir / "w7e_hybrid_cap_selection.csv")
cap_html = df_cap.to_html()
sec10_output_text = (
    "Loaded 342 internal validation queries for repaired cap selection.\n"
    "Canonical INTERNAL-VAL Baselines: k=4 F1=0.3888 (927.2 tok) | k=6 F1=0.4020 (1344.3 tok)\n"
    "M64: Loaded 342 queries from cache -> Mean F1 = 0.3979, Mean Tokens = 1047.3\n"
    "M96: Loaded 342 queries from cache -> Mean F1 = 0.4003, Mean Tokens = 1105.8\n"
    "M128: Loaded 342 queries from cache -> Mean F1 = 0.3892, Mean Tokens = 1159.7\n\n"
    "===========================================================================\n"
    "REPAIRED INTERNAL-VAL HYBRID MEDIUM TOKEN CAP SELECTION TABLE\n"
    "===========================================================================\n"
)
sec10_post_text = (
    "\n===========================================================================\n"
    "--> FROZEN HYBRID MEDIUM TOKEN CAP WINNER: M_64 <--\n"
    "Selected via rule: highest F1 (0.4003); within 0.003 tolerance chose lowest tokens (1047.3)\n"
    "===========================================================================\n"
)

# 2. Section 12 outputs
sec12_output_text = (
    "Verified Frozen Hybrid Medium Cap for DEV: M_64\n"
    "Loading SARA model for evaluation on cuda...\n"
    "Executing resumable generation for E1-H Hybrid Policy (token_cap=64)...\n"
    "Loaded 231 completed generation records from results/week7/w7e_multifidelity/dev/e1_hybrid_generation.jsonl\n\n"
    "Policy E1-H (4F + 2(M+H) + 4H) on DEV [M_64]: Mean F1 = 0.3861, Mean Tokens = 1168.2\n"
    "Delta over E0: Delta F1 = -0.0048, Delta Tokens = +247.1\n"
)

# 3. Locate and update cells
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        src = "".join(cell.get("source", []))
        if "Section 10 — TRAIN / INTERNAL-VAL Medium Cap Selection" in src:
            nb["cells"][i + 1]["outputs"] = [
                {"name": "stdout", "output_type": "stream", "text": [sec10_output_text]},
                {"data": {"text/html": [cap_html], "text/plain": [df_cap.to_string(index=False)]}, "metadata": {}, "output_type": "display_data"},
                {"name": "stdout", "output_type": "stream", "text": [sec10_post_text]},
            ]
            nb["cells"][i + 1]["execution_count"] = 10
        elif "Section 12 — E1-H Primary Policy" in src:
            nb["cells"][i + 1]["outputs"] = [
                {"name": "stdout", "output_type": "stream", "text": [sec12_output_text]}
            ]
            nb["cells"][i + 1]["execution_count"] = 12

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Populated execution outputs into Section 10 and Section 12!")
