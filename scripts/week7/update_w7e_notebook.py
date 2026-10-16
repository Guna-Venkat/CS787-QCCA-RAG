#!/usr/bin/env python3
"""Update week7e_multifidelity_representation.ipynb with corrected scientific conclusions,
softened wording, clean controlled M0 vs M1 ablation, and M2 framing.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NB_PATH = PROJECT_ROOT / "notebooks/week7/week7e_multifidelity_representation.ipynb"

def main():
    print(f"Reading notebook from {NB_PATH}...")
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update Cell 26 (Section 13 Markdown)
    nb["cells"][26]["source"] = [
        "## Section 13 — Controlled Diagnostic: Corrected Extract-Only MEDIUM (E1-M0) vs Hybrid (E1-H)\n",
        "To rigorously isolate the causal effect of retaining latent SFR embeddings without confounding variables:\n",
        "- **Corrected Extract-Only M64 ($E1\\text{-M0}$)**: $4F + 2M_{64} + 4H$ (Ranks 5–6 retain extract text only; latent SFR embeddings removed).\n",
        "- **Repaired Hybrid M64 ($E1\\text{-H}$)**: $4F + 2(M_{64}+H) + 4H$ (Ranks 5–6 retain both extract text AND original SFR embeddings).\n",
        "\n",
        "Holding constant:\n",
        "1. Exact same semantic ranking (`sentence-transformers/all-MiniLM-L6-v2`)\n",
        "2. Exact same extractive token cap ($M_{64}$)\n",
        "3. Fixed prompt builder (no blank document headers)\n",
        "4. Identical SARA checkpoint & greedy generation settings\n",
        "\n",
        "Then:\n",
        "$$\\Delta_{\\text{SFR retention}} = F1_{\\text{Hybrid } M_{64}} - F1_{\\text{Extract-Only } M_{64}}$$\n",
        "cleanly isolates whether retaining the whole-passage latent representation protects against extractive information loss.\n"
    ]

    # 2. Update Cell 27 (Section 13 Code)
    nb["cells"][27]["source"] = [
        "# Load Controlled Ablation Table isolating SFR retention\n",
        "ablation_path = PROJECT_ROOT / \"results/week7/w7e_multifidelity/tables/w7e_m0_vs_m1_ablation.csv\"\n",
        "if ablation_path.is_file():\n",
        "    df_ablation = pd.read_csv(ablation_path)\n",
        "    print(\"Controlled Ablation Table (Isolating SFR Retention):\")\n",
        "    display(df_ablation)\n",
        "    m0_row = df_ablation[df_ablation[\"Policy\"].str.contains(\"Extract-Only\")].iloc[0]\n",
        "    h_row = df_ablation[df_ablation[\"Policy\"].str.contains(\"Hybrid\")].iloc[0]\n",
        "    delta_sfr = float(h_row[\"F1\"]) - float(m0_row[\"F1\"])\n",
        "    delta_tokens = float(h_row[\"Tokens\"]) - float(m0_row[\"Tokens\"])\n",
        "    print(f\"\\nUnconfounded SFR Retention Delta: Delta F1 = {delta_sfr:+.4f} (Tokens: {delta_tokens:+.1f})\")\n",
        "else:\n",
        "    print(\"Controlled ablation table not found.\")\n"
    ]
    nb["cells"][27]["outputs"] = []

    # 3. Update Cell 36 (Section 18 Markdown)
    nb["cells"][36]["source"] = [
        "## Section 18 — Final DEV Evaluation Table\n",
        "We assemble the comprehensive results table across all baselines, learned historical models (W7-C3, W7-D), and W7-E multi-fidelity policies.\n",
        "Note: $E1\\text{-H}$ is the best-performing multi-fidelity configuration tested, but is not a global Pareto improvement over Semantic $k=4$ ($E0$).\n"
    ]

    # 4. Update Cell 37 (Section 18 Code)
    nb["cells"][37]["source"] = [
        "# Load and display updated DEV results table\n",
        "main_dev_table_path = tables_dir / \"w7e_main_dev_results.csv\"\n",
        "if main_dev_table_path.is_file():\n",
        "    df_main_dev = pd.read_csv(main_dev_table_path)\n",
        "    print(\"Main DEV Results Table:\")\n",
        "    display(df_main_dev)\n",
        "else:\n",
        "    print(\"Main DEV table not found.\")\n"
    ]
    nb["cells"][37]["outputs"] = []

    # 5. Update Cell 38 (Section 19 Markdown)
    nb["cells"][38]["source"] = [
        "## Section 19 — Stratified 4→6 Gain Analysis\n",
        "We examine the response surface by stratifying queries into three behavioral cohorts based on their response to full text promotion ($4 \\to 6$):\n",
        "* **Group A ($F_6 - F_4 > 0.01$, $n=36$)**: FULL ranks 5–6 help (Evidence-seeking).\n",
        "* **Group B ($|F_6 - F_4| \\le 0.01$, $n=158$)**: Neutral queries (~68.4% of dataset).\n",
        "* **Group C ($F_6 - F_4 < -0.01$, $n=37$)**: FULL ranks 5–6 hurt (Distraction-prone).\n",
        "\n",
        "**Core Empirical Pattern:**\n",
        "The stratified response surface shows that Hybrid MEDIUM has strongly heterogeneous effects: it recovers substantial utility when marginal passages are beneficial (Group A: recovers 58.4% of full $k4 \\to k6$ gain), remains nearly neutral for the majority of queries (Group B: $+0.0021$), and causes substantial degradation when those passages are harmful (Group C: falls from $0.5330$ to $0.2955$).\n"
    ]

    # 6. Update Cell 39 (Section 19 Code)
    nb["cells"][39]["source"] = [
        "# Load and display 3-Group Stratified Analysis for Hybrid M64 and Extract-Only M64\n",
        "strat_hybrid_path = tables_dir / \"w7e_4to6_stratified_analysis.csv\"\n",
        "strat_m0_path = tables_dir / \"w7e_m0_stratified_analysis.csv\"\n",
        "\n",
        "if strat_hybrid_path.is_file():\n",
        "    df_strat_h = pd.read_csv(strat_hybrid_path)\n",
        "    print(\"Stratified 4->6 Gain Analysis (Hybrid M64):\")\n",
        "    display(df_strat_h)\n",
        "\n",
        "if strat_m0_path.is_file():\n",
        "    df_strat_m0 = pd.read_csv(strat_m0_path)\n",
        "    print(\"\\nStratified 4->6 Gain Analysis (Extract-Only M64 Control):\")\n",
        "    display(df_strat_m0)\n",
        "\n",
        "# Visualization\n",
        "strat_fig_path = figures_dir / \"w7e_stratified_4to6.png\"\n",
        "if strat_fig_path.is_file():\n",
        "    from IPython.display import Image\n",
        "    display(Image(filename=str(strat_fig_path)))\n"
    ]
    nb["cells"][39]["outputs"] = []

    # 7. Update Cell 48 (Section 24 Markdown - Scientific Conclusions)
    nb["cells"][48]["source"] = [
        "## Section 24 — Scientific Conclusions\n",
        "### Synthesis of Empirical Outcomes:\n",
        "1. **Aggregate Pareto Reality**:\n",
        "   Hybrid MEDIUM ($E1\\text{-H}$: $F1=0.3861$, $1168$ tokens) is **not a global Pareto improvement** over Semantic $k=4$ ($E0$: $F1=0.3909$, $921$ tokens), spending $+247$ tokens for a $-0.0048$ F1 loss. It is the best-performing multi-fidelity configuration tested among text-extractive variants, but does not dominate $k=4$.\n",
        "2. **Conditional Utility of Surface Text**:\n",
        "   Explicit surface text is highly conditional:\n",
        "   - In Group A ($n=36$), it recovers **58.4%** of the full $k=6$ benefit ($\\Delta F1 = +0.2039$).\n",
        "   - In Group B ($n=158$), it is nearly neutral ($\\Delta F1 = +0.0021$).\n",
        "   - In Group C ($n=37$), exposing explicit surface text causes severe distractor degradation, dropping F1 from $0.5330$ to $0.2955$.\n",
        "3. **Mechanism: Explicit Text Distraction vs. Latent Protection**:\n",
        "   Explicit surface text can distract downstream generation when marginal passages contain irrelevant content. The clean controlled ablation demonstrates that retaining whole-passage SFR embeddings buffers against extractive information loss ($\\Delta_{\\text{SFR retention}} = +0.0106$, rising from $0.3755$ to $0.3861$), providing $+0.0357$ protection in Group C.\n",
        "4. **Multi-Vector Latent Representation ($M_2$) Benchmark**:\n",
        "   To test whether latent compression could provide higher representation capacity without injecting distracting surface text, we evaluated $M_2$ ($4F + 2(2V) + 4H$) on all 342 INTERNAL-VAL queries using the $\\text{whole} + \\text{best semantic half}$ design:\n",
        "   - Result: $F1 = 0.3691$ ($931.2$ context tokens, $\\Delta F1 = -0.0198$ vs $k=4$).\n",
        "   - Stratified Breakdown: While $M_2$ is substantially safer in Group C ($0.3275$ vs $0.1517$, $+0.1758$ over full text), it only recovers **2.8%** of the factual benefit in Group A ($0.1942$ vs $0.4720$). In extractive QA, verbatim factual tokens are compressed away by the latent bottleneck, which the frozen LM cannot decode from soft embeddings alone.\n",
        "   - Decision: Following the pre-registered stopping rule (*'If M2 shows no signal on INTERNAL-VAL, stop there'*), $M_2$ was not promoted to DEV. Week 7-E is formally frozen.\n"
    ]

    print("Writing updated notebook...")
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Notebook successfully updated!")

if __name__ == "__main__":
    main()
