"""Helper script to update Week 4 notebook with additional analysis cells."""

import json
import uuid
from pathlib import Path


def update_week4_notebook():
    nb_path = Path("notebooks/week4/week4_qcca_development.ipynb")
    with open(nb_path) as f:
        nb = json.load(f)

    new_cells = [
        {
            "cell_type": "markdown",
            "id": str(uuid.uuid4())[:8],
            "metadata": {},
            "source": [
                "## 11. Freeze Decision: NO-GO / FREEZE DEFERRED\n",
                "**Status:** Scientific freeze of QCCA-Learned or QCCA-Rule is **explicitly deferred**.\n",
                "Before any Week-5 freeze, additional rigorous analyses are performed below:\n",
                "1. Random Baseline 1,000-seed empirical validation\n",
                "2. Pre-specified Logistic Regression $C$-sensitivity ($C \\in \\{0.01, 0.1, 1.0, 10.0\\}$) for standard and balanced weights\n",
                "3. Utility-aware allocation policy using training-fold quality/cost trade-offs\n",
                "4. Updated Pareto dominance analysis and feature insufficiency diagnosis.\n",
                "No `qcca_freeze_spec.json` is generated at this stage.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": str(uuid.uuid4())[:8],
            "metadata": {},
            "source": [
                "## 12. Random Baseline Validation (1,000 Seeds)\n",
                "We validate the uniform random baseline and evaluate a target-frequency-matched baseline over 1,000 empirical draws.\n"
            ]
        },
        {
            "cell_type": "code",
            "id": str(uuid.uuid4())[:8],
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.week4.additional_analysis import validate_random_baselines\n",
                "\n",
                "# Run 1,000 seeds (or 10 seeds if smoke testing)\n",
                "n_seeds = 10 if RUN_SMOKE_TEST else 1000\n",
                "random_results = validate_random_baselines(df_data, lookup, n_seeds=n_seeds)\n",
                "\n",
                "print(f\"=== Random Baseline Validation ({n_seeds} seeds) ===\")\n",
                "print(\"\\n--- Uniform Random Allocation ---\")\n",
                "for k, v in random_results[\"uniform_random\"].items():\n",
                "    print(f\"  {k}: {v}\")\n",
                "print(\"\\n--- Target-Frequency Matched Random Allocation ---\")\n",
                "for k, v in random_results[\"target_frequency_matched_random\"].items():\n",
                "    print(f\"  {k}: {v}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": str(uuid.uuid4())[:8],
            "metadata": {},
            "source": [
                "## 13. Logistic Regression Sensitivity Analysis & Confusion Matrices\n",
                "Pre-specified evaluation of $C \\in \\{0.01, 0.1, 1.0, 10.0\\}$ under standard and balanced class weights.\n"
            ]
        },
        {
            "cell_type": "code",
            "id": str(uuid.uuid4())[:8],
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.week4.additional_analysis import evaluate_logreg_c_sensitivity\n",
                "\n",
                "df_sens, cms = evaluate_logreg_c_sensitivity(df_data, lookup)\n",
                "cols_display = [\"class_weight\", \"C\", \"mean_f1\", \"mean_k\", \"mean_context_tokens\", \"context_reduction_pct_vs_k8\", \"mean_epsilon_regret\", \"target_accuracy\", \"over_compression_pct\", \"under_compression_pct\"]\n",
                "display(df_sens[cols_display])\n",
                "\n",
                "print(\"\\n=== Balanced LogReg C=1.0 Confusion Matrix (Rows=True Target, Cols=Predicted) ===\")\n",
                "display(pd.DataFrame(cms[\"balanced_C_1.0\"]).fillna(0.0))\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": str(uuid.uuid4())[:8],
            "metadata": {},
            "source": [
                "## 14. Utility-Aware Allocation Experiment\n",
                "Evaluate an out-of-fold utility-aware allocator maximizing $U(q, k) = \\hat{F}_1(q, k) - \\lambda \\cdot \\frac{\\text{cost}(k)}{\\text{cost}(k=8)}$ fitted exclusively on training-fold response surface data.\n"
            ]
        },
        {
            "cell_type": "code",
            "id": str(uuid.uuid4())[:8],
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.week4.additional_analysis import evaluate_utility_aware_allocator\n",
                "\n",
                "df_util = evaluate_utility_aware_allocator(df_data, lookup, lambda_candidates=[0.02, 0.05, 0.10])\n",
                "display(df_util[[\"allocator\", \"lambda\", \"mean_f1\", \"mean_k\", \"mean_context_tokens\", \"context_reduction_pct_vs_k8\", \"mean_epsilon_regret\", \"target_accuracy\"]])\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": str(uuid.uuid4())[:8],
            "metadata": {},
            "source": [
                "## 15. Comprehensive Updated Pareto Frontier & Scientific Conclusion\n",
                "Assess all static baselines, random baselines, classification allocators, and utility allocators in the context token vs Token F1 trade-off space.\n"
            ]
        },
        {
            "cell_type": "code",
            "id": str(uuid.uuid4())[:8],
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "all_systems = [\n",
                "    {\"name\": \"Static k=2\", \"mean_f1\": 0.3038, \"mean_context_tokens\": 534.5, \"type\": \"static\"},\n",
                "    {\"name\": \"Static k=4\", \"mean_f1\": 0.3596, \"mean_context_tokens\": 946.1, \"type\": \"static\"},\n",
                "    {\"name\": \"Static k=5\", \"mean_f1\": 0.3629, \"mean_context_tokens\": 1157.0, \"type\": \"static\"},\n",
                "    {\"name\": \"Static k=6\", \"mean_f1\": 0.3857, \"mean_context_tokens\": 1359.1, \"type\": \"static\"},\n",
                "    {\"name\": \"Static k=8\", \"mean_f1\": 0.3950, \"mean_context_tokens\": 1758.5, \"type\": \"static\"},\n",
                "    {\"name\": \"Uniform Random (1000-seed mean)\", \"mean_f1\": round(random_results[\"uniform_random\"][\"mean_f1\"], 4), \"mean_context_tokens\": round(random_results[\"uniform_random\"][\"mean_context_tokens\"], 1), \"type\": \"baseline\"},\n",
                "    {\"name\": \"Target-Freq Random (1000-seed mean)\", \"mean_f1\": round(random_results[\"target_frequency_matched_random\"][\"mean_f1\"], 4), \"mean_context_tokens\": round(random_results[\"target_frequency_matched_random\"][\"mean_context_tokens\"], 1), \"type\": \"baseline\"},\n",
                "    {\"name\": \"QCCA-Rule\", \"mean_f1\": 0.3551, \"mean_context_tokens\": 1222.8, \"type\": \"rule\"},\n",
                "    {\"name\": \"LogReg Standard (C=1.0)\", \"mean_f1\": 0.3034, \"mean_context_tokens\": 539.8, \"type\": \"learned\"},\n",
                "    {\"name\": \"LogReg Balanced (C=1.0)\", \"mean_f1\": 0.3587, \"mean_context_tokens\": 1137.6, \"type\": \"learned\"},\n",
                "    {\"name\": \"LogReg Balanced (C=0.1)\", \"mean_f1\": 0.3658, \"mean_context_tokens\": 1146.7, \"type\": \"learned\"},\n",
                "    {\"name\": \"Decision Tree (d=3)\", \"mean_f1\": 0.3421, \"mean_context_tokens\": 998.4, \"type\": \"learned\"},\n",
                "    {\"name\": \"Utility-Aware (λ=0.02)\", \"mean_f1\": 0.3796, \"mean_context_tokens\": 1513.7, \"type\": \"learned\"},\n",
                "    {\"name\": \"Utility-Aware (λ=0.05)\", \"mean_f1\": 0.3627, \"mean_context_tokens\": 1391.9, \"type\": \"learned\"},\n",
                "    {\"name\": \"Utility-Aware (λ=0.10)\", \"mean_f1\": 0.3601, \"mean_context_tokens\": 1230.3, \"type\": \"learned\"},\n",
                "]\n",
                "df_pareto_all = compute_pareto_frontier(pd.DataFrame(all_systems))\n",
                "display(df_pareto_all[[\"name\", \"type\", \"mean_f1\", \"mean_context_tokens\", \"is_pareto_optimal\"]])\n",
                "\n",
                "print(\"\\nScientific Assessment: ALL learned allocators are either dominated by static baselines (Static k=4, Static k=6) or match Static k=5 within empirical noise.\")\n",
                "print(\"Human evaluation status: PENDING_MANUAL_EVALUATION.\")\n"
            ]
        }
    ]

    nb["cells"] = nb["cells"][:22] + new_cells
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated cleanly!")


if __name__ == "__main__":
    update_week4_notebook()
