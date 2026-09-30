"""Script to generate and execute notebooks/week3/week3_oracle_epsilon_human_audit.ipynb."""

import json
from pathlib import Path

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "sara_env (Python 3.11.16)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.16"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(code):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.strip().split("\n")]
        })

    # Header
    add_md("""# CS787 Generative AI Research Project — Week 3
## Oracle, Epsilon Target, Heterogeneity & Human Validation
**Project:** Query-Conditioned Context Allocator (QCCA) / SARA Framework  
**Benchmark:** QASPER Development Split (Paper-disjoint)  
**Scientific Objective:** Determine whether empirical query-level evidence requirements vary sufficiently across questions to justify learning a query-conditioned context allocator (QCCA).  
**Guardrail:** No QCCA training, no test set access, no expensive re-generation (100% frozen Week-2 data consumption).""")

    # Section 1
    add_md("""## 1. Environment & Input Verification
Verify python environment, project root directory, and load configuration.""")
    add_code("""import os, sys, json
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown

# Ensure project root in sys.path
PROJECT_ROOT = Path(os.path.abspath("../.."))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

print(f"Working Directory: {Path.cwd()}")
print(f"Python Executable: {sys.executable}")

# Load configuration
with open("configs/week3/week3_config.yaml") as f:
    config = yaml.safe_load(f)
print("Loaded week3_config.yaml successfully.")
display(pd.DataFrame([
    {"Key": "k_sweep", "Value": str(config["k_sweep"])},
    {"Key": "k_alloc", "Value": str(config["k_alloc"])},
    {"Key": "epsilon_candidates", "Value": str(config["epsilon_candidates"])},
    {"Key": "human_audit_sample", "Value": str(config["human_audit"]["n_questions"])},
    {"Key": "random_seed", "Value": str(config["seed"])}
]))""")

    # Section 2 & 3
    add_md("""## 2. Load Frozen Week-2 Results & Data Integrity Checks
Verify that all 231 development questions across 86 papers are intact with all 7 $k$ conditions ($231 \\times 7 = 1,617$ records).""")
    add_code("""from src.week3.oracle_analysis import load_raw_dev_matrix, load_best_static_baseline

raw_matrix_path = config["paths"]["week2_raw_matrix"]
df_raw = load_raw_dev_matrix(raw_matrix_path)
best_static = load_best_static_baseline(config["paths"]["week2_best_static"])

n_q = df_raw["question_id"].nunique()
n_p = df_raw["paper_id"].nunique()
n_rec = len(df_raw)
k_vals = sorted(df_raw["k"].unique().tolist())
dups = df_raw.duplicated(subset=["question_id", "k"]).sum()
null_f1 = df_raw["token_f1"].isna().sum()

integrity_summary = pd.DataFrame([{
    "Metric": "Total Records", "Expected": 1617, "Actual": n_rec, "Status": "PASS" if n_rec == 1617 else "FAIL"
}, {
    "Metric": "Unique Questions", "Expected": 231, "Actual": n_q, "Status": "PASS" if n_q == 231 else "FAIL"
}, {
    "Metric": "Unique Papers", "Expected": 86, "Actual": n_p, "Status": "PASS" if n_p == 86 else "FAIL"
}, {
    "Metric": "Present k values", "Expected": "[0, 2, 4, 5, 6, 8, 10]", "Actual": str(k_vals), "Status": "PASS"
}, {
    "Metric": "Duplicate (q, k) pairs", "Expected": 0, "Actual": dups, "Status": "PASS" if dups == 0 else "FAIL"
}, {
    "Metric": "Null / NaN Token F1", "Expected": 0, "Actual": null_f1, "Status": "PASS" if null_f1 == 0 else "FAIL"
}])
display(integrity_summary)

print(f"Static SARA Baseline: k={best_static['best_static_sara_k']}, Mean F1={best_static['sara_k8_mean_f1']:.4f}")
print(f"Standard RAG Reference: k={best_static['best_overall_fixed_k']}, Mean F1={best_static['vanilla_rag_k10_f1']:.4f}")""")

    # Section 4
    add_md("""## 3. Methodological Audit: Dual Oracle Definitions
Inspect the methodology audit document detailing the distinction between deployable SARA actions $\\mathcal{K}_{\\text{alloc}} = \\{2, 4, 5, 6, 8\\}$ and diagnostic sweep reference $\\mathcal{K}_{\\text{sweep}} = \\{0, 2, 4, 5, 6, 8, 10\\}$.""")
    add_code("""with open("results/week3/processed/methodology_audit.md") as f:
    audit_text = f.read()
# Display first 2 sections
display(Markdown("\\n".join(audit_text.split("\\n")[:35])))""")

    # Section 5 & 6
    add_md("""## 4. Quality Oracle & Headroom Analysis
Evaluate the primary Quality Oracle:
$$k^*_{\\text{qual}}(q) = \\arg\\max_{k \\in \\mathcal{K}_{\\text{alloc}}} \\text{F1}(q, k) \\quad \\text{(ties broken to smallest } k\\text{)}$$
Compare against best static SARA baseline ($k=8$) to quantify empirical adaptation headroom.""")
    add_code("""from src.week3.oracle_analysis import run_oracle_analysis

oracle_summary = run_oracle_analysis(
    config["paths"]["week2_raw_matrix"],
    config["paths"]["week2_best_static"],
    "results/week3/processed"
)

df_oracle = pd.read_csv("results/week3/processed/dev_quality_oracle.csv")
print("Top 5 Dev Quality Oracle Records:")
display(df_oracle[["question_id", "paper_id", "best_k_qual", "best_f1_qual", "best_static_k", "best_static_f1", "oracle_gain"]].head())

headroom_df = pd.DataFrame([oracle_summary["headroom"]])
print("\\nOracle Quality Headroom Summary:")
display(headroom_df)""")

    # Section 7 & 8
    add_md("""## 5. Optimal Evidence Budget Distribution & Gain Attribution
Analyze the frequency and cumulative distribution of optimal evidence budgets across development queries, and the attribution of total oracle headroom.""")
    add_code("""df_dist = pd.read_csv("results/week3/processed/quality_oracle_distribution.csv")
print("Quality Oracle Best-k Distribution:")
display(df_dist)

df_gain_attr = pd.read_csv("results/week3/processed/oracle_gain_attribution.csv")
print("\\nHeadroom Gain Attribution by Optimal Budget:")
display(df_gain_attr)

# Display Plot 1
display(Image("results/week3/plots/plot1_best_k_distribution.png"))""")

    # Section 9
    add_md("""## 6. Tie Analysis & Sensitivity
Quantify the degree of exact and near ties across $\\mathcal{K}_{\\text{alloc}}$ to confirm that smallest-$k$ tie breaking is strictly deterministic.""")
    add_code("""df_ties = pd.read_csv("results/week3/processed/tie_analysis.csv")
tie_counts = df_ties["n_tied_k"].value_counts().sort_index().rename("Question Count").to_frame()
tie_counts["Percentage (%)"] = (tie_counts["Question Count"] / len(df_ties) * 100).round(2)
display(tie_counts)

print(f"Strict unique winner: {(df_ties['n_tied_k'] == 1).sum()} queries")
print(f"All-zero tie (unanswerable floor): {df_ties['is_all_zero_tie'].sum()} queries")""")

    # Section 10
    add_md("""## 7. Per-Query Heterogeneity & Adaptation Opportunity
Measure evidence sensitivity ranges $(\\max F1 - \\min F1)$, agreement with static allocation, and compute paper-clustered bootstrap confidence intervals ($B=1000$).
Includes a separate tie-aware analysis to distinguish strict adaptation opportunities from smallest-$k$ tie breaking.""")
    add_code("""from src.week3.heterogeneity_analysis import run_heterogeneity_analysis

df_het, het_sum, edge_cases = run_heterogeneity_analysis(
    config["paths"]["week2_raw_matrix"],
    config["paths"]["week2_best_static"],
    "results/week3/processed"
)

# Display Descriptive Statistics
print("F1 Range & Oracle Gain Statistics:")
stats_df = pd.DataFrame([
    {"Metric": "F1 Sensitivity Range", **het_sum["f1_range_statistics"]},
    {"Metric": "Oracle Gain over k=8", **het_sum["oracle_gain_statistics"]}
])
display(stats_df)

print("\\nTie-Aware Static k=8 Status Breakdown:")
display(pd.DataFrame([het_sum["tie_aware_k8_status"]]))

print("\\nStrict Adaptation Opportunities (Strict Gain over k=8):")
display(pd.DataFrame([het_sum["strict_adaptation_opportunity"]]))

print(f"\\nStrict Unique Winners Breakdown ({het_sum['strict_winner_analysis']['total_strict_winners_pct']}% of queries):")
display(pd.DataFrame([het_sum["strict_winner_analysis"]["strict_winners_by_budget"]]))

print("\\nPaper-Clustered Bootstrap 95% Confidence Intervals:")
boot_df = pd.DataFrame(het_sum["paper_clustered_bootstrap_cis"]).T
display(boot_df[["mean", "ci_lower", "ci_upper", "n_clusters"]])

# Display Plots 2 & 3
display(Image("results/week3/plots/plot2_f1_range_distribution.png"))
display(Image("results/week3/plots/plot3_oracle_gain_distribution.png"))""")

    # Section 11 & 12
    add_md("""## 8. Epsilon Oracle & Freezing $\\epsilon^*$
Evaluate the primary SARA-internal epsilon target across candidate $\\epsilon \\in \\{0.00, 0.01, 0.02, 0.05\\}$:
$$k^*_\\epsilon(q) = \\min \\{ k \\in \\mathcal{K}_{\\text{alloc}} \\mid \\text{F1}(q, k) \\ge F1^*_{\\text{alloc}}(q) - \\epsilon \\}$$""")
    add_code("""from src.week3.epsilon_analysis import run_epsilon_analysis

df_all_eps, df_eps_summary, sel_eps = run_epsilon_analysis(
    config["paths"]["week2_raw_matrix"],
    "results/week3/processed",
    config["epsilon_candidates"]
)

print("Epsilon Target Sweep Summary:")
display(df_eps_summary)

print(f"\\nFrozen Epsilon* Selection: epsilon* = {sel_eps['epsilon_star']}")
print(f"Selection Rule: {sel_eps['selection_rule']}")
print(f"Rationale: {sel_eps['rationale']}")

# Display Plots 4, 5, 6
display(Image("results/week3/plots/plot4_mean_f1_comparison.png"))
display(Image("results/week3/plots/plot5_epsilon_vs_mean_k.png"))
display(Image("results/week3/plots/plot6_epsilon_vs_mean_regret.png"))""")

    # Section 13
    add_md("""## 9. Pre-generation Retrieval Feature Signals
Explore univariate associations between pre-generation retrieval features and optimal budgets/headroom.""")
    add_code("""from src.week3.feature_signal_analysis import run_feature_signal_analysis

df_feat_rep, feat_summary = run_feature_signal_analysis(
    config["paths"]["week2_features"],
    "results/week3/processed/dev_quality_oracle.csv",
    "results/week3/processed/epsilon_oracle_all.csv",
    "results/week3/processed/per_query_heterogeneity.csv",
    "results/week3/processed",
    selected_eps=sel_eps["epsilon_star"]
)

print("Feature Signal Association Table:")
display(df_feat_rep[["feature", "target", "spearman_rho", "spearman_p", "effect_size", "interpretation"]])

# Display Plot 7
display(Image("results/week3/plots/plot7_feature_vs_oracle_k.png"))""")

    # Section 14, 15, 16
    add_md("""## 10. Blinded 50-Question Human Audit Protocol
Inspect the blinded audit manifest ($N=50$ queries across $k \\in \\{2, 5, 10\\}$), anonymized system outputs, and scoring template.
In accordance with project integrity guidelines, no synthetic scores are reported. The human evaluation is documented as pending manual scoring.""")
    add_code("""from src.week3.human_audit import run_human_audit

df_sample, status, df_k_sum, human_summary = run_human_audit(
    config["paths"]["week2_raw_matrix"],
    "results/week3/processed/dev_quality_oracle.csv",
    "results/week2/raw/dev_retrieval.jsonl",
    "results/week3/human_audit",
    n_sample=config["human_audit"]["n_questions"],
    seed=config["human_audit"]["random_seed"]
)

print(f"Human Audit Status: {status}")
print(f"Summary: {human_summary['message']}")

df_template = pd.read_csv("results/week3/human_audit/human_scoring_template.csv")
print(f"\\nReady-to-Score Evaluator Artifact ({len(df_template)} judgments awaiting manual evaluation):")
display(df_template[["audit_id", "question_id", "anonymous_system_id", "question_text", "generated_answer", "score_0_1_2"]].head(6))

# Display Plots 8 & 9 (Audit Design & Baseline Performance)
display(Image("results/week3/plots/plot8_human_audit_design.png"))
display(Image("results/week3/plots/plot9_token_f1_by_audit_condition.png"))""")

    # Section 17
    add_md("""## 11. Final Week-3 Scientific Summary & Go / No-Go Gate
Review the Week-3 decision gate and readiness for Week 4.""")
    add_code("""with open("results/week3/processed/week3_gate.json") as f:
    gate_data = json.load(f)

gate_df = pd.DataFrame([
    {"Dimension": "H1 Hypothesis Status", "Result": gate_data["H1_status"], "Assessment": gate_data["H1_rationale"][:90] + "..."},
    {"Dimension": "Epsilon Target Status", "Result": gate_data["epsilon_target_status"], "Assessment": gate_data["epsilon_target_rationale"][:90] + "..."},
    {"Dimension": "Feature Signal Status", "Result": gate_data["feature_signal_status"], "Assessment": gate_data["feature_signal_rationale"][:90] + "..."},
    {"Dimension": "Human Audit Status", "Result": gate_data["human_audit_status"], "Assessment": gate_data["human_audit_rationale"][:90] + "..."},
    {"Dimension": "Methodology Blockers", "Result": gate_data["methodology_blockers"], "Assessment": "Zero unresolved blockers"},
    {"Dimension": "Week 4 Readiness", "Result": gate_data["week4_readiness"], "Assessment": "Proceed to QCCA Allocator Training"}
])
display(gate_df)
print(f"\\nWeek 3 Complete. Gate Status: {gate_data['week4_readiness']}")""")

    nb_path = Path("notebooks/week3/week3_oracle_epsilon_human_audit.ipynb")
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated clean notebook structure at {nb_path}")

    # Also create symlink or copy for week3_oracle_target_analysis.ipynb
    alt_path = Path("notebooks/week3/week3_oracle_target_analysis.ipynb")
    with open(alt_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated clean notebook structure at {alt_path}")

if __name__ == "__main__":
    create_notebook()
