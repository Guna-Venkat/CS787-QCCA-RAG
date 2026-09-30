"""Generate the definitive Week 3 Final Scientific Report adhering strictly to frozen artifacts and audit guidelines."""

import json
import os
from pathlib import Path

def generate_report():
    project_root = Path("/home/gunavenkat/Downloads/CS787-RAG-Project")
    
    report_md = """# Week 3 Final Scientific Report: Oracle, Epsilon Target, Heterogeneity & Human Validation Protocol

**Project:** Query-Conditioned Context Allocator (QCCA)  
**Parent Framework:** SARA — Selective and Adaptive Retrieval-augmented Generation with Context Compression  
**Benchmark:** QASPER Development Split (Paper-disjoint)  
**Date:** 2026-09-30  
**Status:** **AUTOMATIC ORACLE & TARGET ANALYSIS COMPLETE; HUMAN VALIDATION PENDING MANUAL EVALUATION**  

---

## 1. Executive Summary

Week 3 investigated whether the frozen empirical response surface from Week 2 exhibits sufficient query-level heterogeneity in natural-language evidence requirements to justify learning a Query-Conditioned Context Allocator (QCCA).

### Primary Completed Findings
1. **Strong Motivation for Query-Conditioned Allocation:** The optimal natural-language evidence budget varies substantially across queries. Relative to the best static SARA configuration ($k=8$), the deployable per-query oracle achieves a Mean Token F1 of **0.4895** vs **0.3950**, providing **+0.0945 (+23.93%)** absolute headroom (paper-clustered 95% CI: **[+0.0693, +0.1226]**).
2. **Strict Adaptation Opportunity:** For **29.44%** of development queries ($68/231$), at least one smaller allocation achieves strictly higher token-level F1 than static $k=8$. Furthermore, static $k=8$ is the unique strict winner for only **9.96%** of queries ($23/231$), while tying with at least one smaller budget for **60.61%** of queries ($140/231$).
3. **Epsilon Target Frozen at $\\epsilon^* = 0.01$:** $\\epsilon^* = 0.01$ was selected as a conservative development operating point because it introduces negligible mean regret (**0.0001**) relative to the deployable quality oracle while modestly reducing the selected evidence budget to a mean of $k = 3.49$ passages (mean context = **841.3 tokens**, a **52.16%** context reduction vs static $k=8$). Feasibility is **100.0%** across all 231 development questions.
4. **Exploratory Retrieval Features:** Pre-generation retrieval features exhibit weak exploratory associations with optimal evidence budgets (e.g., retrieval score margin $\\Delta_{12}$ with Spearman $\\rho = +0.1320, p = 0.0450$; competitive passage count $N_{\\text{high}}$ with Spearman $\\rho = -0.1421, p = 0.0308$). These statistics motivate candidate features for Week 4 but do not establish predictive performance or causal relationships.

### Pending Evaluation Status
- **Human Qualitative Validation:** **PENDING_MANUAL_EVALUATION**. Human qualitative validation has not yet been conducted. No human scores have been entered, and therefore no human-evaluation results or human-vs-automatic-metric correlations are reported in this report.

---

## 2. Research Question & Experimental Setup

### Scientific Objective
Determining whether pre-generation retrieval signals contain predictive signal for query-adaptive evidence allocation requires establishing first that the evidence requirement itself is non-uniform across queries. Week 3 measures the empirical headroom and constructs frozen supervised targets for Week 4.

### Primary Action Space vs Diagnostic Sweeps
- **Primary Deployable Action Space:** $\\mathcal{K}_{\\text{alloc}} = \\{2, 4, 5, 6, 8\\}$. Represents deployable SARA evidence budgets using the SARA context compressor and trained projector (`sara_qasper_proj_lr5e4_seed42`).
- **Standard RAG Reference ($k=10$):** $k=10$ represents uncompressed Standard RAG using a separate trained LoRA model (`rag_qasper_lora_r16_seed42`). It serves strictly as an external reference ceiling and is **NOT** part of the QCCA action space.
- **Global Diagnostic Sweep:** $\\mathcal{K}_{\\text{sweep}} = \\{0, 2, 4, 5, 6, 8, 10\\}$, used strictly for diagnostic ceiling/floor gap analysis.

---

## 3. Frozen Week-2 Fixed-k Response Surface

All Week 3 analyses consume the frozen Week-2 empirical response matrix ($N = 231$ development questions across 86 paper-disjoint partitions, 1,617 total evaluated records):

| $k$ | Method | Architecture | Mean Token F1 | Mean Context Tokens | Context Reduction vs $k=8$ |
| ---: | :--- | :--- | ---: | ---: | ---: |
| 0 | SARA | Pure Compression (Floor) | 0.1388 | 138.8 | 92.11% |
| 2 | SARA | SARA Context Compressor | 0.3038 | 534.5 | 69.60% |
| 4 | SARA | SARA Context Compressor | 0.3596 | 946.1 | 46.20% |
| 5 | SARA | Parent SARA Reproduction | 0.3629 | 1157.0 | 34.21% |
| 6 | SARA | SARA Context Compressor | 0.3857 | 1359.1 | 22.71% |
| **8** | **SARA** | **Best Static SARA Baseline** | **0.3950** | **1758.5** | **0.00%** |
| 10 | RAG Reference | Standard RAG (Separate Model) | 0.4090 | 2137.9 | -21.58% |

---

## 4. Deployable Quality Oracle & Headroom

The primary offline Quality Oracle over deployable SARA budgets is defined as:
$$F1^*_{\\text{alloc}}(q) = \\max_{k \\in \\mathcal{K}_{\\text{alloc}}} \\text{F1}(q, k)$$
with deterministic smallest-$k$ tie breaking for target assignment:
$$k^*_{\\text{qual}}(q) = \\arg\\max_{k \\in \\mathcal{K}_{\\text{alloc}}} \\text{F1}(q, k)$$

### Headroom Metrics
- **Best Static SARA ($k=8$):** Mean F1 = **0.3950**
- **Deployable Quality Oracle:** Mean F1 = **0.4895**
- **Absolute Oracle Headroom:** **+0.0945 F1**
- **Relative Headroom:** **+23.93%**
- **Paper-Clustered 95% Confidence Interval:** **[+0.0693, +0.1226]**

*Note:* The Quality Oracle represents an offline diagnostic upper bound across deployable SARA actions. It does NOT represent the performance of a learned QCCA model.

---

## 5. Query-Level Heterogeneity & Tie-Aware Analysis

To prevent smallest-$k$ tie breaking from artificially inflating disagreement with the static baseline, we report both the tie-broken target distribution and a separate tie-aware analysis.

### A. Tie-Aware Static $k=8$ Status Breakdown ($N=231$)

| Status Category | Definition | Query Count | Percentage | Scientific Interpretation |
| :--- | :--- | ---: | ---: | :--- |
| **$k=8$ is strictly suboptimal** | $\\text{F1}(q, 8) < \\max_{k \\in \\mathcal{K}_{\\text{alloc}}} \\text{F1}(q, k)$ | **68** | **29.44%** | Queries where at least one smaller budget achieves strictly higher F1 than $k=8$. |
| **$k=8$ ties for maximum with smaller $k$** | $8 \\in \\text{best\\_set}(q)$ and $|\\text{best\\_set}(q)| > 1$ | **140** | **60.61%** | Queries where smaller budgets match $k=8$ quality (context reduction opportunity). |
| **$k=8$ is in exact best set** | $8 \\in \\text{best\\_set}(q)$ | **163** | **70.56%** | Total queries where $k=8$ achieves the maximal F1 (unique or tied). |
| **$k=8$ is unique strict winner** | $\\text{best\\_set}(q) = \\{8\\}$ | **23** | **9.96%** | Queries where static $k=8$ is strictly required to achieve peak quality. |

### B. Strict Adaptation Thresholds (Strict Gain Over Static $k=8$)

For **29.44%** of development queries ($68/231$), at least one smaller allocation achieves strictly higher token-level F1 than static $k=8$:

| Strict Gain Threshold | Query Count | Percentage |
| ---: | ---: | ---: |
| $\\text{strict\\_gain} > 0.00$ | 68 | 29.44% |
| $\\text{strict\\_gain} > 0.01$ | 67 | 29.00% |
| $\\text{strict\\_gain} > 0.02$ | 67 | 29.00% |
| $\\text{strict\\_gain} > 0.05$ | 62 | 26.84% |
| $\\text{strict\\_gain} > 0.10$ | 55 | 23.81% |

### C. Strict Unique Winners Breakdown
Across development queries, **72 / 231 (31.17%)** exhibit a single, unique strict winner:
- Strict winner $k=2$: **17** queries (23.6% of strict winners)
- Strict winner $k=4$: **12** queries (16.7% of strict winners)
- Strict winner $k=5$: **7** queries (9.7% of strict winners)
- Strict winner $k=6$: **13** queries (18.1% of strict winners)
- Strict winner $k=8$: **23** queries (31.9% of strict winners)

---

## 6. Oracle Target Distribution & Gain Attribution

### Tie-Broken Deployable Oracle Distribution

Under deterministic smallest-$k$ tie breaking ($k^*_{\\text{qual}}$), **73.59%** of development queries ($170/231$) are assigned an oracle target of $k=2$ or $k=4$:

| $k$ | Query Count | Percentage | Cumulative (%) |
| ---: | ---: | ---: | ---: |
| 2 | 130 | 56.28% | 56.28% |
| 4 | 40 | 17.32% | 73.59% |
| 5 | 17 | 7.36% | 80.95% |
| 6 | 21 | 9.09% | 90.04% |
| 8 | 23 | 9.96% | 100.00% |

*Interpretation:* Under smallest-$k$ tie breaking, $73.59\%$ of queries are assigned an oracle target of $k \\le 4$. This reflects that $k \\le 4$ is sufficient to achieve peak quality for these queries, not that all $73.59\\%$ uniquely require only $2\\text{--}4$ passages.

### Gain Attribution
- Total empirical oracle headroom across development data: **21.8323 F1 points**.
- Headroom attributable to queries with tie-broken $k^*_{\\text{qual}} \\in \\{2, 4\\}$: **14.8254 F1 points**.
- **Share of Aggregate Oracle Headroom:** **67.91%**.
- **Clarification:** **73.59%** is the query share, whereas **67.91%** is the share of aggregate oracle headroom.

---

## 7. Epsilon Target Analysis & Selection of $\\epsilon^*$

The primary efficiency-aware training target for QCCA is defined as:
$$k^*_{\\epsilon}(q) = \\min \\{ k \\in \\mathcal{K}_{\\text{alloc}} \\mid \\text{F1}(q, k) \\ge F1^*_{\\text{alloc}}(q) - \\epsilon \\}$$

### Epsilon Target Candidate Sweep

| $\\epsilon$ | Mean $k$ | Median $k$ | Mean F1 | Mean Regret | Context Tokens | Context Reduct. vs $k=8$ | Feasibility | % $k=2$ | % $k=4$ | % $k=5$ | % $k=6$ | % $k=8$ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 3.53 | 2 | 0.4895 | 0.0000 | 848.4 | 51.75% | 100.0% | 56.28% | 17.32% | 7.36% | 9.09% | 9.96% |
| **0.01\*** | **3.49** | **2** | **0.4894** | **0.0001** | **841.3** | **52.16%** | **100.0%** | **57.14%** | **17.32%** | **7.36%** | **8.23%** | **9.96%** |
| 0.02 | 3.45 | 2 | 0.4892 | 0.0003 | 831.6 | 52.71% | 100.0% | 58.01% | 17.32% | 6.93% | 8.66% | 9.09% |
| 0.05 | 3.43 | 2 | 0.4888 | 0.0007 | 828.0 | 52.91% | 100.0% | 58.44% | 16.88% | 6.93% | 9.09% | 8.66% |

### Frozen Operating Point: $\\epsilon^* = 0.01$
$\\epsilon^* = 0.01$ was selected as a conservative development operating point because it introduces negligible mean regret relative to the deployable quality oracle while modestly reducing the selected evidence budget. The choice was made entirely from development-set oracle statistics and independently of downstream allocator performance.

At $\\epsilon^* = 0.01$:
- **Mean selected $k$:** **3.49**
- **Mean context tokens:** **841.3 tokens** (vs $1758.5$ for static $k=8$)
- **Context reduction vs $k=8$:** **52.16%**
- **Mean quality regret:** **0.0001 F1** (retains 99.98% of Quality Oracle F1)
- **Target Feasibility:** **100.0%** across all 231 development queries.

---

## 8. Exploratory Retrieval Feature Analysis

Pre-generation retrieval features show weak exploratory associations with optimal evidence budgets:

| Feature | Description | Spearman $\\rho$ vs $k^*_{\\text{qual}}$ | p-value | Interpretation |
| :--- | :--- | ---: | ---: | :--- |
| **$\\Delta_{12}$** | Top-1 vs Top-2 BM25 score gap | **+0.1320** | **0.0450** | Weak exploratory positive association with optimal budget. |
| **$N_{\\text{high}}$** | Passages with score $\\ge 0.5 s_1$ | **-0.1421** | **0.0308** | Weak exploratory negative association with optimal budget. |
| $\\rho_1$ | Top-1 relative score mass | +0.0518 | 0.4336 | Negligible association. |
| $H_{\\text{norm}}$ | Normalized score entropy | +0.0532 | 0.4214 | Negligible association. |
| $|\\mathcal{Q}|$ | Query token length | -0.0125 | 0.8503 | Negligible association. |

*Scientific Guardrail:* These statistics motivate candidate features for Week 4 but do not establish predictive performance or causal relationships.

---

## 9. Human Qualitative Validation: PENDING

**Human qualitative validation has not yet been conducted. No human scores have been entered, and therefore no human-evaluation results or human-vs-automatic-metric correlations are reported in this Week 3 report.**

### Prepared Audit Protocol
- **Sample Size:** 50 development questions $\\times$ 3 conditions ($k \\in \\{2, 5, 10\\}$) = 150 total judgments.
- **Rubric:**
  - `0 = Incorrect`: Factually wrong, ungrounded, or fails to address question.
  - `1 = Partially correct`: Contains relevant factual information but incomplete/imprecise.
  - `2 = Fully correct`: Accurately, completely, and concisely answers question.
- **Blinding Protocol:** Presentation order randomized per question under anonymous identifiers (**System A**, **System B**, **System C**). Actual budget $k$ is strictly hidden from the evaluation sheet. Secret key map stored at `results/week3/human_audit/blinding_map.json`.
- **Ready-to-Score Artifact:** Generated at `results/week3/human_audit/human_scoring_template.csv`.
- **Provenance Note:** An earlier automated heuristic was identified as inappropriate for human evaluation and its synthetic scores were removed; no synthetic scores are retained in the final Week 3 results.

---

## 10. Scientific Interpretation

The frozen development response surface exhibits substantial query-level heterogeneity in the amount of textual evidence associated with maximum observed quality. Relative to the best static SARA allocation ($k=8$), the deployable per-query oracle provides 0.0945 absolute F1 headroom, while 29.44% of queries have a strictly better smaller allocation. Under an $\\epsilon=0.01$ target, the oracle selects a mean budget of 3.49 passages with negligible mean regret, corresponding to approximately 52% context reduction relative to static $k=8$. These results provide a concrete motivation and target definition for learning a query-conditioned allocator.

This does not establish that a learned allocator can predict these targets or generalize them to unseen questions. That question is deferred to Week 4.

---

## 11. Limitations

1. **Oracle Bounds vs Model Performance:** Quality and Epsilon Oracles represent theoretical upper bounds computed post-hoc from observed responses; they do not represent learned model performance.
2. **Development Split Scope:** All findings are derived from the development partition. Generalization to unseen papers on the test set is untested.
3. **Automatic Metric Noise:** Token-level F1 is an automatic string-matching metric and may not capture full semantic nuances.
4. **Human Validation Pending:** Qualitative validation remains pending manual evaluation.
5. **Exploratory Feature Signal:** Observed retrieval feature correlations are weak and do not prove causal relationships or guarantee classifier accuracy.
6. **Separate Reference Arm ($k=10$):** Standard RAG $k=10$ uses a separate model architecture and LoRA checkpoint; it serves as a reference ceiling rather than a deployable SARA action.

---

## 12. Week-3 Go / No-Go Gate

- **Automatic oracle analysis:** **COMPLETE**
- **Heterogeneity analysis:** **COMPLETE**
- **Epsilon target definition:** **COMPLETE**
- **Epsilon target:** **FROZEN at 0.01**
- **Human qualitative validation:** **PENDING**
- **Week-3 methodology blockers:** **NONE**
- **Week-4 readiness:** **READY**

Week 4 may proceed because the automatic target-generation methodology is frozen. Human validation remains an independent pending qualitative-validation task.

---

## 13. Artifact Manifest & Reproducibility Verification

All result files, scripts, and notebook outputs are verified intact and reproducible:
- **Primary Report:** `results/week3/week3_final_scientific_report.md`
- **Processed Report Copy:** `results/week3/processed/week3_scientific_report.md`
- **Input Manifest:** `results/week3/processed/week2_input_manifest.json`
- **Master Summary JSON:** `results/week3/processed/week3_summary.json`
- **Gate JSON:** `results/week3/processed/week3_gate.json`
- **Frozen Target Metadata:** `results/week3/processed/selected_epsilon.json`
- **Human Scoring Template:** `results/week3/human_audit/human_scoring_template.csv`
- **Interactive Notebooks:**
  - `notebooks/week3/week3_oracle_epsilon_human_audit.ipynb`
  - `notebooks/week3/week3_oracle_target_analysis.ipynb`
- **Test Suite:** `tests/week3/` (12/12 passed)
"""

    # Write report files
    out_file1 = project_root / "results/week3/week3_final_scientific_report.md"
    out_file2 = project_root / "results/week3/processed/week3_scientific_report.md"
    
    out_file1.parent.mkdir(parents=True, exist_ok=True)
    out_file2.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_file1, "w", encoding="utf-8") as f:
        f.write(report_md)
    with open(out_file2, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print(f"Saved report to {out_file1}")
    print(f"Saved report to {out_file2}")

if __name__ == "__main__":
    generate_report()
