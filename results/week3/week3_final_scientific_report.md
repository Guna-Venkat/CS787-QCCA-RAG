# Week 3 Scientific Report: Oracle, Epsilon Target, Heterogeneity & Human Validation

**Project:** Query-Conditioned Context Allocator (QCCA)  
**Parent Framework:** SARA — Selective and Adaptive Retrieval-augmented Generation with Context Compression  
**Benchmark:** QASPER Development Split (Paper-disjoint)  
**Date:** 2026-09-30 17:58:07  
**Status:** **WEEK-3 ORACLE & TARGET ANALYSIS COMPLETE; HUMAN AUDIT PENDING MANUAL EVALUATION**  

---

## 1. Executive Summary

Week 3 investigated whether the frozen empirical response surface from Week 2 exhibits sufficient query-level heterogeneity in evidence requirements to justify learning a Query-Conditioned Context Allocator (QCCA).

### Scope and Completion Status

- **COMPLETED:**
  - Automatic fixed-$k$ response surface analysis across $\mathcal{K}_{\text{sweep}} = \{0, 2, 4, 5, 6, 8, 10\}$
  - Deployable Quality Oracle evaluation over $\mathcal{K}_{\text{alloc}} = \{2, 4, 5, 6, 8\}$
  - Tie-aware heterogeneity analysis and strict winner distribution
  - Epsilon-target analysis and selection/freeze of operating point $\epsilon^* = 0.01$
  - Exploratory retrieval-feature association analysis
- **PENDING:**
  - Human qualitative validation (150 blinded evaluation judgments across 50 development questions)

> **Important:** Human qualitative validation has NOT yet been performed. No human scores have been entered. This report presents only automatic metric analyses and target definition experiments.

### Summary of Primary Findings

1. **Substantial Oracle Quality Headroom:** The deployable per-query quality oracle achieves a Mean Token F1 of **0.4895**, providing **+0.0945 (+23.93%)** absolute headroom over the best static SARA baseline ($k=8$, Mean F1 = **0.3950**), with a paper-clustered 95% confidence interval of **[+0.0693, +0.1226]**.
2. **Prevalent Query-Level Heterogeneity:** For **29.44%** of development queries, at least one smaller allocation achieves strictly higher token-level F1 than $k=8$. Furthermore, static $k=8$ is tied for maximum quality with a smaller budget in **60.61%** of queries, and is the unique strict winner in only **9.96%** of queries.
3. **Concentration of Target Allocations:** Under deterministic smallest-$k$ tie breaking, **73.59%** of development queries are assigned an oracle target of $k=2$ or $k=4$, accounting for **67.91%** of aggregate oracle headroom gains.
4. **Frozen Operating Point $\epsilon^* = 0.01$:** Selecting $\epsilon^* = 0.01$ reduces the mean budget to **3.49 passages** (mean context length of **841.3 tokens**), achieving a **52.16% context token reduction** relative to static $k=8$ (1,758.5 tokens) with negligible mean regret (**0.0001** F1).
5. **Pre-generation Retrieval Feature Signals:** Weak exploratory associations were identified for the top-1/top-2 score margin $\Delta_{12}$ (Spearman $\rho = +0.1320, p = 0.0450$) and competitive passage count $N_{\text{high}}$ (Spearman $\rho = -0.1421, p = 0.0308$), motivating non-linear multivariate modeling in Week 4.

---

## 2. Research Question and Hypothesis

### Central Research Question

> **What did we learn from the Week-2 fixed-$k$ response surface, and does the observed query-level heterogeneity provide sufficient motivation for a query-conditioned evidence allocator?**

### Scientific Hypotheses

- **Hypothesis H1 (Heterogeneity & Headroom):** Evidence requirements vary significantly across queries in long-document QA. An optimal per-query context allocation policy over deployable budgets $\mathcal{K}_{\text{alloc}}$ can achieve substantial quality gains (higher token-level F1) and context token savings over any fixed static allocation baseline.
- **Hypothesis H2 (Epsilon Target Feasibility):** A target formulation with tolerance parameter $\epsilon^* = 0.01$ allows selecting smaller evidence budgets with negligible quality regret relative to the exact maximum oracle.

---

## 3. Experimental Setup

### Dataset & Benchmark Partition

- **Dataset:** QASPER (Question Answering on Scientific Papers) development split.
- **Volume:** 231 unique questions across 86 scientific papers (paper-disjoint from training split).
- **Format:** Long-document questions requiring multi-passage context extraction and reasoning.

### Allocation Action Spaces

- **Deployable SARA Allocation Space ($\mathcal{K}_{\text{alloc}}$):**
  $$\mathcal{K}_{\text{alloc}} = \{2, 4, 5, 6, 8\}$$
  All primary QCCA target definitions, quality oracle computations, and operational evaluations are strictly restricted to $\mathcal{K}_{\text{alloc}}$.
- **Global Diagnostic Sweep Space ($\mathcal{K}_{\text{sweep}}$):**
  $$\mathcal{K}_{\text{sweep}} = \{0, 2, 4, 5, 6, 8, 10\}$$
- **Reference & Diagnostic Conditions:**
  - **$k=0$ (Pure Compression):** SARA model evaluated with zero retrieved context passages (138.8 context tokens). Serves as a diagnostic lower bound.
  - **$k=10$ (Standard RAG Reference Ceiling):** Separate uncompressed Vanilla RAG checkpoint (`rag_qasper_lora_r16_seed42`) using 10 raw retrieved passages (2,137.9 tokens). **$k=10$ is NOT part of the deployable SARA action space $\mathcal{K}_{\text{alloc}}$**.

### Primary Automatic Evaluation Metrics

- **Token F1:** Primary quality metric measuring word-level overlap between generated response and gold references.
- **Exact Match (EM):** String-level exact match accuracy.
- **ROUGE-L:** Longest common subsequence recall-oriented score.
- **Mean Context Tokens:** Average prompt token count passed to the language model after context compression.

---

## 4. Frozen Week-2 Response Surface

The empirical fixed-$k$ response surface generated during Week 2 provides the immutable input foundation for all Week 3 oracle and heterogeneity analyses.

| $k$ | Method / Architecture | Checkpoint ID | Mean F1 | Mean EM | Mean ROUGE-L | Mean Context Tokens | Context Reduct. vs $k=10$ |
|---:|:---|:---|---:|---:|---:|---:|---:|
|  0 | SARA (Pure Comp.) | `sara_qasper_proj_lr5e4_seed42` | 0.1388 | 0.0866 | 0.1400 |   138.8 | 93.51% |
|  2 | SARA | `sara_qasper_proj_lr5e4_seed42` | 0.3038 | 0.1429 | 0.2680 |   534.5 | 74.99% |
|  4 | SARA | `sara_qasper_proj_lr5e4_seed42` | 0.3596 | 0.1688 | 0.3134 |   946.1 | 55.75% |
|  5 | SARA (Parent Default) | `sara_qasper_proj_lr5e4_seed42` | 0.3629 | 0.1861 | 0.3169 | 1,157.0 | 45.88% |
|  6 | SARA | `sara_qasper_proj_lr5e4_seed42` | 0.3857 | 0.1861 | 0.3340 | 1,359.1 | 36.43% |
| **8** | **SARA (Best Static)** | `sara_qasper_proj_lr5e4_seed42` | **0.3950** | 0.1861 | 0.3377 | **1,758.5** | **17.75%** |
| 10 | RAG Reference Ceiling | `rag_qasper_lora_r16_seed42` | 0.4090 | 0.2035 | 0.3523 | 2,137.9 | 0.00% |

### Key Observations

- **Best Static SARA Baseline:** Static **$k=8$** yields the highest average token F1 (**0.3950**) among all deployable SARA allocation options.
- **Reference Ceiling:** Standard RAG at $k=10$ achieves **0.4090 F1** using uncompressed context (2,137.9 tokens). It serves as an uncompressed reference ceiling, not a deployable SARA allocation action.

---

## 5. Deployable Quality Oracle

### Mathematical Formulation

The primary deployable quality oracle $F1^*_{\text{alloc}}(q)$ defines the empirical upper bound of per-query allocation performance over the deployable action space $\mathcal{K}_{\text{alloc}} = \{2, 4, 5, 6, 8\}$:

$$F1^*_{\text{alloc}}(q) = \max_{k \in \mathcal{K}_{\text{alloc}}} F1(q, k)$$

The deterministic tie-broken oracle target $k^*_{\text{qual}}(q)$ selects the minimal evidence budget achieving peak quality:

$$k^*_{\text{qual}}(q) = \min \left\{ k \in \mathcal{K}_{\text{alloc}} : F1(q, k) = F1^*_{\text{alloc}}(q) \right\}$$

### Quality Headroom Results

- **Static $k=8$ Baseline Mean F1:** **0.3950**
- **Deployable Per-Query Oracle Mean F1:** **0.4895**
- **Absolute Oracle Headroom:** **+0.0945 F1 points**
- **Relative Headroom:** **+23.93%**
- **Paper-Clustered 95% Confidence Interval:** **[+0.0693, +0.1226]** (computed via 1,000 bootstrap resamples clustered by paper ID across 86 papers)

> **Methodological Clarification:** The deployable quality oracle represents an offline upper-bound diagnostic analysis assuming perfect per-query selection. It does NOT represent the actual performance of a learned QCCA model.

---

## 6. Query-Level Heterogeneity

### Oracle Target Distribution

Under deterministic smallest-$k$ tie breaking ($k^*_{\text{qual}}$), the distribution of optimal evidence budgets across the $N=231$ development queries is as follows:

|   k |   count |   pct |   cumulative_pct |
|----:|--------:|------:|-----------------:|
|   2 |     130 | 56.28 |            56.28 |
|   4 |      40 | 17.32 |            73.59 |
|   5 |      17 |  7.36 |            80.95 |
|   6 |      21 |  9.09 |            90.04 |
|   8 |      23 |  9.96 |           100    |

### Key Structural Insights

- **Concentration at Low Budgets:** Under deterministic smallest-$k$ tie breaking, **73.59% of development queries** (170/231) are assigned an oracle target of $k=2$ or $k=4$.
- **High-Budget Necessity:** Only **9.96% of queries** (23/231) require the full $k=8$ budget under exact tie breaking.

> **Precise Interpretation:** "Under deterministic smallest-k tie breaking, 73.59% of development queries are assigned an oracle target of k=2 or k=4." This reflects tie-broken target assignments and should not be interpreted as claiming that 73.59% of queries strictly require only 2–4 passages to avoid quality loss.

---

## 7. Tie-Aware Analysis

Because multiple evidence budgets can yield identical token-level F1 scores for a given query, evaluating static $k=8$ requires tie-aware analysis across the full $N=231$ development set.

### Breakdown of Static $k=8$ Status

| Status Category | Query Count | Percentage | Description / Significance |
|:---|---:|---:|:---|
| **$k=8$ is strictly suboptimal** ($	ext{gain} > 0$) | **68** | **29.44%** | Queries where at least one smaller budget $k < 8$ achieves strictly higher F1 than $k=8$. |
| **$k=8$ ties for maximum with smaller budget** | **140** | **60.61%** | Queries where $k=8$ matches peak F1, but a smaller budget achieves identical quality. |
| **$k=8$ is unique strict winner** | **23** | **9.96%** | Queries where $k=8$ strictly outperforms all smaller budgets $k \in \{2, 4, 5, 6\}$. |
| **$k=8$ is in exact best set** | **163** | **70.56%** | Total queries where $k=8$ achieves maximal F1 (unique winner or tied). |

### Strict Gain Thresholds Over Static $k=8$

| Strict Gain Threshold over $k=8$ | Query Count | Percentage of Queries |
|---:|---:|---:|
| $> 0.00$ F1 | 68 | 29.44% |
| $> 0.01$ F1 | 67 | 29.00% |
| $> 0.02$ F1 | 67 | 29.00% |
| $> 0.05$ F1 | 62 | 26.84% |
| $> 0.10$ F1 | 55 | 23.81% |

### Precise Scientific Wording

> "For 29.44% of development queries, at least one smaller allocation achieves strictly higher token-level F1 than k=8."

Static $k=8$ is NOT described as "actively harmful"; rather, fixed allocation forces over-allocation when smaller budgets achieve equal or superior performance.

---

## 8. Oracle Gain Attribution

To understand where oracle headroom originates, we attribute the aggregate oracle headroom gain (**21.8323 F1 points** across 231 queries) across target budget categories:

|   best_k_qual |   question_count |   total_gain |   mean_gain_per_question |   pct_of_total_gain |
|--------------:|-----------------:|-------------:|-------------------------:|--------------------:|
|             2 |              130 |       8.3497 |                   0.0642 |               38.24 |
|             4 |               40 |       6.4757 |                   0.1619 |               29.66 |
|             5 |               17 |       3.4256 |                   0.2015 |               15.69 |
|             6 |               21 |       3.5813 |                   0.1705 |               16.4  |
|             8 |               23 |       0      |                   0      |                0    |

### Distinguishing Query Share from Headroom Attribution

- **Query Share for $k \in \{2, 4\}$:** **73.59%** (170 / 231 queries).
- **Headroom Attribution Share for $k \in \{2, 4\}$:** **67.91%** (14.8254 / 21.8323 aggregate F1 points).

Queries assigned $k=2$ or $k=4$ under tie breaking account for over two-thirds of total potential quality headroom.

---

## 9. Epsilon Target Analysis

### Mathematical Formulation

To allow conservative budget selection when a smaller budget yields near-optimal quality within a small tolerance $\epsilon$, we define the $\epsilon$-oracle target:

$$k^*_{\epsilon}(q) = \min \left\{ k \in \mathcal{K}_{\text{alloc}} : F1(q, k) \ge F1^*_{\text{alloc}}(q) - \epsilon \right\}$$

### Epsilon Sweep Results ($\mathcal{K}_{\text{alloc}}$)

|   epsilon |   mean_k |   median_k |   mean_f1 |   mean_regret |   context_reduction_pct_vs_k8 |   feasibility_pct_sara |
|----------:|---------:|-----------:|----------:|--------------:|------------------------------:|-----------------------:|
|      0    |     3.53 |          2 |    0.4895 |        0      |                         51.75 |                    100 |
|      0.01 |     3.49 |          2 |    0.4894 |        0.0001 |                         52.16 |                    100 |
|      0.02 |     3.45 |          2 |    0.4892 |        0.0003 |                         52.71 |                    100 |
|      0.05 |     3.43 |          2 |    0.4888 |        0.0007 |                         52.91 |                    100 |

### Analysis of Regret and Budget Sizing

As $\epsilon$ increases from 0.00 to 0.05, mean budget drops from 3.53 to 3.43 passages, with mean quality regret remaining below 0.0007 F1 points. Feasibility is **100.0%** across all candidate $\epsilon$ values.

---

## 10. Selected Operating Point

### Frozen Operating Point: $\epsilon^* = 0.01$

- **Selection Rationale:** $\epsilon^* = 0.01$ was selected as a conservative development operating point because it introduces negligible mean regret relative to the deployable quality oracle while modestly reducing the selected evidence budget. The choice was made entirely from development-set oracle statistics and independently of downstream allocator performance.
- **Frozen Target Properties at $\epsilon^* = 0.01$:**
  - **Mean Selected Budget ($k$):** **3.49 passages** (vs 8.00 static)
  - **Mean Context Length:** **841.3 tokens**
  - **Context Reduction vs Static $k=8$ (1,758.5 tokens):** **52.16%**
  - **Context Reduction vs Standard RAG $k=10$ (2,137.9 tokens):** **60.65%**
  - **Mean Quality Retained (F1):** **0.4894** (vs oracle 0.4895)
  - **Mean Regret vs Oracle:** **0.0001 F1 points**
  - **Target SARA Feasibility:** **100.0%** (231 / 231 queries)

---

## 11. Exploratory Feature Signals

We conducted exploratory rank-correlation analysis between pre-generation retrieval features (derived from BM25 scores over top-10 passages) and the tie-broken target $k^*_{\text{qual}}$.

| Feature Symbol | Description | Spearman $\rho$ | $p$-value | Signal Strength | Scientific Interpretation |
|:---|:---|---:|---:|:---|:---|
| **$\Delta_{12}$** | Score gap between top-1 and top-2 passage | **+0.1320** | **0.0450** | Weak Positive | Queries with a dominant lead passage slightly favor higher budgets. |
| **$N_{\text{high}}$** | Passages with BM25 score $\ge 0.5 s_1$ | **-0.1421** | **0.0308** | Weak Negative | Multiple highly competitive passages weakly correlate with lower budgets. |
| **$\rho_1$** | Top-1 score relative mass ($s_1 / \sum s_i$) | +0.0518 | 0.4336 | Negligible | Univariate relative mass provides no linear predictive signal. |
| **$H_{\text{norm}}$** | Normalized score entropy | +0.0532 | 0.4214 | Negligible | Retrieval score dispersion alone does not dictate budget linearly. |
| **$|\mathcal{Q}|$** | Query token length | -0.0125 | 0.8503 | Negligible | Query length is invariant to required evidence volume. |

### Key Takeaway

Pre-generation retrieval features show **weak exploratory associations**. They motivate candidate feature sets for Week 4 classifier training, but do NOT establish predictive performance or causal relationships.

---

## 12. Human Qualitative Validation — PENDING

### Audit Status: PENDING_MANUAL_EVALUATION

> **Human qualitative validation has not yet been conducted. No human scores have been entered, and therefore no human-evaluation results or human-vs-automatic-metric correlations are reported in this Week 3 report.**

### Prepared Evaluation Protocol

- **Sample Size:** 50 development questions $\times$ 3 conditions ($k \in \{2, 5, 10\}$) = **150 total judgments**.
- **Rubric:**
  - `0`: Incorrect / Unanswered
  - `1`: Partially correct / incomplete answer
  - `2`: Fully correct and complete answer
- **Blinding & Randomization:**
  - System outputs labeled anonymized as `System A`, `System B`, `System C`.
  - Presentation order randomized per question.
  - Key mapping $k \to \text{System}$ stored separately in blinded manifest (`results/week3/human_audit/blinding_map.json`).
- **Scoring Template Artifact:**
  `results/week3/human_audit/human_scoring_template.csv`

### Audit Artifact Readiness

The audit artifact is ready for manual evaluation. It is intentionally excluded from the quantitative Week 3 findings until genuine human scores are entered. An earlier automated heuristic was identified as inappropriate for human evaluation and its synthetic scores were removed; no synthetic scores are retained in the final Week 3 results.

---

## 13. Scientific Interpretation

### Central Week-3 Conclusion

> "The frozen development response surface exhibits substantial query-level heterogeneity in the amount of textual evidence associated with maximum observed quality. Relative to the best static SARA allocation ($k=8$), the deployable per-query oracle provides 0.0945 absolute F1 headroom, while 29.44% of queries have a strictly better smaller allocation. Under an $\epsilon=0.01$ target, the oracle selects a mean budget of 3.49 passages with negligible mean regret, corresponding to approximately 52% context reduction relative to static $k=8$. These results provide a concrete motivation and target definition for learning a query-conditioned allocator."

### Explicit Qualification

> This upper-bound analysis does not establish that a learned allocator can predict these targets or generalize them to unseen questions. That research question is deferred to Week 4.

---

## 14. Limitations

1. **Oracle Bounds vs Learned Allocators:** Quality oracle performance represents an offline upper bound assuming perfect per-query selection, not the achievable performance of a learned model.
2. **Development Partition Limits:** All analysis is conducted on the 231 QASPER development queries; test-set generalization cannot be inferred from development oracle statistics.
3. **Metric Limitations:** Token-level F1 is an automatic surface-overlap metric and may not capture full semantic correctness or factual precision.
4. **Pending Human Validation:** Human qualitative evaluation has not yet been conducted (`PENDING_MANUAL_EVALUATION`).
5. **Weak Feature Correlations:** Exploratory retrieval features exhibit weak univariate correlations ($ho \in [-0.14, +0.13]$), requiring non-linear non-trivial modeling in Week 4.
6. **Reference Action Exclusion:** Standard RAG $k=10$ uses an uncompressed model checkpoint and is excluded from deployable SARA allocation actions.
7. **Absence of Causal Claims:** Observed feature signals are exploratory associations and do not establish causal relationships between retrieval distributions and answer quality.

---

## 15. Week-3 Conclusions

1. **H1 Supported:** Per-query evidence requirements vary significantly, providing +0.0945 F1 headroom (+23.93%) over static $k=8$.
2. **Target Frozen:** Operating point $\epsilon^* = 0.01$ is frozen, achieving 3.49 mean passages (841.3 tokens) with 52.16% context reduction vs static $k=8$.
3. **Feature Baseline Established:** Pre-generation features $\Delta_{12}$ and $N_{\text{high}}$ provide initial non-trivial inputs for Week 4 classifier training.
4. **Methodological Rigor Enforced:** Pure automatic metrics reported; human validation cleanly isolated as pending manual evaluation.

---

## 16. Week-4 Handoff / Frozen Decisions

### Week-3 Gate Status Summary

- **Automatic Oracle Analysis:** COMPLETE
- **Heterogeneity Analysis:** COMPLETE
- **Epsilon Target Definition:** COMPLETE
- **Epsilon Target:** FROZEN at $\epsilon^* = 0.01$
- **Human Qualitative Validation:** PENDING (`PENDING_MANUAL_EVALUATION`)
- **Week-3 Methodology Blockers:** NONE
- **Week-4 Readiness:** **READY**

### Frozen Operational Specifications for Week 4

1. **Supervised Target Dataset:** Target labels $k^*_{\epsilon=0.01}(q)$ generated from `dev_quality_oracle.csv`.
2. **Deployable Action Space:** $\mathcal{K}_{\text{alloc}} = \{2, 4, 5, 6, 8\}$.
3. **Baseline Comparison Standard:** Static $k=8$ SARA baseline (Mean F1 = 0.3950, 1,758.5 tokens).
4. **Primary Evaluation Metric:** Test-set Token F1 and context token savings vs static $k=8$.

> **Handoff Conclusion:** Week 4 model training may proceed because the automatic target-generation methodology is frozen. Human validation remains an independent pending qualitative-validation task.

---

## 17. Artifact Manifest

All artifacts produced during Week 3 are cataloged below:

| Artifact Relative Path | Description / Purpose | Integrity Status |
|:---|:---|:---|
| `results/week3/processed/week2_input_manifest.json` | Immutable input manifest of Week 2 response surface | VERIFIED |
| `results/week3/processed/dev_quality_oracle.csv` | Per-query deployable quality oracle and $\epsilon$-targets | VERIFIED |
| `results/week3/processed/quality_oracle_distribution.csv` | Target distribution across $\mathcal{K}_{\text{alloc}}$ | VERIFIED |
| `results/week3/processed/per_query_heterogeneity.csv` | Per-query F1 range, gain, and tie-aware status | VERIFIED |
| `results/week3/processed/tie_analysis.csv` | Comprehensive tie breakdown for all $k$ budgets | VERIFIED |
| `results/week3/processed/oracle_gain_attribution.csv` | Headroom gain attribution by target budget | VERIFIED |
| `results/week3/processed/epsilon_summary.csv` | Epsilon sweep statistical summary | VERIFIED |
| `results/week3/processed/selected_epsilon.json` | Frozen operating point $\epsilon^* = 0.01$ metadata | VERIFIED |
| `results/week3/processed/feature_signal_analysis.csv` | Spearman rank correlation of retrieval features | VERIFIED |
| `results/week3/processed/edge_cases.json` | Edge case verification catalog | VERIFIED |
| `results/week3/processed/week3_summary.json` | Master aggregated summary JSON | VERIFIED |
| `results/week3/processed/week3_gate.json` | Week 3 Go / No-Go decision gate record | VERIFIED |
| `results/week3/human_audit/human_scoring_template.csv` | Blinded human audit scoring template (150 items) | READY (PENDING) |
| `results/week3/human_audit/blinding_map.json` | Blinded system-to-$k$ mapping dictionary | SECURED |
| `results/week3/week3_final_scientific_report.md` | Final scientific report (Publication format) | COMPLETED |

---

## 18. Reproducibility / Validation Status

### Execution Command

To regenerate all Week 3 analyses, tables, gate files, and figures from frozen Week-2 artifacts:

```bash
bash scripts/week3/01_oracle_analysis.sh
bash scripts/week3/02_epsilon_analysis.sh
bash scripts/week3/03_human_audit.sh
bash scripts/week3/04_generate_report.sh
```

### Automated Test Suite Status

Run pytest validation:

```bash
pytest tests/week3/ -v
```

**Result:** `12 passed in 0.43s` (100% passing).
