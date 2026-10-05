# Week 0 Scientific & Implementation Audit

**Project:** Query-Conditioned Context Allocation (QCCA) for SARA  
**Subject:** Methodological, Mathematical, and Provenance Audit of Week 0 Results  
**Audit Date:** October 2026  
**Auditor:** Scientific Pair-Programming & Systems Review  
**Status:** COMPLETE — Strict Audit Decision Rendered  

---

## Executive Summary

This document reports a comprehensive, line-by-line scientific and implementation audit of all numerical claims, mathematical formulations, and statistical assertions reported for **Week 0: Dataset + RAG Evidence Understanding / Motivation Study**.

### Critical Audit Finding:
- **Part A (Dataset & BM25 Retrieval Characterization):** **GENUINELY EXECUTED AND VERIFIED.** Computed directly over official QASPER train (772 papers, 2,088 queries) and dev splits (86 papers, 231 queries) and true top-10 BM25 retrieval runs. All paper counts, question lengths, document lengths, and retrieval statistics are reproducibly confirmed from raw data.
- **Part B (Layer-Wise Hidden State Evolution) & Part C (Attention & Ablation Utilization):** **INVALID AS EMPIRICAL LM FINDINGS (SYNTHETIC SIMULATION).** In the initial pipeline execution (`scripts/week0/run_week0_analysis.py`), execution ran without the `--run-gpu` flag. As a result, the code executed deterministic offline mock simulations (`run_mock_layer_analysis` and `run_mock_utilization_analysis`) utilizing hardcoded sinusoidal and exponential formulas with synthetic Gaussian noise. **Zero forward passes through Mistral-7B or SARA were executed for the reported numbers.**
- **Action Taken:** In accordance with strict scientific integrity rules, all simulated Part B and Part C numerical results (e.g., $N_{\text{eff}} = 2.48$, Gini $= 0.624$, Layer-16 similarity $= 0.4510$, Pearson $r = 0.5070$, $p < 10^{-6}$) are marked **INVALID / RED**, designated as unverified placeholder smoke-tests, and stripped of empirical claims across all Week 0 reports.

---

## 1. Provenance & Execution Verification Table

The following table documents the exact provenance of every major metric previously reported in Week 0:

| Metric Name | Previously Reported Value | Source File | Source Function | Input Data Artifact | Real Model Inference? | Audit Verdict |
|---|---|---|---|---|---|---|
| **Dev Paper Count** | 86 papers | `src/week0/dataset_adapter.py` | `QASPERAdapter.load_split` | `QASPER_dev.jsonl` | N/A (Data) | **GREEN (Verified)** |
| **Dev Question Count** | 231 questions | `src/week0/dataset_adapter.py` | `QASPERAdapter.load_split` | `QASPER_dev.jsonl` | N/A (Data) | **GREEN (Verified)** |
| **Train Paper Count** | 772 papers | `src/week0/dataset_adapter.py` | `QASPERAdapter.load_split` | `QASPER_train_split.jsonl` | N/A (Data) | **GREEN (Verified)** |
| **Train Question Count** | 2,088 questions | `src/week0/dataset_adapter.py` | `QASPERAdapter.load_split` | `QASPER_train_split.jsonl` | N/A (Data) | **GREEN (Verified)** |
| **Mean Document Length (Dev)** | 4,228.1 words | `src/week0/dataset_analyzer.py` | `analyze_dataset` | `qasper-train.arrow` / text chunks | N/A (Data) | **GREEN (Verified)** |
| **Mean Question Length (Dev)** | 8.39 words | `src/week0/dataset_analyzer.py` | `analyze_dataset` | `QASPER_dev.jsonl` | N/A (Data) | **GREEN (Verified)** |
| **Mean Gold Evidence Count (Dev)** | 1.64 paragraphs | `src/week0/dataset_analyzer.py` | `analyze_dataset` | `QASPER_dev.jsonl` | N/A (Data) | **GREEN (Verified)** |
| **Single-Paragraph Evidence %** | 67.10% (of answerable) | `src/week0/dataset_analyzer.py` | `analyze_dataset` | `QASPER_dev.jsonl` | N/A (Data) | **GREEN (Verified)** |
| **BM25 Recall_any@2 (Dev)** | 51.52% | `src/week0/dataset_analyzer.py` | `analyze_dataset` | `dev_retrieval.jsonl` | N/A (BM25) | **YELLOW (Verified, Renamed)** |
| **BM25 Recall_any@4 (Dev)** | 70.13% | `src/week0/dataset_analyzer.py` | `analyze_dataset` | `dev_retrieval.jsonl` | N/A (BM25) | **YELLOW (Verified, Renamed)** |
| **BM25 Recall_any@6 (Dev)** | 80.95% | `src/week0/dataset_analyzer.py` | `analyze_dataset` | `dev_retrieval.jsonl` | N/A (BM25) | **YELLOW (Verified, Renamed)** |
| **BM25 Recall_any@10 (Dev)** | 87.01% | `src/week0/dataset_analyzer.py` | `analyze_dataset` | `dev_retrieval.jsonl` | N/A (BM25) | **YELLOW (Verified, Renamed)** |
| **BM25 Recall_all@2 (Dev)** | 37.23% (Newly computed) | `src/week0/dataset_analyzer.py` | `analyze_dataset` | `dev_retrieval.jsonl` | N/A (BM25) | **GREEN (Verified)** |
| **BM25 Recall_all@4 (Dev)** | 55.84% (Newly computed) | `src/week0/dataset_analyzer.py` | `analyze_dataset` | `dev_retrieval.jsonl` | N/A (BM25) | **GREEN (Verified)** |
| **BM25 Recall_all@6 (Dev)** | 70.56% (Newly computed) | `src/week0/dataset_analyzer.py` | `analyze_dataset` | `dev_retrieval.jsonl` | N/A (BM25) | **GREEN (Verified)** |
| **BM25 Recall_all@10 (Dev)** | 80.09% (Newly computed) | `src/week0/dataset_analyzer.py` | `analyze_dataset` | `dev_retrieval.jsonl` | N/A (BM25) | **GREEN (Verified)** |
| **Layer-0 Mean Query-Passage Sim** | 0.3512 | `src/week0/layer_analyzer.py` | `run_mock_layer_analysis` | `selected_examples.json` (40 Qs) | **NO (Synthetic math.sin)** | **RED (INVALID)** |
| **Layer-16 Mean Query-Passage Sim** | 0.4510 | `src/week0/layer_analyzer.py` | `run_mock_layer_analysis` | `selected_examples.json` (40 Qs) | **NO (Synthetic math.sin)** | **RED (INVALID)** |
| **Layer-31 Mean Query-Passage Sim** | 0.3642 | `src/week0/layer_analyzer.py` | `run_mock_layer_analysis` | `selected_examples.json` (40 Qs) | **NO (Synthetic math.sin)** | **RED (INVALID)** |
| **Evidence Separation Gap (L16, L31)**| +0.0749, +0.1427 | `src/week0/layer_analyzer.py` | `run_mock_layer_analysis` | `selected_examples.json` (40 Qs) | **NO (Synthetic ev_boost)** | **RED (INVALID)** |
| **Layer Separation p-values** | $p < 0.001, p < 0.0001$ | `src/week0/layer_analyzer.py` | `run_mock_layer_analysis` | `selected_examples.json` (40 Qs) | **NO (Simulated data)** | **RED (INVALID)** |
| **Effective Passages ($N_{\text{eff}}$)** | 2.48 ± 0.82 | `src/week0/evidence_utilization.py`| `run_mock_utilization_analysis` | `selected_examples.json` (40 Qs) | **NO (Simulated exp decay)**| **RED (INVALID)** |
| **Utilization Gini Coefficient** | 0.624 ± 0.142 | `src/week0/evidence_utilization.py`| `run_mock_utilization_analysis` | `selected_examples.json` (40 Qs) | **NO (Simulated exp decay)**| **RED (INVALID)** |
| **Top-3 Utilization Share** | 82.4% | `src/week0/evidence_utilization.py`| `run_mock_utilization_analysis` | `selected_examples.json` (40 Qs) | **NO (Simulated exp decay)**| **RED (INVALID)** |
| **Ranks 7–10 Importance Share** | < 4.7% | `src/week0/evidence_utilization.py`| `run_mock_utilization_analysis` | `selected_examples.json` (40 Qs) | **NO (Simulated exp decay)**| **RED (INVALID)** |
| **Attn ↔ Ablation Pearson $r$** | 0.5070 ($p < 10^{-6}$) | `src/week0/evidence_utilization.py`| `run_mock_utilization_analysis` | `selected_examples.json` (40 Qs) | **NO (Simulated arrays)** | **RED (INVALID)** |
| **Attn ↔ Ablation Spearman $\rho$**| 0.6538 ($p < 10^{-6}$) | `src/week0/evidence_utilization.py`| `run_mock_utilization_analysis` | `selected_examples.json` (40 Qs) | **NO (Simulated arrays)** | **RED (INVALID)** |
| **Attn ↔ Gold Evidence Pearson $r$**| 0.4466 ($p < 10^{-6}$) | `src/week0/evidence_utilization.py`| `run_mock_utilization_analysis` | `selected_examples.json` (40 Qs) | **NO (Simulated arrays)** | **RED (INVALID)** |
| **BM25 ↔ Gold Evidence Pearson $r$**| 0.1169 ($p = 0.0193$) | `src/week0/evidence_utilization.py`| `run_mock_utilization_analysis` | `selected_examples.json` (40 Qs) | **NO (Simulated arrays)** | **RED (INVALID)** |

---

## 2. Model Input Format & Causal Mask Verification

### Prompt Layout Specification
In `src/week2/sweep_runner.py::build_evaluation_prompt`, the standard evaluation prompt for $k=10$ full-text passages is structured as follows:

```text
<s> [INST] Answer my questions based on the given context.
---
## Context
Document 1. <passage 1 text>

Document 2. <passage 2 text>
...
Document 10. <passage 10 text>
---
## Your Task
Answer the following question in a succinct manner. Use a single phrase or a short sentence if possible.
Question: <question text>
Your Answer: [/INST]
```

### Exact Token-Span Mapping for a Concrete Example
To audit token layout and causal visibility, a human-readable token-span map was generated using the Mistral-7B tokenizer for development query `2201` (Paper `798`):
- **Artifact:** [`results/week0/layer_analysis/token_span_map_example.json`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week0/layer_analysis/token_span_map_example.json)
- **Total Prompt Tokens:** 2,355 tokens.

| Semantic Component | Prompt Text Prefix | Token Range $[t_{\text{start}}, t_{\text{end}})$ | Token Count | Relative Position |
|---|---|---|---|---|
| **Instruction Prefix** | `<s> [INST] Answer my questions...` | $[0, 19)$ | 19 tokens | Head |
| **Passage $D_1$ (Rank 1)** | `Document 1. We begin by describing...` | $[19, 229)$ | 210 tokens | Precedes Query |
| **Passage $D_2$ (Rank 2)** | `Document 2. The classifier relying on...` | $[231, 315)$ | 84 tokens | Precedes Query |
| **Passage $D_3$ (Rank 3)** | `Document 3. Using Gaussian Processes...` | $[317, 554)$ | 237 tokens | Precedes Query |
| **Passage $D_4$ (Rank 4)** | `Document 4. Our work advances the...` | $[556, 818)$ | 262 tokens | Precedes Query |
| **Passage $D_5$ (Rank 5)** | `Document 5. We argue that this makes...` | $[820, 1072)$ | 252 tokens | Precedes Query |
| **Passage $D_6$ (Rank 6)** | `Document 6. The central concept of...` | $[1074, 1274)$ | 200 tokens | Precedes Query |
| **Passage $D_7$ (Rank 7)** | `Document 7. $ The above integrals...` | $[1276, 1529)$ | 253 tokens | Precedes Query |
| **Passage $D_8$ (Rank 8)** | `Document 8. Figure 1 and Table 4...` | $[1531, 1787)$ | 256 tokens | Precedes Query |
| **Passage $D_9$ (Rank 9)** | `Document 9. into the conversational...` | $[1789, 2028)$ | 239 tokens | Precedes Query |
| **Passage $D_{10}$ (Rank 10)**| `Document 10. To assess and compare...` | $[2030, 2301)$ | 271 tokens | Precedes Query |
| **Task Specification** | `## Your Task\nAnswer the following...` | $[2304, 2331)$ | 27 tokens | Context-Query Bridge |
| **Query ($Q$)** | `Question: Why is a Gaussian process...`| $[2331, 2347)$ | 16 tokens | Follows Context |
| **Answer Closure** | `Your Answer: [/INST]` | $[2348, 2355)$ | 7 tokens | Generation Boundary |

### Mathematical Validity Under Mistral's Causal Mask
Mistral-7B is a decoder-only autoregressive language model enforcing a strictly lower-triangular causal attention mask:
$$M_{i, j} = \begin{cases} 1 & \text{if } j \le i \\ 0 & \text{if } j > i \end{cases}$$

**Consequences for Analysis:**
1. **Passages CANNOT attend to the Query:** Passage tokens $D_1 \dots D_{10}$ occupy positions $19 \dots 2301$, whereas Query tokens occupy positions $2331 \dots 2347$. Because $j > i$ for all query tokens relative to passage tokens, $M_{i, j} \equiv 0$. The hidden states of passage tokens across all 32 layers are **completely query-agnostic** (pure context encodings).
2. **The Query DOES attend to Passages:** Because query tokens appear after all passages ($2331 > 2301$), query tokens attend over all passage tokens.
3. **Terminology Rule:** It is physically and mathematically impossible for "passages to attend to the query". All attention-based metrics must be designated as **"query-to-passage attention"** and interpreted as an **"attention-based passage utilization proxy"**.

---

## 3. Attention Metric Mathematical Audit

### Exact Statistic Definition
The attention metric implemented in `src/week0/evidence_utilization.py::extract_causal_passage_attention` is defined as:
- **Source Token Positions:** Query tokens $t \in I(Q) = [q_{\text{start}}, q_{\text{end}})$.
- **Target Key Positions:** Passage tokens $k \in I(D_i) = [p_{\text{start}}, p_{\text{end}})$.
- **Layer Selection:** $l \in \{0, 4, 8, 16, 24, 31\}$.
- **Head Aggregation:** Uniform arithmetic mean across all $H=32$ heads:
  $$\bar{A}^{(l)}_{t, k} = \frac{1}{H} \sum_{h=1}^H A^{(l, h)}_{t, k}$$
- **Token Aggregation & Length Normalization:**
  $$\text{Raw Mass}(D_i) = \sum_{t \in I(Q)} \sum_{k \in I(D_i)} \bar{A}^{(l)}_{t, k}$$
  $$\bar{A}_i^{(l)} = \frac{\text{Raw Mass}(D_i)}{|I(Q)| \cdot |I(D_i)|}$$
- **Passage Attention Share ($s_i^{(l)}$):**
  $$s_i^{(l)} = \frac{\bar{A}_i^{(l)}}{\sum_{j=1}^{10} \bar{A}_j^{(l)}}$$
- **Padding:** Evaluated at batch size 1 (unpadded).
- **Execution Timing:** Prompt processing only (does not include generated answer tokens).

### Defensible Terminology
The earlier draft referred to this metric as "causal attention diagnostics." Merely operating inside a causal decoder does not make attention a causal estimator of evidence utility. The term has been replaced with:  
**"attention-based passage utilization proxy"**.

---

## 4. Leave-One-Passage-Out Importance ($\Delta_i$) Audit

### Mathematical Definition
For gold answer string $y_{\text{gold}}$, prompt $x(q, D_1 \dots D_{10})$, and ablated prompt $x(q, D_{\setminus \{i\}})$:
$$\Delta_i = \log P(y_{\text{gold}} \mid q, D_1 \dots D_{10}) - \log P(y_{\text{gold}} \mid q, D_{\setminus \{i\}})$$
where $\log P(y_{\text{gold}} \mid \cdot)$ is computed via teacher forcing under causal language modeling:
$$\log P(y_{\text{gold}} \mid x) = \sum_{t=1}^{|y_{\text{gold}}|} \log P(y_t \mid x, y_{<t})$$

### Important Clarifications:
1. **Total vs Per-Token Log Probability:** $\log P(y_{\text{gold}} \mid x)$ is the **total answer log-probability** (sum of target token log-probabilities). Because $y_{\text{gold}}$ is identical between full and ablated prompts, the length normalizer $|y_{\text{gold}}|$ is a constant scalar factor for any given query.
2. **Sign Interpretation:**
   - $\Delta_i > 0$: Removing passage $D_i$ decreases gold-answer likelihood $\implies$ **beneficial evidence**.
   - $\Delta_i \approx 0$: Removing passage $D_i$ produces no noticeable change ($|\Delta_i| \le \epsilon$) $\implies$ **neutral / redundant / ignored**.
   - $\Delta_i < 0$: Removing passage $D_i$ *increases* gold-answer likelihood $\implies$ **negative interference / distraction**.
3. **Audit of Clamping Bug in Mock Logic:**
   In `src/week0/evidence_utilization.py::run_mock_utilization_analysis`, the code contained `drop = float(max(0.0, base_drop + ev_drop + noise))`, which artificially clamped all negative values to zero. In a genuine model evaluation, negative $\Delta_i$ values naturally occur when passages contain conflicting entities or distractor tokens. **Negative values must be preserved and reported.**

---

## 5. Concentration Metrics ($N_{\text{eff}}$, Gini, Top-$k$) Audit

### The Mathematical Flaw in Naive Normalization
The previous code used:
$$p_i = \frac{\Delta_i}{\sum_{j=1}^n \Delta_j}$$
When $\Delta_i < 0$, $p_i$ becomes negative. Negative values violate the probability simplex ($\sum p_i = 1, p_i \ge 0$), causing undefined operations in Shannon entropy ($p_i \ln p_i$) and invalid Gini coefficients.

### Three Formulations Evaluated:

| Formulation | Mathematical Definition | Property | Recommended Use Case |
|---|---|---|---|
| **A. Positive Contribution (Useful Evidence)** | $c_i = \max(\Delta_i, 0)$<br>$p_i = \frac{c_i}{\sum c_j}$ | Discards negative drops; quantifies how beneficial evidence mass is allocated. If all $\Delta_i \le 0$, $N_{\text{eff}} = 0$. | **Primary / Most Interpretable** for evidence allocation in RAG. |
| **B. Absolute Sensitivity (Perturbation Impact)** | $c_i = |\Delta_i|$<br>$p_i = \frac{|\Delta_i|}{\sum |\Delta_j|}$ | Treats positive and negative interference equally; measures total model sensitivity. | Diagnostic for model instability or distraction susceptibility. |
| **C. Softmax Transformation** | $p_i = \frac{\exp(\Delta_i / \tau)}{\sum \exp(\Delta_j / \tau)}$ | Heavily sensitive to arbitrary temperature $\tau$; assigns positive probability to strongly harmful passages. | Not recommended for evidence allocation. |

### Corrective Code Implementation
`src/week0/evidence_utilization.py::compute_concentration_metrics` has been updated to explicitly support `mode="positive"` and `mode="absolute"`, with clean edge-case handling for degenerate inputs ($\sum c_i = 0 \implies N_{\text{eff}} = 0, G = 0$). Unit tests covering signed inputs and edge cases were added and verified.

---

## 6. Gold Evidence Matching Audit

### Matching Algorithm Specification
In `src/week0/dataset_analyzer.py::check_passage_matches_evidence`, a retrieved passage is marked as evidence if:
1. **Highlighted Span Substring:** Any annotated highlighted span (length $> 15$ characters) appears verbatim as a substring of the lowercase passage text; OR
2. **Content Word Overlap Recall:** At least $30\%$ of the content words in an annotated gold evidence paragraph appear in the passage text; OR
3. **Passage Content Word Precision:** At least $50\%$ of the passage content words match the gold evidence paragraph and match count $\ge 10$.

### Scientific Critique & Naming
- This is a heuristic token-overlap/substring hybrid.
- QASPER evidence paragraphs represent annotator-labeled minimal sufficient evidence; non-annotated passages often contain relevant background, definitions, or context.
- **Terminology Rule:** Calling non-matching passages "distractors" implies proof of model degradation. Across all reports, "distractor" has been renamed to **"non-annotated retrieved passage"**.

---

## 7. Statistical Testing & Pseudo-Replication Audit

### The Pseudo-Replication Error in Previous Reports
Previous reports cited correlation p-values of $p < 10^{-6}$ across $N=400$ passage observations.
- **The Error:** The 400 passages were drawn from 40 queries across only $\approx 20$ papers. Passages within the same prompt compete for softmax attention and are conditionally dependent.
- **Consequence:** Treating $N=400$ passages as independent identically distributed samples is **statistical pseudo-replication**, which drastically underestimates standard errors and produces spurious significance claims.
- **Correction:** In all code and reports, correlation tables now explicitly report both $N_{\text{passages}}$ and $N_{\text{queries}}$, and label uncorrected p-values as exploratory. Future live evaluations must compute query-level paired bootstrap confidence intervals.

---

## 8. Part A Independent Recomputation Audit

All Part A numbers were recomputed from scratch over official QASPER splits:

| Metric | Train Split | Dev Split | Verification Status |
|---|---|---|---|
| **Papers** | 772 | 86 | Exact match |
| **Questions** | 2,088 | 231 | Exact match |
| **Questions / Paper** | $2.70 \pm 1.84$ (Median: 2.0) | $2.69 \pm 1.76$ (Median: 2.0) | Exact match |
| **Mean Doc Length** | 3,694.3 words (Median: 3,526.5) | 4,228.1 words (Median: 3,617.5) | Exact match |
| **Mean Question Length** | 8.39 words | 8.39 words | Exact match |
| **Mean Answer Length** | 13.65 words | 14.45 words | Exact match |
| **Extractive Questions** | 56.47% (1,179 / 2,088) | 57.58% (133 / 231) | Exact match |
| **Free-form / Abstractive** | 26.34% (550 / 2,088) | 25.97% (60 / 231) | Corrected in CSV summary |
| **Yes / No Questions** | 17.19% (359 / 2,088) | 16.45% (38 / 231) | Exact match |
| **Evidence Paras Mean** | 1.78 paragraphs | 1.64 paragraphs | Exact match |
| **Single-Paragraph Evidence** | 62.4% (Train) | 67.1% (Dev answerable) | Exact match |
| **Two-Paragraph Evidence** | 22.8% (Train) | 22.1% (Dev answerable) | Exact match |
| **Three+ Paragraph Evidence** | 14.8% (Train) | 10.8% (Dev answerable) | Exact match |

### Disambiguating BM25 Retrieval Metrics
To prevent ambiguity, retrieval recall is now split into two formal definitions:
- **`Recall_any@k`:** At least one annotated evidence paragraph is present in top-$k$ retrieved chunks.
- **`Recall_all@k`:** All annotated evidence paragraphs are present in top-$k$ retrieved chunks.
- **`Mean_Coverage@k`:** The average fraction of annotated evidence paragraphs retrieved in top-$k$.

| Retrieval Depth ($k$) | `Recall_any@k` (%) | `Recall_all@k` (%) | `Mean_Coverage@k` (%) |
|---|---|---|---|
| **$k=1$** | 33.33% | 22.94% | 28.99% |
| **$k=2$** | 51.52% | 37.23% | 45.83% |
| **$k=4$** | 70.13% | 55.84% | 66.52% |
| **$k=6$** | 80.95% | 70.56% | 79.85% |
| **$k=8$** | 84.85% | 77.06% | 85.53% |
| **$k=10$** | 87.01% | 80.09% | 88.27% |

*Finding:* At $k=2$, while $51.52\%$ of queries retrieve at least one evidence paragraph, only $37.23\%$ retrieve complete evidence. Expanding to $k=6$ increases complete evidence retrieval to $70.56\%$.

---

## 9. Final Audit Classification: GREEN / YELLOW / RED

### GREEN: Verified & Scientifically Defensible
- **QASPER Corpus Profile:** Document length distributions, question lengths, answer category proportions, and evidence paragraph counts.
- **BM25 Evidence Recall Disambiguation:** Both `Recall_any@k` and `Recall_all@k` curves across $k \in \{1, 2, 4, 6, 8, 10\}$.
- **Evidence Sparsity Finding:** $67.1\%$ of answerable dev questions require only a single evidence paragraph, while $32.9\%$ require multi-paragraph synthesis.
- **Prompt Token Mapping:** Verification of causal decoder token order (Query follows Context).
- **Unit Test Coverage:** All 13 unit tests for adapters, metrics, signed ablations, and synthetic attention aggregation pass 100%.

### YELLOW: Verified Implementation, Cautious Interpretation Required
- **Gold-Evidence Matching:** Uses token-overlap and substring heuristics; non-matching passages should be referred to as "non-annotated retrieved passages" rather than "distractors".
- **Attention-Based Utilization Proxy:** Normalized query-to-passage attention reflects geometric attention mass, not causal sufficiency.
- **BM25 Retrieval Saturation:** While `Recall_any` gains taper after $k=6$ ($+3.9\%$ from $6 \rightarrow 8$, $+2.2\%$ from $8 \rightarrow 10$), `Recall_all` continues to gain $+9.5\%$ from $6 \rightarrow 10$.

### RED: Invalid / Unverified — Removed From Week 0 Findings
- **Simulated Layer Similarity Values:** Layer-0 ($0.3512$), Layer-16 ($0.4510$), Layer-31 ($0.3642$) came from mock sinusoidal generators. **Must not be reported as model findings.**
- **Simulated Evidence Separation:** $+0.0035$ at L0, $+0.0749$ at L16, $+0.1427$ at L31 came from mock boost logic. **Must not be reported as model findings.**
- **Simulated Concentration Statistics:** $N_{\text{eff}} = 2.48 \pm 0.82$, Gini $= 0.624 \pm 0.142$, top-3 utilization $= 82.4\%$ came from simulated exponential decay. **Must not be reported as model findings.**
- **Simulated Correlation Coefficients:** Pearson $r=0.5070$, Spearman $\rho=0.6538$, and $p < 10^{-6}$ were computed on synthetic arrays. **Must not be reported as model findings.**
- **Overclaim Language:** Statements claiming Mistral "understands", "proves optimal $k$", or that 10 passages "penalize the reader" have been purged.

---

## 10. Summary Findings

### Safe Week 0 Findings
1. **Evidence Requirements are Heterogeneous:** In QASPER, $67.1\%$ of answerable questions are satisfied by a single evidence paragraph, while $32.9\%$ require multi-paragraph evidence.
2. **Retrieval Depth Exhibits Differential Gains:** BM25 `Recall_any@k` rises from $51.52\%$ at $k=2$ to $80.95\%$ at $k=6$, but complete evidence (`Recall_all@k`) reaches only $70.56\%$ at $k=6$ and $80.09\%$ at $k=10$.
3. **Causal Directionality is Strictly Query-to-Passage:** In standard decoder-only RAG prompt formatting, passages precede the query; passage hidden states are query-agnostic, and attention flows from the query back to the passages.

### Exploratory Findings
- Heuristic evidence matching suggests that top-4 BM25 chunks contain the majority of first-hit evidence paragraphs ($70.13\%$), while deeper ranks provide diminishing marginal returns for single-paragraph queries.

### Invalidated Findings (Purged)
- All numerical estimates of Mistral-7B layer similarity, representation drift, attention concentration ($N_{\text{eff}}$), Gini coefficient, and leave-one-out likelihood drops are invalidated due to execution defaulting to offline mock simulation.

### Recommended Next Experiment
- **Do NOT execute the full 231-query GPU expansion yet.**
- First execute a clean, verified GPU smoke-test over a small sample of 5–10 development queries with actual Mistral-7B forward passes, verifying that layer extraction and teacher-forced leave-one-out log-probability computation run end-to-end without memory issues or numerical instability. Only then decide whether a full 40-query or 231-query run is scientifically necessary.
