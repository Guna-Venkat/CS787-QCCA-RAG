# Methodological Audit: Dual Oracle Definitions & Epsilon Target Formulation

**Project:** Query-Conditioned Context Allocator (QCCA)  
**Parent Framework:** SARA (Selective and Adaptive Retrieval-augmented Generation with Context Compression)  
**Phase:** Week 3 (Oracle, Epsilon Target, Heterogeneity & Human Validation)  
**Date:** 2026-09-30  
**Status:** Methodological Clarification & Refinement Documented and Locked

---

## 1. Background & The Original Plan v2.1 Formulation

In *Plan v2.1* (Section 4.1, Equation 365), the epsilon-constrained optimal evidence budget was formally defined as:
$$k^*_\epsilon(q) = \min \left\{ k \in \mathcal{K}_{\text{alloc}} \;\Big|\; \text{F1}(q, k) \ge \max_{j \in \mathcal{K}_{\text{sweep}}} \text{F1}(q, j) - \epsilon \right\}$$
where:
$$\mathcal{K}_{\text{sweep}} = \{0, 2, 4, 5, 6, 8, 10\}$$
$$\mathcal{K}_{\text{alloc}} = \{2, 4, 5, 6, 8\}$$

The intent of this formulation was to identify the minimum acceptable evidence budget $k \in \mathcal{K}_{\text{alloc}}$ that retains performance within a margin $\epsilon$ of the best achievable quality across all empirical conditions.

---

## 2. Identified Methodological Inconsistency

During the empirical pre-flight audit of the frozen Week-2 fixed-$k$ response matrix ($N = 231$ questions across 86 papers), an asymmetry between the reference space ($\mathcal{K}_{\text{sweep}}$) and the target action space ($\mathcal{K}_{\text{alloc}}$) was identified:

1. **Model & Checkpoint Heterogeneity ($k=10$):**
   - Condition $k=10$ represents uncompressed **Standard RAG** using full natural-language context and a dedicated RAG LoRA checkpoint (`rag_qasper_lora_r16_seed42`).
   - Conditions $k \in \{2, 4, 5, 6, 8\}$ represent deployable **SARA** configurations using the SARA context compressor and trained projector (`sara_qasper_proj_lr5e4_seed42`).
   - Consequently, $k=10$ is not a valid operational state of the SARA projector; it is an external ceiling benchmark.

2. **The Infeasibility Dilemma:**
   - When evaluating $\max_{j \in \mathcal{K}_{\text{sweep}}} \text{F1}(q, j)$ as the reference quality, for **31 out of 231 questions (13.42%)**, the global maximum occurs strictly outside $\mathcal{K}_{\text{alloc}}$:
     - For **20 questions**, Standard RAG ($k=10$) uniquely outperforms all SARA configurations.
     - For **14 questions**, Pure Compression ($k=0$) uniquely outperforms all $k \ge 2$ configurations (e.g., short abstractive queries where retrieved passages induce distraction).
     - For **3 questions**, both $k=0$ and $k=10$ strictly exceed $\max_{k \in \mathcal{K}_{\text{alloc}}} \text{F1}(q, k)$.
   - Under the literal Plan v2.1 formulation at $\epsilon = 0.00$, the set $\{ k \in \mathcal{K}_{\text{alloc}} \mid \text{F1}(q, k) \ge \max_{j \in \mathcal{K}_{\text{sweep}}} \text{F1}(q, j) \}$ is **empty** for 31 questions.
   - Even at $\epsilon = 0.05$, **27 questions (11.69%)** remain infeasible.

3. **Consequence for Supervised Target Construction:**
   - If an allocator is trained in Week 4 to predict an action in $\mathcal{K}_{\text{alloc}} = \{2, 4, 5, 6, 8\}$, assigning an artificial fallback (or dropping 13.4% of queries) introduces either arbitrary label bias or sample selection bias.
   - An allocator operating inside SARA cannot physically select $k=10$ (which requires disabling compression and swapping model weights). Penalizing an allocator for failing to achieve the quality of a different model architecture violates the principle of controlled budget allocation.

---

## 3. Methodological Resolution & Final Specification

To maintain complete scientific integrity while ensuring well-posed supervised targets for QCCA, we adopt the following definitive two-tier standard:

### A. Primary QCCA Target Space: SARA-Internal Formulation
The primary target action space is strictly:
$$\mathcal{K}_{\text{alloc}} = \{2, 4, 5, 6, 8\}$$

1. **Quality Oracle (Deployable SARA Upper Bound):**
   $$F1^*_{\text{alloc}}(q) = \max_{k \in \mathcal{K}_{\text{alloc}}} \text{F1}(q, k)$$
   $$k^*_{\text{qual}}(q) = \arg\max_{k \in \mathcal{K}_{\text{alloc}}} \text{F1}(q, k) \quad \text{(smallest } k \text{ on tie)}$$

2. **Primary Epsilon Target (Efficiency-Aware Target for QCCA):**
   $$k^*_\epsilon(q) = \min \left\{ k \in \mathcal{K}_{\text{alloc}} \;\Big|\; \text{F1}(q, k) \ge F1^*_{\text{alloc}}(q) - \epsilon \right\}$$
   - **Guaranteed Feasibility:** Because $F1^*_{\text{alloc}}(q)$ is defined over $\mathcal{K}_{\text{alloc}}$, there is **always at least one feasible action** in $\mathcal{K}_{\text{alloc}}$ for every question (namely $k^*_{\text{qual}}(q)$ itself at $\epsilon = 0$, or smaller $k$ when within $\epsilon$).
   - Feasibility is **100.0%** across all 231 development questions for all candidate $\epsilon \in \{0.00, 0.01, 0.02, 0.05\}$.
   - No downstream QCCA training target relies on or suffers from infeasibility artifacts.

### B. Global-Sweep Reference: Offline Diagnostic Only
The original global-sweep reference over $\mathcal{K}_{\text{sweep}} = \{0, 2, 4, 5, 6, 8, 10\}$ is preserved in its entirety as an **offline theoretical diagnostic**:
$$F1^*_{\text{sweep}}(q) = \max_{j \in \mathcal{K}_{\text{sweep}}} \text{F1}(q, j)$$

We track and report two diagnostic headroom gaps:
1. **RAG Ceiling Gap:**
   $$\Delta_{\text{RAG}}(q) = \text{F1}(q, 10) - F1^*_{\text{alloc}}(q)$$
   Measures where full uncompressed context + uncompressed LoRA outperforms the compressed SARA architecture.
2. **Pure-Compression Floor Gap:**
   $$\Delta_{\text{Comp}}(q) = \text{F1}(q, 0) - F1^*_{\text{alloc}}(q)$$
   Identifies queries where zero context is superior to any retrieved evidence, isolating retrieval degradation.

---

## 4. Summary of Principles

| Dimension | Primary QCCA Target | Global Sweep Diagnostic |
|---|---|---|
| **Action Space** | $\mathcal{K}_{\text{alloc}} = \{2, 4, 5, 6, 8\}$ | $\mathcal{K}_{\text{sweep}} = \{0, 2, 4, 5, 6, 8, 10\}$ |
| **Reference Quality** | $F1^*_{\text{alloc}}(q) = \max_{k \in \mathcal{K}_{\text{alloc}}} \text{F1}(q, k)$ | $F1^*_{\text{sweep}}(q) = \max_{j \in \mathcal{K}_{\text{sweep}}} \text{F1}(q, j)$ |
| **Feasibility on Dev** | **100.0%** (231 / 231) | 86.6% ($\epsilon=0.0$), 88.3% ($\epsilon=0.05$) |
| **Role in Project** | Target labels for QCCA training in Weeks 4–5 | Diagnostic bounding of SARA representation limits |
| **Model Invariance** | Strictly within SARA architecture | Across SARA, pure compression, and standard RAG |

This clarification ensures that QCCA is trained to solve the exact problem it is designed for: **allocating natural-language context budgets within the deployable SARA system**.
