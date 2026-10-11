# Week 5: Content-Aware Allocation Preflight Report

**Project:** Query-Conditioned Context Allocator (QCCA) in SARA  
**Date:** October 2026  
**Auditor:** SARA Research Engineering Agent  
**Artifact Directory:** `results/week5/preflight/`  
**Decision:** `W5_PREFLIGHT_STATUS = BLOCKED` (Pending Train Response Surface Generation)

---

## 1. Research Question

Week 5 investigates the fundamental information boundary of evidence budget allocation in SARA:
> **"Does directly reading retrieved passage natural-language text provide allocation information that handcrafted query/retrieval features miss?"**

Prior weeks established:
- **Weeks 2–3:** A fixed-budget response surface $F(q, k)$ over $k \in \{2, 4, 5, 6, 8\}$ reveals massive per-query headroom ($\Delta F_1 = +0.0945$ vs static $k=8$) and substantial evidence heterogeneity.
- **Week 4:** 64 pre-generation features (retrieval scores, query complexity, lexical coverage, SFR similarity, and passage diversity) capture coarse stopping signals, but scaling classifier capacity (XGBoost, Random Forests, MLPs) failed to close the oracle gap.
- **Week 5 Hypothesis:** Pre-generation statistical features compress away the fine-grained semantic overlap and nuanced sufficiency signals necessary to identify whether an early budget ($k \in \{2, 4\}$) is already sufficient or whether expansion ($k \in \{6, 8\}$) is required.

---

## 2. Repository State

Following the consolidation of historical nonlinear modeling benchmarks into Week 4:
- All historical modeling code, tests, notebooks, and result tables were cleanly moved into `src/week4/`, `tests/week4/`, `notebooks/week4/`, and `results/week4/model_benchmark/`.
- All legacy references and import paths were redirected to `src.week4`.
- All 65 unit tests in `tests/week4/` pass without failure (**65/65 passed**).
- Weeks 0, 1, 2, 3, and 4 are strictly partitioned and scientifically **FROZEN**.

---

## 3. Split Integrity

An exhaustive audit of the QASPER dataset and manifest file (`Baselines/SARA-main/data/manifests/qasper_split.json`) verified the canonical split dimensions:

| Split | Papers ($N$) | Questions ($N$) | Unique QIDs | Document Overlap vs Train | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **TRAIN** | 772 | 2,088 | 2,088 | — | Fit / Supervision Set |
| **DEV** | 86 | 231 | 231 | 0 papers (0%) | Model & Threshold Selection |
| **TEST** | 411 | 1,309 | 1,309 | 0 papers (0%) | **100% Frozen & Untouched** |

### Split Integrity Guarantees:
1. **Paper Disjointness:** Train and Dev papers are strictly disjoint ($\text{Train} \cap \text{Dev} = \emptyset$).
2. **Test Isolation:** The 411 test papers originate from the official AllenAI QASPER test partition and share zero documents with Train or Dev.
3. **Question Disjointness:** Question IDs join uniquely across all retrieval, response-surface, and oracle tables without collisions.
4. **Recorded Artifact:** Saved to `results/week5/preflight/split_audit.json`.

---

## 4. Train Response-Surface Availability

A systematic repository-wide search was conducted for fixed-$k$ SARA generation outputs on the 2,088 training queries:
- Locations audited: `results/week2/`, `results/week3/`, `results/week4/`, `outputs/`, `Baselines/SARA-main/outputs/`.
- Coverage across budgets $k \in \{2, 4, 5, 6, 8\}$: **0 / 2,088 queries (0.0%)**.

### Verdict: **STATUS C — ABSENT**
- Train fixed-$k$ response-surface outputs have never been generated for the 2,088 training questions.
- Detailed per-question audit saved to `results/week5/preflight/train_response_surface_audit.csv` (all 2,088 rows set to `False`).
- Supervision template created at `results/week5/supervision/train_allocation_supervision.csv` marked `PENDING_FULL_TRAIN_RESPONSE_SURFACE`.

---

## 5. Supervision Integrity

The canonical DEV supervision table was derived directly from the frozen Week-2 fixed-$k$ response matrix (`results/week2/processed/dev_per_query_k_matrix.csv`) using the frozen Week-3 oracle definition ($\epsilon = 0.01$):
- **Best $F_1$:** $\max_{k \in \{2,4,5,6,8\}} F_1(q, k)$
- **Regret:** $R(q, k) = \text{Best } F_1(q) - F_1(q, k) \ge 0$
- **Epsilon Admissibility:** $\mathbb{I}[R(q, k) \le 0.01]$
- **Oracle Budget:** $\min \{k \in \{2,4,5,6,8\} \mid R(q, k) \le 0.01\}$

### Target Distribution Reproduction (DEV Split, $N=231$):
$$\begin{aligned}
k=2&: \mathbf{132} \quad (57.1\%) \\
k=4&: \mathbf{40} \quad (17.3\%) \\
k=5&: \mathbf{17} \quad (7.4\%) \\
k=6&: \mathbf{19} \quad (8.2\%) \\
k=8&: \mathbf{23} \quad (10.0\%)
\end{aligned}$$
**Result:** **100% Bit-Exact Match** with the frozen Week-3 target distribution.  
Saved to `results/week5/supervision/dev_allocation_supervision.csv`.

---

## 6. Week 4 Baseline Reuse

To ensure rigorous benchmarking against cheap pre-generation features:
- All 64 features in `results/week4/feature_discovery/feature_metadata.csv` were verified as `available_pre_generation: True` with `leakage_status: VERIFIED_PRE_GEN`.
- **Locked 5-Feature Core:** `bm25_mean_score`, `lex_coverage_top2`, `sem_sim_mean_top6`, `evidence_query_cluster_span`, `evidence_lexical_overlap_mean`.
- **Baseline Policy Quality:** Balanced Logistic Regression achieves $F_1 = 0.3842$, Context Tokens $= 1,115.9$ ($36.54\%$ token reduction vs static $k=8$).
- Detailed audit documented in `results/week5/preflight/w4_feature_reuse.md`.

---

## 7. Information Ablation Matrix

To isolate the marginal value of raw text semantics, Week 5 freezes four input conditions:

| Condition | Inputs | Scientific Hypothesis / Role |
| :--- | :--- | :--- |
| **I1: Query Only** | $q$ | Tests whether question complexity/syntax alone predicts allocation difficulty. |
| **I2: Query + Features** | $q \oplus \mathbf{x}_{\text{W4}}$ | Semantic query representation combined with cheap retrieval distribution signals. |
| **I3: Query + Passages** | $q \oplus [D_1, \dots, D_{10}]$ | **Core test:** Direct reading of retrieved evidence chunks without tabular shortcuts. |
| **I4: Query + Passages + Features** | $q \oplus [D_1, \dots, D_{10}] \oplus \mathbf{x}_{\text{W4}}$ | Full multimodal fusion testing complementarity of content and retrieval state. |

---

## 8. Target Formulations

Two mathematical target formulations are defined:

1. **T1 — Direct $k$ Classification:**
   - Target: $k^*_{\epsilon=0.01} \in \{2, 4, 5, 6, 8\}$.
   - Loss: Class-weighted Cross-Entropy (accounting for the 57.1% base rate of $k=2$).
   - Diagnostics: Macro-$F_1$, Balanced Accuracy, MAE in $k$.

2. **T2 — Response-Surface / Regret Prediction:**
   - Target: Continuous vector $[R(q, 2), R(q, 4), R(q, 5), R(q, 6), R(q, 8)]$, where $R(q, k) \ge 0$.
   - Loss: Multi-output Huber / Smooth L1 loss.
   - Inference Rule: Select $\min \{k \mid \widehat{R}(q, k) \le 0.01\}$. If no budget meets the threshold, default to $k=8$ (risk-averse fall-back).

---

## 9. Model Architecture Plan

As documented in `results/week5/preflight/model_architecture_plan.md`:
- **Philosophy:** Scientific evaluation of information availability, not model parameter scale. No 7B-parameter LLMs or LoRA fine-tuning for allocation.
- **Selected Backbone:** MiniLM / BGE Cross-Encoder + Shallow Transformer Aggregator (~35M parameters).
- **Encoder:** Shared cross-encoder $h_i = \text{Encoder}(q, D_i)$ for $i \in \{1 \dots 10\}$.
- **Aggregator:** 2-layer Transformer Encoder with learnable rank embeddings $r_1 \dots r_{10}$, attention pooling, and tabular projection layer.
- **Heads:** Dual modular heads for T1 (5 logits) and T2 (5 Softplus regret outputs).
- **Memory & Speed:** $<1.2$ GB VRAM footprint; forward latency $<15$ ms per query.

---

## 10. Evaluation Protocol

All models will be evaluated strictly end-to-end through the frozen SARA response surface:

1. **Policy $F_1$:** Mean token $F_1$ achieved by routing query $q$ to its predicted budget $k$:
   $$\overline{F}_1 = \frac{1}{N} \sum_{i=1}^N F_1(q_i, \widehat{k}_i)$$
2. **Context Cost:** Mean high-fidelity context tokens consumed.
3. **Token Reduction vs Static $k=8$:**
   $$\Delta_{\text{tokens}} = \frac{1758.5 - \overline{\text{Tokens}}}{1758.5} \times 100\%$$
4. **Mean Regret:** $\frac{1}{N} \sum_{i=1}^N [F_1(q_i, \text{best}) - F_1(q_i, \widehat{k}_i)]$.
5. **Recovered Oracle Headroom (ROH):**
   $$\text{ROH} = \frac{\overline{F}_1(\text{policy}) - 0.3950}{0.4895 - 0.3950}$$
6. **Pareto Position:** Dominance relative to static baselines $k \in \{2, 4, 5, 6, 8\}$ and Week 4 policies.

---

## 11. Leakage Audit

A comprehensive leakage audit confirmed:
- Zero gold answer tokens, generated text tokens, or task metric values are accessible to the encoder.
- BM25 retrieval uses only query text and unannotated paper text chunks.
- Normalization parameters (StandardScaler) are fitted exclusively on training splits.
- Test set remains 100% frozen and untouched.
- Unit test `test_no_test_set_leakage` passed cleanly.

---

## 12. Compute & Resource Estimate

To generate the missing fixed-$k$ response surface on the 2,088 training queries:
- **Total Evaluations:** $2,088 \times 5 = 10,440$ SARA forward passes.
- **Per-Evaluation Latency:** ~1.50 seconds (greedy decoding, max 64 tokens, bfloat16 on NVIDIA A100/RTX GPU).
- **Total Compute Required:** $10,440 \times 1.5\text{ s} \approx 15,660\text{ s} \approx \mathbf{4.35\text{ GPU-hours}}$.
- **Storage Footprint:**
  - `train_retrieval.jsonl`: ~15 MB
  - `train_fixed_k_matrix.jsonl`: ~16 MB
  - `train_allocation_supervision.csv`: ~0.5 MB
  - Total Storage: $\approx \mathbf{31.5\text{ MB}}$.
- **Resumability:** `scripts/week5/build_train_response_surface.py` writes incrementally to JSONL and automatically skips already evaluated `(question_id, k)` pairs.

---

## 13. Blocking Issues

1. **Absence of Training Supervision:**
   - SARA fixed-$k$ generation has only been run on the 231 DEV queries.
   - The 2,088 TRAIN queries have 0 evaluations (`STATUS C: ABSENT`).
   - Supervised semantic models (I1–I4, T1–T2) cannot be trained on the training split until fixed-$k$ response supervision exists.

---

## 14. Go / No-Go Decision

$$\mathbf{W5\_PREFLIGHT\_STATUS = BLOCKED}$$

### Minimum Next Action Required:
1. Review and approve GPU compute allocation for `scripts/week5/build_train_response_surface.py` (~4.35 GPU-hours).
2. Execute the batch runner across the 2,088 training questions for $k \in \{2, 4, 5, 6, 8\}$.
3. Populate the canonical supervision table `results/week5/supervision/train_allocation_supervision.csv`.
4. Transition `W5_PREFLIGHT_STATUS` to `READY` and proceed to model training.
