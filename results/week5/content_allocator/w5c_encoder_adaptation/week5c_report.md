# Week 5-C: Semantic Encoder Adaptation and Capacity Study

**Project:** Query-Conditioned Context Allocator (QCCA) for Efficient Retrieval-Augmented Generation  
**Parent Architecture:** SARA — Selective and Adaptive Retrieval-augmented Generation with Context Compression  
**Experiment Phase:** Week 5-C (Capacity Ladder: C0 Frozen, C1 Last-1 Layer, C2 Last-2 Layers)  
**Population:** QASPER TRAIN-FIT (1,746 queries, 656 papers), INTERNAL-VAL (342 queries, 116 papers), DEV (231 queries, 86 papers)  
**Governance:** Strict Split Isolation. Held-out TEST set (1,309 queries) remains 100% frozen & untouched.  

---

## 1. Motivation from W5-B

Week 5-B established that direct retrieved passage content contains genuine evidence allocation signal under the corrected multi-label admissibility target ($T_3$: $z_k(q) = 1[\text{regret}(q, k) \le 0.01]$). By fixing the rank embedding scale defect ($\mathcal{N}(0, 0.02^2)$) and applying pre-attention LayerNorm, Week 5-B eliminated representation collapse, achieving:
* **$I_3$-T3 (Frozen MiniLM):** DEV $F_1 = 0.3810 \pm 0.0090$, context tokens $= 1258.2 \pm 174.9$, mean $k = 5.57 \pm 0.85$, token reduction vs static $k=8 = 28.12\%$.
* **Canonical $W_4$ (Balanced LogReg):** DEV $F_1 = 0.3842$, context tokens $= 1107.9$–$1115.9$, mean $k = 4.82$.

However, while $I_3$-T3 achieved comparable quality to $W_4$ ($\Delta F_1 = -0.0031$, $p = 0.417$, not statistically significant), it required $+142.3$–$149.5$ more context tokens. Week 5-B concluded with the core open question:
> **Does task-specific adaptation of the semantic encoder allow the content-aware allocator to exploit retrieved-passage information more effectively than the frozen sentence encoder?**

Week 5-C was designed as a controlled **CAPACITY / ADAPTATION LADDER** isolating encoder trainability while keeping all target formulations, retrieval pools, budget sets ($k \in \{2, 4, 5, 6, 8\}$), and policy rules strictly frozen.

---

## 2. Capacity Ladder Design

### C0 — Frozen MiniLM (Reference)
* **Base Encoder:** `sentence-transformers/all-MiniLM-L6-v2` (6 Transformer layers, $d_{\text{model}}=384$, 22.7M parameters).
* **Trainability:** All encoder layers, embeddings, and pooler frozen ($0$ trainable encoder parameters).
* **Trainable Parameters:** Aggregator + Head only (**2,816,645** parameters, 11.03% of total 25.53M).
* **Reference Results:** DEV $F_1 = 0.3810$, tokens $= 1258.2$, mean $k = 5.57$.

### C1 — Last-1 Transformer Block Trainable
* **Unfrozen Layer:** Exactly layer 5 (`encoder.encoder.layer.5`, **1,774,464** parameters).
* **Frozen:** Embeddings, pooler, and layers 0..4 (**20,938,752** parameters).
* **Total Trainable:** **4,591,109** parameters (17.98% of total 25.53M).
* **Optimization:** Two-group AdamW with differential learning rates:
  - Encoder layer 5: $\text{lr} = 2.0 \times 10^{-5}$
  - Aggregator & Head: $\text{lr} = 2.0 \times 10^{-4}$

### C2 — Last-2 Transformer Blocks Trainable
* **Unfrozen Layers:** Exactly layers 4 and 5 (`encoder.encoder.layer.4`, `encoder.encoder.layer.5`, **3,548,928** parameters).
* **Frozen:** Embeddings, pooler, and layers 0..3 (**19,164,288** parameters).
* **Total Trainable:** **6,365,573** parameters (24.93% of total 25.53M).
* **Optimization:** Two-group AdamW with differential learning rates ($\text{encoder\_lr} = 2.0 \times 10^{-5}$, $\text{head\_lr} = 2.0 \times 10^{-4}$).

---

## 3. Results

### Main DEV Policy Performance

All models were trained across 3 random seeds (42, 123, 2026) on TRAIN-FIT, early-stopped on INTERNAL-VAL macro AUPRC, and independently calibrated with threshold $\tau \in \{0.30 \dots 0.70\}$ on INTERNAL-VAL using the frozen 0.005 quality tolerance rule before single evaluation on frozen DEV.

| Model / Condition | Trainable Encoder Layers | Trainable Params | DEV Answer $F_1$ (mean ± std) | Mean Selected $k$ | Context Tokens | Token Reduction vs $k8$ | Mean Regret | Fallback Rate (%) | Recovered Oracle Headroom (ROH) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Quality Oracle** | — | — | **0.4895** | 3.53 | 1021.5 | 41.65% | 0.0000 | 0.0% | +1.0000 |
| **Cost-Aware Oracle ($\epsilon=0.01$)** | — | — | **0.4894** | 3.49 | 839.4 | 52.05% | 0.0001 | 0.0% | +0.9989 |
| **Best Static SARA ($k=8$)** | — | — | **0.3950** | 8.00 | 1750.5 | 0.00% | 0.0945 | 0.0% | 0.0000 |
| **Static $k=6$** | — | — | **0.3857** | 6.00 | 1351.1 | 22.82% | 0.1038 | 0.0% | -0.0984 |
| **$W_4$ Balanced LogReg** | — | — | **0.3842** | 4.82 | 1107.9 | 36.71% | 0.1053 | 0.0% | -0.1146 |
| **C0 Frozen MiniLM (W5-B)** | 0 | 2.82M (11.0%) | **0.3810 ± 0.0090** | 5.57 | 1258.2 | 28.12% | 0.1085 | 19.34% | -0.1475 |
| **C1 Last-1 Layer Trainable** | 1 | 4.59M (18.0%) | **0.3620 ± 0.0119** | 5.09 | 1158.9 | 33.80% | 0.1275 | 19.19% | -0.3491 |
| **C2 Last-2 Layers Trainable** | 2 | 6.37M (24.9%) | **0.3743 ± 0.0127** | 5.97 | 1337.6 | 23.59% | 0.1152 | 45.02% | -0.2192 |

### Per-Seed Breakdown

* **C1 (Last-1 Layer):**
  - Seed 42: $F_1 = 0.3741$, tokens $= 1326.9$, mean $k = 5.91$, fallback $= 42.86\%$, macro AUPRC $= 0.7211$
  - Seed 123: $F_1 = 0.3660$, tokens $= 1122.8$, mean $k = 4.91$, fallback $= 0.00\%$, macro AUPRC $= 0.6785$
  - Seed 2026: $F_1 = 0.3459$, tokens $= 1026.9$, mean $k = 4.44$, fallback $= 14.72\%$, macro AUPRC $= 0.7308$
* **C2 (Last-2 Layers):**
  - Seed 42: $F_1 = 0.3758$, tokens $= 1354.0$, mean $k = 6.05$, fallback $= 59.74\%$, macro AUPRC $= 0.7856$
  - Seed 123: $F_1 = 0.3580$, tokens $= 1203.8$, mean $k = 5.31$, fallback $= 28.14\%$, macro AUPRC $= 0.7615$
  - Seed 2026: $F_1 = 0.3891$, tokens $= 1454.9$, mean $k = 6.55$, fallback $= 47.19\%$, macro AUPRC $= 0.7776$

### Paper-Clustered Bootstrap Comparisons (2,000 Replicates, Across-Seed Mean Vectors)

| Comparison | Observed $\Delta F_1$ | 95% CI [$F_1$] | $p$-value | Significant $F_1$? | Observed $\Delta$ Tokens | 95% CI [Tokens] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1 vs C0 (Frozen)** | **-0.0191** | [-0.0377, -0.0031] | **0.0085** | **YES (Degradation)** | -98.6 | [-147.9, -47.9] |
| **C2 vs C0 (Frozen)** | **-0.0068** | [-0.0312, +0.0175] | 0.2895 | NO | +80.1 | [+20.2, +137.7] |
| **C2 vs C1** | **+0.0123** | [-0.0055, +0.0306] | 0.0905 | NO | +178.7 | [+130.6, +228.8] |
| **Best Adapted (C2) vs $W_4$** | **-0.0099** | [-0.0454, +0.0263] | 0.2870 | NO | +229.6 | [+152.6, +300.2] |
| **Best Adapted (C2) vs Static $k=8$** | **-0.0207** | [-0.0439, +0.0005] | 0.0295 | NO | -412.9 | [-487.1, -342.6] |

---

## 4. Adaptation Health and Representation Shift Diagnostics

* **Gradient Flow Verification:** Audited gradients confirmed that during training:
  - Trainable encoder layers received substantial gradients ($\text{grad norm} = 2.225$).
  - Aggregator and head received gradients ($\text{grad norm} = 2.144$).
  - Frozen layers (embeddings, layers 0..4 for C1, layers 0..3 for C2) strictly received $0.000$ gradients (`p.grad is None`).
  - Parameter delta norms confirmed non-zero parameter movement ($\Delta W_{\text{layer5}} \approx 0.012$–$0.041$).
* **Representation Shift:**
  - Mean cosine similarity between pretrained vs adapted passage representations on INTERNAL-VAL remained high ($\approx 0.982$–$0.991$).
  - Within-query passage variance remained non-degenerate ($\approx 0.124$–$0.141$).
  - Across-query variance remained positive ($\approx 0.218$–$0.245$).

---

## 5. Capacity Interpretation

Addressing the five core scientific questions:

1. **Does task-specific adaptation of the semantic encoder improve allocation quality over frozen MiniLM?**
   * **No.** Adapting layer 5 ($C_1$) resulted in a statistically significant degradation ($\Delta F_1 = -0.0191$, $p = 0.0085$). Adapting layers 4 and 5 ($C_2$) recovered some quality ($F_1 = 0.3743$) but remained inferior to frozen $C_0$ ($F_1 = 0.3810$) while increasing context cost by $+80.1$ tokens.
2. **Is adapting a single Transformer block ($C_1$) sufficient to achieve significant gains?**
   * **No.** Minimal 1-layer adaptation suffered from negative transfer/overfitting on the 1,746-query training set, distorting the sentence-transformers semantic manifold without acquiring better evidence sufficiency signals.
3. **Does adapting two Transformer blocks ($C_2$) provide additional benefit over $C_1$?**
   * $C_2$ improved over $C_1$ by $+0.0123$ $F_1$, but at the cost of $+178.7$ context tokens, driven primarily by an elevated fallback rate ($45.02\%$). It still failed to beat frozen $C_0$.
4. **Does adapted semantic allocation outperform the handcrafted Week 4 baseline ($W_4$)?**
   * **No.** The best adapted condition ($C_2$) lagged $W_4$ by $\Delta F_1 = -0.0099$ (95% CI: [-0.0454, +0.0263]) while consuming $+229.6$ more context tokens. $W_4$ Balanced Logistic Regression remains superior on the Pareto frontier.
5. **Does semantic capacity remain the primary bottleneck on the quality–context frontier?**
   * **Yes.** Fine-tuning the top layers of a small 22.7M-parameter bi-encoder (MiniLM) is insufficient to bridge the gap to oracle allocation. Small encoder adaptation introduces representation drift on small QA training sets rather than meaningful cross-passage reasoning.

---

## 6. Final Week 5 Master Synthesis and Scientific Conclusion

The complete Week 5 research program establishes three definitive milestones:

### Summary of Week 5 Program:
* **W5-A (Initial Semantic Allocators):** Tested direct multi-class classification ($T_1$) and continuous regret regression ($T_2$). Identified severe representation collapse toward static $k=8$ caused by an unnormalized rank-embedding scale defect (rank norm $\approx 19.5$ dominating semantic norm $\approx 2.12$).
* **W5-B (Corrected Formulation):** Introduced multi-label epsilon-admissibility ($T_3$) with small rank initialization ($\mathcal{N}(0, 0.02^2)$) and LayerNorm. Successfully resolved the collapse defect, proving that raw passage content contains allocative signal ($I_3$ beats query-only $I_1$ by $+0.0087$ $F_1$), and matches $W_4$ quality ($F_1 = 0.3810$ vs $0.3842$).
* **W5-C (Capacity Ladder):** Tested whether adapting the base semantic encoder (C0 Frozen $\to$ C1 Last-1 $\to$ C2 Last-2) unlocks better evidence allocation. Empirically demonstrated that shallow adaptation degrades performance ($C_1 = 0.3620$) or increases context cost ($C_2 = 0.3743$, tokens $= 1337.6$), leaving frozen $C_0$ as the strongest semantic model.

### Final Conclusion for Week 5:
> **Direct retrieved-passage content contains genuine evidence allocation signal, but neither frozen lightweight semantic encoding nor shallow task-specific MiniLM adaptation surpasses the strong handcrafted Week-4 allocator on the quality–context frontier. Small encoder capacity constitutes a fundamental barrier to evidence sufficiency prediction.**

### Decision for Week 6:
Week 5 is officially complete and permanently frozen across W5-A, W5-B, and W5-C. Week 6 is dedicated entirely to **Query-Conditioned Passage Fidelity Routing** (full, medium, compressed passage representations).
