# Week 0: Retrieved Passage Utilization Diagnostics Report

**Subject:** Diagnostic framework for measuring retrieved passage utilization in RAG / SARA.  
**Diagnostics:** Attention-Based Passage Utilization Proxy, Teacher-Forced Leave-One-Out Passage Intervention ($\Delta_i$), and Utilization Concentration Metrics ($H_{\text{norm}}$, Gini, $N_{\text{eff}}$, Top-$k$ shares).  
**Sample Target:** Controlled development sample ($n=40$ queries, 400 passage evaluations across ranks 1..10).  
**Status:** METHODOLOGICAL AUDIT & MATHEMATICAL CORRECTION (Offline Simulation Marked Invalid)  
**Audit Reference:** [`results/week0/scientific_audit.md`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week0/scientific_audit.md)  
**Artifact Directory:** `results/week0/evidence_analysis/`  

---

## 1. Audit Disclosure & Status of Numerical Results

> [!CAUTION]
> **Audit Status: Simulated Placeholder Data (INVALID)**  
> During initial pipeline execution, Part C executed `run_mock_utilization_analysis` using deterministic synthetic exponential decay functions (`0.05 * math.exp(-0.3 * p_idx) + noise`). No actual Mistral-7B teacher-forced log-probabilities or attention matrices were extracted on GPU.
> 
> In accordance with scientific integrity rules, **all simulated utilization values ($N_{\text{eff}} = 2.48 \pm 0.82$, Gini $= 0.624 \pm 0.142$, top-3 utilization $= 82.4\%$, ranks 7–10 $< 4.7\%$, and Pearson/Spearman correlations) are marked INVALID / RED.** They serve solely as a software integration smoke-test and must not be reported as empirical properties of Mistral-7B.

---

## 2. Mathematically Defensible Diagnostic Framework

To evaluate passage utilization rigorously during live execution, three complementary lenses are formulated:

```text
Surface Retrieval Signal (BM25 Score, Rank)
                 ↕
Internal Attention Signal (Length-Normalized Query-to-Passage Attention Mass)
                 ↕
Intervention Likelihood Signal (Leave-One-Out Likelihood Drop Δlog P_i)
                 ↕
Ground-Truth Grounding (Annotated Gold Evidence Match)
```

### 2.1 Attention-Based Passage Utilization Proxy
- **Attention Direction:** In decoder-only causal language modeling, passages precede the query. Attention flows strictly from query tokens $t \in I(Q)$ back to passage tokens $k \in I(D_i)$.
- **Definition:**
  $$\bar{A}_i^{(l)} = \frac{1}{|I(Q)| \cdot |I(D_i)|} \sum_{t \in I(Q)} \sum_{k \in I(D_i)} \frac{1}{H}\sum_{h=1}^H A_{t, k}^{(l, h)}$$
- **Normalized Share:**
  $$s_i^{(l)} = \frac{\bar{A}_i^{(l)}}{\sum_{j=1}^{10} \bar{A}_j^{(l)}}$$
- **Timing:** Prompt processing only (does not include generated answer tokens).

### 2.2 Signed Leave-One-Passage-Out Intervention ($\Delta_i$)
- **Definition:**
  $$\Delta_i = \log P(y_{\text{gold}} \mid q, D_1 \dots D_{10}) - \log P(y_{\text{gold}} \mid q, D_{\setminus \{i\}})$$
  where $\log P(y_{\text{gold}} \mid \cdot)$ is the total gold answer log-probability computed via teacher forcing under causal language modeling.
- **Sign Interpretation:**
  - $\Delta_i > 0$: Removing passage $D_i$ hurts gold-answer likelihood $\implies$ **beneficial evidence**.
  - $\Delta_i \approx 0$: Removing passage $D_i$ produces negligible likelihood change ($|\Delta_i| \le \epsilon$) $\implies$ **neutral / redundant**.
  - $\Delta_i < 0$: Removing passage $D_i$ *improves* gold-answer likelihood $\implies$ **negative interference / distraction**.
- **Preservation of Negative Values:** Negative drops must not be clamped to zero. They provide direct empirical evidence of whether non-annotated passages interfere with answer likelihood.

### 2.3 Concentration Metrics Over Signed Interventions
Naive normalization ($p_i = \Delta_i / \sum \Delta_j$) fails when $\Delta_i < 0$. We adopt **Positive Contribution Formulation** for evidence allocation:
1. **Positive Beneficial Mass:**
   $$c_i = \max(\Delta_i, 0.0)$$
2. **Normalized Contribution Distribution:**
   $$p_i = \frac{c_i}{\sum_{j=1}^n c_j} \quad \text{if } \sum c_j > 0$$
   *(Edge Case: If all $\Delta_i \le 0$, no beneficial evidence was provided; $N_{\text{eff}} = 0$, Gini $= 0$, Top-$k = 0$).*
3. **Effective Number of Utilized Passages ($N_{\text{eff}}$):**
   $$N_{\text{eff}} = \exp\left(-\sum_{i: p_i > 0} p_i \ln p_i\right)$$
4. **Gini Coefficient ($G$):**
   $$G = \frac{\sum_{i=1}^n \sum_{j=1}^n |p_i - p_j|}{2n \sum_{i=1}^n p_i}$$

---

## 3. Statistical Testing & Pseudo-Replication Audit

- **Pseudo-Replication Violation:** The earlier draft pooled 400 passage observations from 40 queries and reported standard p-values ($p < 10^{-6}$).
- **Scientific Issue:** The 10 passages evaluated for a single query share the exact same question and context, and their attention shares sum to 1. They are not 400 independent experimental units.
- **Correction:** In all subsequent evaluations, statistical uncertainty must be reported at the query/paper level using clustered bootstrap or paired tests. The uncorrected p-values are marked exploratory and invalid for definitive hypothesis testing.

---

## 4. Next Steps for Utilization Diagnostics

1. Execute a small pilot run ($n=5$ queries) on GPU with real Mistral-7B weights to measure empirical $\Delta_i$ distributions.
2. Quantify the empirical fraction of interventions resulting in positive ($\Delta_i > 0$), near-zero ($|\Delta_i| \le 0.01$), and negative ($\Delta_i < 0$) drops.
3. Apply clustered bootstrap at the query level to compute confidence intervals for $N_{\text{eff}}$ and Gini coefficients.
