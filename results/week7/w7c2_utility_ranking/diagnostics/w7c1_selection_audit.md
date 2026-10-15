# W7-C1 Model and Feature Selection Audit

**Date:** 2026-10-15  
**Artifact:** `results/week7/w7c2_utility_ranking/diagnostics/w7c1_selection_audit.md`  
**Purpose:** Investigate the failure modes of W7-C1 and audit why feature set F4 was reported as the winner despite INTERNAL-VAL metrics.

---

## 1. Canonical INTERNAL-VAL Feature Ablation Metrics (W7-C1)

From `results/week7/w7c_retrieval_aware_allocator/tables/w7c_feature_ablation_internal_val.csv`:

| Ablation | Description | Features | Calibrated $\tau$ | Val Downstream $F_1$ | Val Tokens | Val Mean $k$ | Over-Allocation Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **F0** | Old W4 baseline | 6 | 0.90 | **0.438401** | 1756.1 | 8.00 | 90.06% |
| **F1** | Query + Retrieval Geometry | 57 | 0.90 | 0.435477 | 1664.5 | 7.57 | 87.13% |
| **F2** | Retrieval-Aware | 70 | 0.90 | 0.435477 | 1655.8 | 7.53 | 85.09% |
| **F3** | Set-Aware | 79 | 0.90 | 0.432553 | 1647.5 | 7.48 | 84.80% |
| **F4** | Full Sufficiency Bank | 99 | 0.80 | 0.431919 | 1505.1 | 6.80 | 73.68% |

---

## 2. Declared Selection Rule vs. Execution Reality

### The Declared Rule:
1. Maximize downstream answer $F_1$ on INTERNAL-VAL.
2. If within $0.003$ $F_1$ of the maximum, select the configuration yielding lower context tokens.
3. If still tied, prefer the simpler model.

### The Audit Finding:
- **Maximum $F_1$ on INTERNAL-VAL:** $0.438401$ (achieved by F0).
- **$0.003$ Tolerance Band:** $[0.435401, 0.438401]$.
- **F2 vs. F0:** $\Delta F_1 = 0.438401 - 0.435477 = 0.002924 \le 0.003$. F2 saved **100.3 tokens** compared to F0 (1655.8 vs 1756.1). Under the declared rule, **F2 should have been selected** over F0, F1, F3, and F4.
- **F4 vs. F0:** $\Delta F_1 = 0.438401 - 0.431919 = 0.006482 > 0.003$. F4 was outside the allowable $0.003$ tolerance band.
- **Conclusion:** F4 was erroneously selected due to manual override (`FROZEN_FEATURE_SET = "F4"` in the notebook), which favored the maximal feature bank over the strict quantitative selection protocol.

---

## 3. The Fundamental Structural Failure of W7-C1

The audit reveals a much deeper, structural failure of the independent sufficiency classification paradigm:

1. **The Fallback-to-$k8$ Trap:**
   - In W7-C1, the policy was formulated as:
     $$\hat{k} = \min \{ k \in \{2,4,5,6,8\} : P(z(q,k)=1) \ge \tau \}, \quad \text{fallback: } k=8$$
   - Because sufficiency $z(q,k)=1$ was rare at $k=2$ and $k=4$, and the calibrated threshold $\tau$ was high ($0.80$ to $0.90$), the model rarely triggered at small $k$.
   - Consequently, the policy fell back to $k=8$ for $74\%$ to $90\%$ of queries!
   - On DEV, this pushed mean $k$ to $6.85$ and context tokens to $\approx 1503.8$.

2. **The Non-Monotonicity Trap:**
   - The fallback assumes that $k=8$ is "safe" (i.e. larger context contains all evidence).
   - In reality, the SARA response surface is **non-monotonic**: for many queries, extraneous distractors at $k=8$ actively degrade generation quality ($F(q,8) < F(q,4)$).
   - Falling back to $k=8$ directly caused catastrophic quality drops on queries where $k=4$ was optimal, dragging DEV answer $F_1$ down to **0.3783** (far below static $k=4$'s **0.3909**).

---

## 4. Key Takeaways for W7-C2

1. **Abandon Independent Sufficiency Classification:** Replace $P(z=1) \ge \tau$ with direct response-surface utility prediction ($\hat{F}(q,k)$) and query-grouped learning-to-rank.
2. **Never Fall Back to $k=8$:** Evaluate all candidate budgets comparatively. Pick the budget with the highest predicted downstream utility.
3. **Automate Selection Rule:** Ensure the programmatic model selector strictly enforces the $0.003$ $F_1$ tie-breaker rule without manual hardcoding.
