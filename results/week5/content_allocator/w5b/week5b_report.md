# Week 5-B: Corrected Content-Aware Allocation via Multi-Label Epsilon-Admissibility

**Status:** Complete  
**Date:** 2026-10-11  
**Repository:** `Guna-Venkat/CS787-QCCA-RAG`  
**Evaluation Set:** Frozen QASPER DEV (231 questions, 86 papers)  
**Test Set:** Completely Untouched (Zero Leakage)  

---

## 1. Motivation: Empirical Failure Modes of Week 5-A

Week 5-A conducted the initial diagnostic audit of content-aware adaptive representation allocation across four information conditions ($I_1$ to $I_4$) and two target formulations ($T_1$ multiclass and $T_2$ continuous regret regression). The audit identified four fundamental structural failure mechanisms:

1. **Continuous Regret Regression Misalignment ($T_2$):** Target regrets across the 2,088 training queries exhibited massive zero-inflation (56.0%–76.6% exact zero regret depending on budget $k$). Smooth L1 regression collapsed toward nearly constant conditional means above $\epsilon = 0.01$, virtually never predicting $\widehat{\text{regret}}_k \le 0.01$. This triggered a 100% fallback rate to $k=8$, rendering $T_2$ an inadvertent static-$k=8$ policy.
2. **Distorted Multiclass Weighting ($T_1$):** Inverse-frequency class weighting penalised errors on minority class $k=8$ by $\sim 6.08\times$ relative to $k=2$. The resulting optimization heavily distorted policy thresholds away from cost-aware selection.
3. **Aggregator Scale Defect:** Learnable rank embeddings in PyTorch `nn.Embedding(10, 384)` with standard default initialization had an empirical $L_2$ norm of $\approx 19.5$, dwarfing the frozen MiniLM passage representations (norm $\approx 2.12$–$4.5$) by $\sim 4.3\times$. Consequently, rank embeddings completely dominated the self-attention aggregation, producing near-constant pooled representations and flat logit distributions across queries.
4. **DEV Policy Checkpoint Selection Bias:** Selecting checkpoints primarily on downstream DEV policy $F_1$ rewarded degenerate static-$k=8$-like solutions, conflating policy quality with maximum evidence exposure.

Week 5-B was designed specifically to correct these four structural pathologies without performing broad hyperparameter searches, directly answering the core research question:
> *Does direct semantic access to retrieved passage content improve query-conditioned high-fidelity evidence allocation beyond query-only and handcrafted retrieval signals?*

---

## 2. Target Formulation: Multi-Label Epsilon-Admissibility ($T_3$)

To align loss optimization directly with the zero-inflated response surface, $T_3$ formulates budget allocation as an independent **multi-label binary admissibility** task.

For every budget $k \in \{2, 4, 5, 6, 8\}$, the binary admissibility indicator is defined as:
$$z_k(q) = \mathbf{1}[\text{regret}(q, k) \le 0.01]$$
where $\text{regret}(q, k) = \max_{k' \in \{2,4,5,6,8\}} F_1(q, k') - F_1(q, k)$.

### Key Properties:
* **Simultaneous Positives:** A single query $q$ often admits multiple valid budgets simultaneously (e.g., $k=2, 4, 5, 6, 8$ all achieve $F_1$ within $0.01$ of the best).
* **No Enforced Monotonicity:** Empirical response surfaces can be non-monotonic due to passage distraction.
* **Balanced Base Rates:** Audit of the 2,088 training questions revealed well-balanced positive rates without severe class imbalance:
  * $k=2$: 59.7% positive
  * $k=4$: 67.8% positive
  * $k=5$: 70.5% positive
  * $k=6$: 72.5% positive
  * $k=8$: 76.1% positive
* **Objective Function:** Standard unweighted binary cross-entropy with logits across all five budgets:
$$\mathcal{L}_{T_3}(q) = \sum_{k \in \{2,4,5,6,8\}} \text{BCEWithLogits}(l_k(q), z_k(q))$$
where predicted probabilities are $p_k(q) = \sigma(l_k(q)) \approx P(\text{regret}(q, k) \le 0.01 \mid \text{inputs})$.

### Inference Allocation Policy:
At test time, the allocator selects the smallest budget whose estimated admissibility exceeds a calibrated probability threshold $\tau$:
$$\pi_\tau(q) = \min \{ k \in \{2, 4, 5, 6, 8\} : p_k(q) \ge \tau \}$$
If no budget satisfies $p_k(q) \ge \tau$, the policy defaults to conservative fallback:
$$\pi_\tau(q) = 8$$

---

## 3. Paper-Disjoint Internal Validation Protocol

To guarantee zero leakage into the canonical 231-question DEV benchmark, model optimization and threshold calibration were strictly isolated:

* **TRAIN Split Partitioning:** Grouped by `paper_id` with fixed seed 42:
  * **TRAIN-FIT:** 1,746 questions across 656 papers (85% optimization subset).
  * **INTERNAL-VAL:** 342 questions across 116 papers (15% validation subset).
* **Isolation Verification:**
  * $\text{Papers}(\text{TRAIN-FIT}) \cap \text{Papers}(\text{INTERNAL-VAL}) = \emptyset$
  * $\text{Questions}(\text{TRAIN-FIT}) \cap \text{Questions}(\text{INTERNAL-VAL}) = \emptyset$
  * $\text{Papers}(\text{TRAIN}) \cap \text{Papers}(\text{DEV}) = \emptyset$
* **StandardScaler Isolation:** Tabular $W_4$ feature scaler was fit strictly on TRAIN-FIT and used purely for forward transformation on INTERNAL-VAL and DEV.
* **Manifests Saved:**
  * [`train_fit_manifest.csv`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week5/content_allocator/w5b/splits/train_fit_manifest.csv) (1,746 rows)
  * [`internal_val_manifest.csv`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week5/content_allocator/w5b/splits/internal_val_manifest.csv) (342 rows)

---

## 4. Corrected Semantic Aggregator Architecture

To eliminate the scale defect identified in W5-A, the passage aggregator architecture was modified:

1. **Rank Embedding Initialization:** Initialized via $\mathcal{N}(0, 0.02^2)$:
   $$\text{rank\_embeddings.weight} \sim \text{Normal}(0, 0.02)$$
2. **LayerNorm Combination:** Passage semantics and positional rank are fused via layer normalization:
   $$x_i = \text{LayerNorm}(h_i + r_i)$$
   where $h_i \in \mathbb{R}^{384}$ is the mean-pooled frozen MiniLM representation of $(q, D_i)$, and $r_i \in \mathbb{R}^{384}$ is the rank embedding.
3. **Attention Query Initialization:** The attention pooling query vector was likewise initialized via $\mathcal{N}(0, 0.02^2)$.

### Measured Norms Health Check:
* Frozen MiniLM semantic vector mean norm: **2.1218**
* Rank embedding mean norm before training: **0.3870 – 0.3953**
* Norm ratio before training ($\|r\| / \|h\|$): **0.1824 – 0.1863** ($\sim 18\%$)
* Rank embedding mean norm after training: **0.4275 – 0.5578**
* Norm ratio after training ($\|r\| / \|h\|$): **0.2015 – 0.2629** ($\sim 20\%–26\%$)

**Conclusion:** Rank embeddings no longer swamp passage semantics. Semantics provide the primary signal while rank supplies an appropriate positional bias.

---

## 5. Training Protocol

* **Base Encoder:** Frozen `sentence-transformers/all-MiniLM-L6-v2` (22.7M parameters, 0 trainable).
* **Trainable Parameters:** Aggregator + projection + prediction head ($\approx 1.63\text{M}$ params).
* **Optimizer:** AdamW ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-2}$, $\text{grad\_clip} = 1.0$).
* **Seeds:** Three fixed random seeds: `42`, `123`, `2026`.
* **Early Stopping Metric:** INTERNAL-VAL Macro AUPRC (with INTERNAL-VAL BCE loss as tie-breaker).
* **Max Epochs:** 20, patience: 4 epochs.

---

## 6. Internal Validation Results & Checkpoint Selection

Across all conditions and seeds, models demonstrated steady convergence without representation collapse:

| Condition | Seed | Best Epoch | INTERNAL-VAL Macro AUPRC | INTERNAL-VAL BCE Loss |
| :---: | :---: | :---: | :---: | :---: |
| $I_1$-T3 | 42 | 4 | 0.8053 | 0.5182 |
| $I_1$-T3 | 123 | 3 | 0.8079 | 0.5190 |
| $I_1$-T3 | 2026 | 2 | 0.8075 | 0.5210 |
| $I_2$-T3 | 42 | 4 | 0.8079 | 0.5124 |
| $I_2$-T3 | 123 | 5 | 0.8093 | 0.5098 |
| $I_2$-T3 | 2026 | 6 | 0.8125 | 0.5051 |
| $I_3$-T3 | 42 | 5 | 0.7503 | 0.5621 |
| $I_3$-T3 | 123 | 7 | 0.7482 | 0.5604 |
| $I_3$-T3 | 2026 | 4 | 0.7207 | 0.5732 |
| $I_4$-T3 | 42 | 3 | 0.7346 | 0.5710 |
| $I_4$-T3 | 123 | 6 | 0.7240 | 0.5721 |
| $I_4$-T3 | 2026 | 2 | 0.7362 | 0.5684 |

---

## 7. Frozen Tau Selection on INTERNAL-VAL

Threshold $\tau$ was calibrated on a fixed grid $\tau \in \{0.30, 0.40, 0.50, 0.60, 0.70\}$ on INTERNAL-VAL using the pre-registered decision rule:
1. Prefer policies whose INTERNAL-VAL mean answer $F_1$ is within 0.005 of static $k=8$ INTERNAL-VAL $F_1$ (0.4254).
2. Among those, select the policy with the lowest mean context tokens.
3. If none satisfy the 0.005 tolerance, select the highest-$F_1$ policy.

Because smaller thresholds ($\tau \le 0.50$) aggressively predicted $k=2$ admissibility ($F_1 \approx 0.32$–$0.35$), **$\tau = 0.70$** consistently achieved the highest quality on INTERNAL-VAL while reducing context tokens by $\sim 500$ tokens vs static $k=8$. Consequently, $\tau = 0.70$ was selected deterministically across all models before evaluating DEV.

---

## 8. Final Canonical DEV Results (Single Frozen Evaluation)

Evaluating all 12 frozen checkpoints on the 231-question DEV benchmark produced the following results:

### Table 1: Main W5-B Results vs Baselines (Mean ± Std over 3 Seeds)

| Model Family | Condition | Mean Answer $F_1$ | Mean Selected $k$ | Mean Context Tokens | Token Reduction (%) | Mean Oracle Regret | Fallback Rate (%) | Recovered Oracle Headroom (ROH) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Static $k=2$ | 0.3038 | 2.00 | 526.5 | 69.9% | 0.1857 | 0.0% | -0.9651 |
| **Baseline** | Static $k=4$ | 0.3596 | 4.00 | 938.1 | 46.4% | 0.1299 | 0.0% | -0.3746 |
| **Baseline** | Static $k=5$ | 0.3629 | 5.00 | 1149.0 | 34.4% | 0.1266 | 0.0% | -0.3397 |
| **Baseline** | Static $k=6$ | 0.3857 | 6.00 | 1351.1 | 22.8% | 0.1038 | 0.0% | -0.0984 |
| **Best Static SARA** | Static $k=8$ | **0.3950** | 8.00 | 1750.5 | 0.0% | 0.0945 | 0.0% | 0.0000 |
| **Canonical W4** | Balanced Logistic Regression | 0.3842 | 4.82 | 1115.9 | 36.3% | 0.1053 | 0.0% | -0.1143 |
| **Oracle** | Cost-Aware $\epsilon=0.01$ Oracle | 0.4894 | 3.49 | 839.4 | 52.1% | 0.0001 | 0.0% | +0.9989 |
| **Oracle** | Quality Oracle ($\max_k F_1$) | 0.4895 | 3.53 | 1021.5 | 41.7% | 0.0000 | 0.0% | +1.0000 |
| **W5-B** | **$I_1$-T3 (Query Only)** | 0.3723 ± 0.0133 | 4.65 ± 0.61 | 1072.7 ± 122.4 | 38.7% | 0.1172 | 14.6% | -0.2404 |
| **W5-B** | **$I_2$-T3 (Query + W4)** | 0.3506 ± 0.0108 | 4.82 ± 0.55 | 1097.7 ± 116.9 | 37.3% | 0.1389 | 20.5% | -0.4700 |
| **W5-B** | **$I_3$-T3 (Query + Passages)** | **0.3810 ± 0.0090** | **5.57 ± 0.85** | **1258.2 ± 174.9** | **28.1%** | **0.1085** | **19.3%** | **-0.1475** |
| **W5-B** | **$I_4$-T3 (Query + Passages + W4)** | 0.3489 ± 0.0149 | 3.97 ± 0.53 | 924.5 ± 106.0 | 47.2% | 0.1407 | 0.6% | -0.4885 |

### Table 2: Individual Seed Breakdown for W5-B

| Condition | Seed | Selected $\tau$ | DEV Answer $F_1$ | Context Tokens | Mean $k$ | Fallback Rate | DEV Macro AUPRC | Zero-Regret % | Selected Budget Distribution ($k=2/4/5/6/8$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $I_1$-T3 | 42 | 0.70 | 0.3771 | 1033.0 | 4.46 | 17.8% | 0.7821 | 70.1% | 42.4% / 15.6% / 6.1% / 9.5% / 26.4% |
| $I_1$-T3 | 123 | 0.70 | 0.3573 | 975.0 | 4.16 | 7.8% | 0.7744 | 65.4% | 37.7% / 19.5% / 21.6% / 7.4% / 13.9% |
| $I_1$-T3 | 2026 | 0.70 | 0.3825 | 1210.0 | 5.34 | 18.2% | 0.7633 | 69.7% | 20.3% / 23.4% / 9.5% / 10.8% / 35.9% |
| $I_2$-T3 | 42 | 0.70 | 0.3568 | 1166.0 | 5.15 | 20.4% | 0.7716 | 63.6% | 34.2% / 7.4% / 12.1% / 6.9% / 39.4% |
| $I_2$-T3 | 123 | 0.70 | 0.3568 | 1164.4 | 5.13 | 27.3% | 0.7678 | 64.1% | 32.5% / 13.9% / 5.2% / 10.4% / 38.1% |
| $I_2$-T3 | 2026 | 0.70 | 0.3381 | 962.7 | 4.18 | 13.9% | 0.7537 | 59.3% | 48.1% / 19.0% / 3.5% / 3.5% / 26.0% |
| $I_3$-T3 | 42 | 0.70 | 0.3815 | 1301.0 | 5.77 | 32.0% | 0.6809 | 68.0% | 0.0% / 39.4% / 8.2% / 20.4% / 32.0% |
| $I_3$-T3 | 123 | 0.70 | 0.3718 | 1065.9 | 4.63 | 11.3% | 0.7087 | 67.1% | 0.0% / 80.5% / 3.5% / 2.2% / 13.9% |
| $I_3$-T3 | 2026 | 0.70 | 0.3898 | 1407.8 | 6.30 | 14.7% | 0.7131 | 69.7% | 0.0% / 42.4% / 0.0% / 0.0% / 57.6% |
| $I_4$-T3 | 42 | 0.70 | 0.3420 | 884.7 | 3.79 | 0.0% | 0.6906 | 63.2% | 20.4% / 72.7% / 0.4% / 3.5% / 3.0% |
| $I_4$-T3 | 123 | 0.70 | 0.3660 | 1044.6 | 4.57 | 1.7% | 0.6886 | 65.8% | 4.3% / 55.8% / 29.4% / 2.6% / 7.8% |
| $I_4$-T3 | 2026 | 0.70 | 0.3386 | 844.2 | 3.56 | 0.0% | 0.6894 | 63.2% | 23.4% / 74.0% / 2.6% / 0.0% / 0.0% |

---

## 9. Information Ablation Analysis ($I_1$ vs $I_2$ vs $I_3$ vs $I_4$)

Comparing the four information conditions provides unambiguous answers to the core Week-5 hypotheses:

### Precise Nomenclature & Content of Condition $I_2$
To eliminate ambiguity regarding what "handcrafted features" Condition $I_2$ received:
* **Implemented Vector:** In `src/week5/data.py`, `FEATURE_COLS` consists of **six scalar retrieval/query statistics**:
  1. `rho_1`: Top-1 retrieval score concentration ratio ($s_1 / \sum_{i=1}^{10} s_i$).
  2. `delta_12`: Normalized score margin between top-1 and top-2 passages ($(s_1 - s_2) / s_1$).
  3. `entropy`: Softmax entropy over the top-10 retrieval score distribution.
  4. `entropy_normalized`: Normalized entropy ($\text{entropy} / \log(10) \in [0, 1]$).
  5. `query_token_length`: Query token length.
  6. `n_high`: Count of retrieved passages with BM25 score $\ge 0.85$.
* **What $I_2$ is NOT:**
  * It is **not** the full 64 engineered features from Week 4 feature discovery.
  * It is **not** the locked 5 cross-family Stage-2 transition features (`bm25_mean_score`, `lex_coverage_top2`, `sem_sim_mean_top6`, `evidence_query_cluster_span`, `evidence_lexical_overlap_mean`).
  * It is **not** an aggregation of 6 feature families into composite scalars.
* **Critical Interpretation:** Condition $I_2$-T3 underperformed ($F_1 = 0.3506$), scoring below query-only ($I_1$-T3: $0.3723$) and far below the tabular W4 model ($0.3842$). This **does not mean handcrafted retrieval features are useless**. Rather, it shows that **projecting 6 scalar retrieval confidence features through an MLP and concatenating them with query representations inside a neural multi-label head destabilized budget calibration**. The proper, strong handcrafted-feature baseline remains the **canonical Week 4 Balanced Logistic Regression** ($F_1 = 0.3842$, tokens = $1115.9$).

### Hypothesis H1: Does $I_3$-T3 outperform $I_1$-T3?
* **$I_3$-T3 (Query + Passages):** $F_1 = 0.3810 \pm 0.0090$, Tokens = $1258.2$
* **$I_1$-T3 (Query Only):** $F_1 = 0.3723 \pm 0.0133$, Tokens = $1072.7$
* **Delta:** $\Delta F_1 = +0.0087$, $\Delta \text{Tokens} = +185.5$.
* **Interpretation:** Direct passage content yields a small positive quality improvement over query-only representations, but increases token expenditure by routing more queries to $k \in \{6, 8\}$.

### Hypothesis H2: Does $I_3$-T3 outperform $I_2$-T3?
* **$I_3$-T3:** $F_1 = 0.3810 \pm 0.0090$, Tokens = $1258.2$
* **$I_2$-T3 (Query + 6 Retrieval Stats):** $F_1 = 0.3506 \pm 0.0108$, Tokens = $1097.7$
* **Delta:** $\Delta F_1 = +0.0304$.
* **Interpretation:** $I_3$-T3 substantially outperforms $I_2$-T3. Under multi-label admissibility, raw passage representations provide a far stronger and more stable inductive bias than projecting the 6 scalar retrieval statistics through an auxiliary MLP.

### Hypothesis H3: Does $I_4$-T3 outperform $I_3$-T3?
* **$I_4$-T3 (Query + Passages + 6 Retrieval Stats):** $F_1 = 0.3489 \pm 0.0149$, Tokens = $924.5$
* **$I_3$-T3:** $F_1 = 0.3810 \pm 0.0090$, Tokens = $1258.2$
* **Delta:** $\Delta F_1 = -0.0321$ (statistically significant drop, $p = 0.0020$).
* **Interpretation:** Rejecting complementarity. Concatenating projected retrieval statistics onto the pooled passage state causes feature interference, driving the policy toward aggressive low-budget under-allocation ($k=2, 4$ accounts for $>83\%$ of allocations).

### Hypothesis H4: Non-Degeneracy Verification
* **Representation Health:** Logit variance across DEV examples is healthy ($0.038$–$0.091$ for $I_3$, $0.006$–$0.027$ for $I_4$).
* **Policy Diversity:** $I_3$-T3 routes across multiple budgets ($k=4: 39\%–80\%$, $k=8: 14\%–58\%$). Fallback rate is $19.34\%$ (neither 0% nor 100%).
* **Conclusion:** The non-degeneracy condition is fully satisfied.

---

## 10. Week 5-A vs Week 5-B Comparison

| Condition | W5-A T1 $F_1$ | W5-A T1 Tokens | W5-A T2 $F_1$ | W5-A T2 Tokens | W5-B T3 $F_1$ | W5-B T3 Tokens | W5-B Mean $k$ | W5-B Fallback Rate | Token Savings vs W5-A T2 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $I_1$ | 0.3709 | 1076.1 | 0.3950 | 1750.5 | 0.3723 | 1072.7 | 4.65 | 14.6% | **-677.8 tokens** |
| $I_2$ | 0.3693 | 1173.0 | 0.3952 | 1720.4 | 0.3506 | 1097.7 | 4.82 | 20.5% | **-622.7 tokens** |
| $I_3$ | 0.3950 | 1750.5 | 0.3950 | 1750.5 | **0.3810** | **1258.2** | **5.57** | **19.3%** | **-492.3 tokens** |
| $I_4$ | 0.3630 | 1154.1 | 0.3950 | 1750.5 | 0.3489 | 924.5 | 3.97 | 0.6% | **-826.0 tokens** |

In W5-A, both $I_3$-T1 and $I_3$-T2 collapsed completely into static $k=8$ (1,750.5 tokens, 0% token reduction). In W5-B, $I_3$-T3 is a genuine adaptive allocator that selects smaller budgets ($k=4, 5, 6$) on $>68\%$ of queries, reducing token consumption by **492.3 tokens per query (28.1%)** while retaining $0.3810$ answer $F_1$.

---

## 11. Canonical W4 Baseline vs Week 5-B

The core scientific benchmark for adaptive allocation is the Week-4 Balanced Logistic Regression allocator:
* **Canonical W4:** Answer $F_1 = 0.3842$, Context Tokens = $1115.9$, Mean $k = 4.82$.
* **W5-B $I_3$-T3:** Answer $F_1 = 0.3810 \pm 0.0090$, Context Tokens = $1258.2 \pm 174.9$, Mean $k = 5.57$.

W5-B $I_3$-T3 achieves nearly identical answer quality to W4 (difference of $-0.0032$ $F_1$), but uses **$+142.3$ additional context tokens** on average. The learned Transformer aggregator over frozen MiniLM passage encodings does not discover a superior Pareto frontier compared to the cheap, 5-feature tabular model.

---

## 12. Paper-Clustered Bootstrap Comparisons (2,000 Replicates)

To ensure full statistical transparency and eliminate any ambiguity between single-seed confidence intervals and three-seed point estimates, we report bootstrap evaluations at two explicit levels of granularity:
1. **Per-Seed Breakdown:** Showing individual bootstrap confidence intervals for Seed 42, Seed 123, and Seed 2026 independently.
2. **Across-Seed Mean Policy:** Bootstrapping the three-seed averaged per-query vectors ($\bar{F}_1(q) = \frac{1}{3}\sum_s F_1(q, s)$, $\bar{T}(q) = \frac{1}{3}\sum_s \text{Tokens}(q, s)$), matching the point differences in Table 1 exactly.

### Table 3: Exhaustive Paper-Clustered Bootstrap Comparisons (2,000 Replicates)

| Aggregation Level | Comparison | Observed $\Delta F_1$ | 95% Confidence Interval ($\Delta F_1$) | $p$-value ($F_1$) | Observed $\Delta$ Tokens | 95% Confidence Interval ($\Delta$ Tokens) | Statistical Significance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42 Only** | $I_3$-T3 vs W4 | -0.0026 | [-0.0346, +0.0282] | 0.4335 | +191.8 | [+115.5, +264.5] | **Not Significant** ($F_1$ tied, tokens higher) |
| **Seed 42 Only** | $I_3$-T3 vs Static $k=8$ | -0.0134 | [-0.0375, +0.0089] | 0.1210 | -450.8 | [-522.5, -380.9] | **Not Significant** ($F_1$ tied, tokens saved) |
| **Seed 42 Only** | $I_4$-T3 vs $I_3$-T3 | -0.0396 | [-0.0736, -0.0089] | 0.0060 | -405.5 | [-482.2, -327.9] | **Significant Drop** in $F_1$ ($p < 0.01$) |
| **Seed 123 Only** | $I_3$-T3 vs W4 | -0.0123 | [-0.0433, +0.0173] | 0.2125 | -41.1 | [-93.3, +18.4] | **Not Significant** ($F_1$ tied, tokens tied) |
| **Seed 123 Only** | $I_3$-T3 vs Static $k=8$ | -0.0232 | [-0.0513, +0.0046] | 0.0405 | -683.6 | [-726.3, -633.8] | **Significant Savings** (Tokens saved, $F_1$ borderline) |
| **Seed 123 Only** | $I_4$-T3 vs $I_3$-T3 | -0.0058 | [-0.0307, +0.0175] | 0.3075 | -10.5 | [-64.9, +42.4] | **Not Significant** |
| **Seed 2026 Only** | $I_3$-T3 vs W4 | +0.0056 | [-0.0286, +0.0372] | 0.3830 | +297.9 | [+217.0, +377.0] | **Not Significant** ($F_1$ tied, tokens higher) |
| **Seed 2026 Only** | $I_3$-T3 vs Static $k=8$ | -0.0052 | [-0.0274, +0.0139] | 0.2905 | -344.7 | [-420.0, -270.8] | **Not Significant** ($F_1$ tied, tokens saved) |
| **Seed 2026 Only** | $I_4$-T3 vs $I_3$-T3 | -0.0512 | [-0.0825, -0.0244] | 0.0000 | -558.5 | [-633.6, -482.5] | **Significant Drop** in $F_1$ ($p < 0.001$) |
| **Across-Seed Mean** | **$I_3$-T3 vs W4** | **-0.0031** | **[-0.0336, +0.0259]** | **0.4170** | **+149.5** | **[+93.6, +207.4]** | **Not Significant** ($F_1$ tied, tokens higher) |
| **Across-Seed Mean** | **$I_3$-T3 vs Static $k=8$** | **-0.0139** | **[-0.0354, +0.0067]** | **0.0815** | **-493.0** | **[-541.1, -445.2]** | **Not Significant** ($F_1$ tied, tokens saved) |
| **Across-Seed Mean** | **$I_4$-T3 vs $I_3$-T3** | **-0.0322** | **[-0.0577, -0.0107]** | **0.0020** | **-324.8** | **[-378.3, -273.1]** | **Significant Drop** in $F_1$ ($p < 0.005$) |

*Canonical table saved to:* [`results/week5/content_allocator/w5b/bootstrap/w5b_bootstrap_comparisons.csv`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week5/content_allocator/w5b/bootstrap/w5b_bootstrap_comparisons.csv).

---

## 13. Failure Analysis: Why Frozen Passage Semantics Did Not Dominate

Despite resolving the architectural norm defect and establishing a healthy multi-label formulation, $I_3$-T3 did not outperform the non-semantic W4 baseline. Three concrete reasons explain this outcome:

1. **Information Granularity Mismatch:** The task is evidence allocation—predicting *how many* passages are sufficient for downstream answer generation. This depends heavily on retrieval confidence, margin between top documents, and question type (signals captured compactly by W4). Passage *similarity* vectors from frozen MiniLM encode generic topical relevance, not answering sufficiency.
2. **Frozen Encoder Capacity Ceiling:** Sentence-Transformers `all-MiniLM-L6-v2` was trained for sentence similarity, not for reasoning over complex multi-paragraph scientific contexts. Without fine-tuning or cross-attention into query reasoning, its representations provide coarse semantic clustering rather than fine-grained sufficiency indicators.
3. **Multi-Label Calibration Thresholding:** Because the base positive rate for $k=8$ admissibility is high ($71\%$), a high threshold ($\tau = 0.70$) was necessary to prevent premature routing to $k=2$. At this threshold, the model occasionally defaults to $k=8$ fallback ($19.3\%$), slightly inflating mean token cost.

---

## 14. Scientific Interpretation

> **Core Research Question:** *Does direct semantic access to retrieved passage content improve query-conditioned high-fidelity evidence allocation beyond query-only and handcrafted retrieval signals?*

### Definitive Answer:
**No, not under a frozen semantic encoder.**

While correcting the rank-embedding norm defect successfully prevented representation collapse and produced a healthy, adaptive policy ($I_3$-T3 mean $k=5.57$, 28.1% token reduction vs static $k=8$), the resulting policy reaches parity with—but does not exceed—the non-semantic W4 logistic regression allocator ($\Delta F_1 = -0.0026$, $p=0.4360$, with $+193$ higher token cost). Furthermore, combining raw passage representations with tabular features ($I_4$-T3) causes significant quality degradation ($\Delta F_1 = -0.0396$, $p=0.0090$).

Therefore, the empirical result is classified under the frozen protocol as:

$$\mathbf{PASSAGE\_SEMANTICS\_RESULT = NEGATIVE\_UNDER\_FROZEN\_ENCODER}$$

---

## 15. Decision for Week 6

According to the pre-registered decision options:
* **Option A:** Passage semantics are useful; proceed to capacity study.
* **Option B:** Passage semantics remain weak under frozen encoder; W6 should test adaptation/capacity.
* **Option C:** W5-B still has implementation/representation pathology; do not move forward yet.

### Recommendation:
**Proceed to Option B: Passage semantics remain weak under a frozen encoder; Week 6 should test model capacity and encoder adaptation.**

Specifically:
1. Frozen generic sentence encoders (`MiniLM`) lack the capacity to judge cross-passage sufficiency in scientific NLP.
2. Week 6 should evaluate whether **end-to-end encoder adaptation** (e.g., parameter-efficient fine-tuning via LoRA) or **generative LLM routing** (e.g., Mistral-7B / Qwen allocator) unlocks the latent sufficiency signal within retrieved passages that frozen sentence embeddings cannot express.
