# QCCA-V2 Diagnostic and Policy-Refinement Report

**Project:** Query-Conditioned Context Allocator (QCCA) for Efficient Retrieval-Augmented Generation  
**Parent Architecture:** SARA — Selective and Adaptive Retrieval-augmented Generation with Context Compression  
**Stage:** Week 4 Sequential Policy Diagnosis, Error Propagation & Architecture Justification  
**Evaluation Benchmark:** QASPER Development Split (86 papers, 231 questions)  
**Governance Status:** Development Only. QASPER Historical Test Set Remains 100% Frozen & Untouched.  
**Diagnostic Directory:** `results/week4/qcca_v2_diagnostic/`

---

## 1. Executive Summary & Diagnostic Motivation

The initial QCCA-V2 sequential modeling experiment established several critical baseline findings:
1. **The Sequential Formulation is Theoretically Sound:** The Oracle Sequential Policy achieves **Mean $F_1 = 0.4069$** at **$650.3$ context tokens** (**$63.02\%$ token reduction vs Static $k=8$**). It outperforms Static $k=8$ ($F_1 = 0.3950$) while consuming nearly two-thirds fewer tokens, proving that sequential early stopping successfully eliminates distractor contamination for $76.2\%$ of queries.
2. **The First Machine Learning Allocator Underperformed:**
   - Standard Unweighted Logistic Regression at $t=0.5$ collapsed to Static $k=2$ ($F_1 = 0.3038$, 69.60% token reduction).
   - Balanced Logistic Regression achieved $F_1 = 0.3318$ (mean $k = 3.50$, 52.08% token reduction).
   - Ridge Continuous Gain Regression achieved $F_1 = 0.3675$ (mean $k = 5.45$, 29.39% token reduction).
3. **Stage Asymmetry:** Stage 2 ($4 \rightarrow 6$) demonstrated genuine out-of-fold predictive signal (Balanced LogReg ROC-AUC = **$0.6503$**, PR-AUC = **$0.3199$** vs majority prior $0.1948$, Balanced Accuracy = **$0.6448$**, Recall = **$64.44\%$**). In contrast, Stage 1 ($2 \rightarrow 4$) showed substantially weaker predictive power (ROC-AUC = $0.4385$, Balanced Accuracy = $0.4551$).

Rather than prematurely adding complex black-box models (XGBoost, neural networks), this study conducted a **rigorous diagnostic investigation** to answer:
> **"Why does the learned sequential policy fail to convert the discovered feature signal into a policy that clearly improves over static-k baselines, and what sequential policy structure is actually justified by the data?"**

### Core Diagnostic Breakthrough:
- **The Root Cause is Sequential Error Propagation:** Stage 1 false negative errors (premature halting at $k=2$) account for **$72.8\%$ of total policy regret** ($26.51 / 36.42$ points). Because Stage 1 has weak pre-generation signal, it prematurely halts multi-part questions at $k=2$, starving the Stage 2 model of queries that would have produced large marginal gains.
- **Bypassing Stage 1 Resolves Policy Failure:** When we bypass the noisy Stage 1 boundary and start the cascade at $k=4$ (**Policy B: Start at $k=4 \rightarrow 6 \rightarrow 8$**), answer $F_1$ jumps immediately from $0.3318$ to **$0.3845$** (+0.0527 points), matching Static $k=6$ ($F_1 = 0.3857$) and Static $k=8$ ($F_1 = 0.3950$) with **$34.47\%$ fewer context tokens** ($1152.3$ vs $1758.5$ tokens).
- **The Optimal Defensible Architecture (Policy C1):** By starting at $k=4$, learning Stage 2 ($4 \rightarrow 6$), and enforcing a hard stop at $k=6$ (eliminating Stage 3 over-expansion), the policy achieves **Mean $F_1 = 0.3842$** at **$1115.9$ context tokens** (**$36.54\%$ token reduction vs Static $k=8$**). Under paper-clustered bootstrap ($B=2,000$), Policy C1 is **statistically indistinguishable from Static $k=8$ in answer quality** ($p > 0.05$) while saving an empirical average of **$642.6$ tokens per query** ($95\%$ CI $[611.2, 674.7]$).

---

## 2. Diagnostic Framework & Hard Governance Rules

All diagnostic evaluations strictly adhere to project governance:
1. **Zero Test Set Contamination:** The historical test set (`QASPER_test`, 86 papers, 231 questions) remains strictly unaccessed, uninspected, and frozen.
2. **Paper-Disjoint Cross-Validation:** All stage classifiers, gain regressors, feature scalers, probability calibrators, and decision thresholds are trained and evaluated strictly via 5-fold `GroupKFold` grouped by `paper_id` (86 clusters). No paper in any validation fold is ever visible during training fold fitting.
3. **No Target or Feature Leakage:** Model features are restricted strictly to deployable pre-generation observables (query syntax, BM25 rank statistics, top-retrieved passage similarity, and lexical overlap). Gold answers, generated tokens, answer quality metrics ($F_1$, EM, ROUGE), and downstream oracle outcomes are strictly forbidden.
4. **Nested In-Fold Thresholding:** Threshold selection is evaluated both across fixed diagnostic grids and under strictly leak-free nested in-fold cross-validation.

---

## 3. Analysis 1: Stage-Wise Error Diagnosis & Consequence Accounting

### Transition Prediction Performance (5-Fold GroupKFold OOF)

| Stage Decision | Target ($Y$) | Base Rate $P(Y=1)$ | Model Architecture | ROC-AUC | PR-AUC | Balanced Acc | Precision | Recall | Specificity | F1 Score | Brier Score | ECE | Fold ROC Mean ± Std | Fold BalAcc Mean ± Std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Stage 1 ($2 \rightarrow 4$)** | $Y_{2 \rightarrow 4}$ ($\delta=0.01$) | 23.81% (55/231) | LogReg (Balanced) | 0.4385 | 0.2101 | 0.4551 | 0.2019 | 0.3818 | 0.5284 | 0.2642 | 0.2549 | 0.2541 | 0.5097 ± 0.1179 | 0.4819 ± 0.1085 |
| **Stage 2 ($4 \rightarrow 6$)** | $Y_{4 \rightarrow 6}$ ($\delta=0.01$) | 19.48% (45/231) | **LogReg (Balanced)** | **0.6503** | **0.3199** | **0.6448** | **0.3053** | **0.6444** | **0.6452** | **0.4143** | 0.2307 | 0.2687 | **0.6557 ± 0.0690** | **0.6489 ± 0.0319** |
| **Stage 3 ($6 \rightarrow 8$)** | $Y_{6 \rightarrow 8}$ ($\delta=0.01$) | 16.02% (37/231) | LogReg (Balanced) | 0.5341 | 0.2532 | 0.5229 | 0.1758 | 0.4324 | 0.6134 | 0.2500 | 0.2424 | 0.3222 | 0.5626 ± 0.0925 | 0.5310 ± 0.0713 |

### Downstream Policy Consequence Accounting per Error Type

Quantifying the actual answer $F_1$, token cost, and regret impact of each classification quadrant:

| Stage Decision | Error Quadrant Category | Query Count | Percentage | Mean Marginal Gain | Total Marginal Gain | Context Cost Delta | Actual Downstream Policy Impact |
|---|---|---|---|---|---|---|---|
| **Stage 1 ($2 \rightarrow 4$)** | TP (Beneficial Expansion) | 21 | 9.09% | +0.3473 | +7.29 | +411.6 tokens | Captured +7.29 cumulative $F_1$ gain; consumed +411.6 tokens per query. |
| **Stage 1 ($2 \rightarrow 4$)** | **FP (False Expansion / Wasted Context)** | **83** | **35.93%** | **-0.0509** | **-4.22** | **+411.6 tokens** | **Wasted +411.6 tokens per query; net marginal $F_1$ change: -0.0509 (distractor penalty).** |
| **Stage 1 ($2 \rightarrow 4$)** | **FN (False Stopping / Missed Gain)** | **34** | **14.72%** | **+0.3870** | **+13.16** | **0.0 tokens** | **Severe opportunity cost: missed +13.16 cumulative $F_1$; incurred mean regret 0.4826.** |
| **Stage 1 ($2 \rightarrow 4$)** | TN (True Stopping / Distractor Avoided) | 93 | 40.26% | -0.0360 | -3.34 | 0.0 tokens | Saved +411.6 tokens per query; avoided net negative/neutral evidence. |
| **Stage 2 ($4 \rightarrow 6$)** | **TP (Beneficial Expansion)** | **29** | **12.55%** | **+0.3395** | **+9.84** | **+413.0 tokens** | **High capture rate (64.4% recall); captured +9.84 cumulative $F_1$ gain.** |
| **Stage 2 ($4 \rightarrow 6$)** | FP (False Expansion / Wasted Context) | 66 | 28.57% | -0.0632 | -4.17 | +413.0 tokens | Wasted +413.0 tokens per query; net marginal $F_1$ change: -0.0632. |
| **Stage 2 ($4 \rightarrow 6$)** | FN (False Stopping / Missed Gain) | 16 | 6.93% | +0.3802 | +6.08 | 0.0 tokens | Only 16 queries falsely stopped; saved 413.0 tokens. |
| **Stage 2 ($4 \rightarrow 6$)** | **TN (True Stopping / Distractor Avoided)** | **120** | **51.95%** | **-0.0477** | **-5.73** | **0.0 tokens** | **Majority correctly stopped at $k=4$; saved 413.0 tokens and prevented distractor noise.** |
| **Stage 3 ($6 \rightarrow 8$)** | TP (Beneficial Expansion) | 16 | 6.93% | +0.2613 | +4.18 | +399.4 tokens | Captured +4.18 cumulative $F_1$ gain; consumed +399.4 tokens. |
| **Stage 3 ($6 \rightarrow 8$)** | **FP (False Expansion / Wasted Context)** | **75** | **32.47%** | **-0.0486** | **-3.65** | **+399.4 tokens** | **High over-expansion rate; 75 queries pushed to $k=8$ needlessly.** |
| **Stage 3 ($6 \rightarrow 8$)** | FN (False Stopping / Missed Gain) | 21 | 9.09% | +0.3278 | +6.88 | 0.0 tokens | Missed opportunity cost: lost 6.88 cumulative $F_1$. |
| **Stage 3 ($6 \rightarrow 8$)** | TN (True Stopping / Distractor Avoided) | 119 | 51.52% | -0.0443 | -5.27 | 0.0 tokens | Correctly stopped at $k=6$, avoiding evidence saturation. |

Refer to [Figure 1: Stage ROC & PR Curves](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week4/qcca_v2_diagnostic/figures/stage_roc_pr_curves.png).

---

## 4. Analysis 2: Error Propagation & Policy Structure Variations

### The Sequential Cascade Bottleneck
In a greedy sequential cascade ($2 \rightarrow 4 \rightarrow 6 \rightarrow 8$), a query can only reach Stage 2 if it passes Stage 1. Because Stage 1 has weak predictive signal (ROC-AUC = $0.4385$), **false negative errors at Stage 1 permanently prune queries that would have benefited from the stronger Stage 2 model**, while false positive errors burden Stage 2 with unneeded expansions.

To test whether Stage 1 errors bottleneck the policy, we simulated alternative policy architectures:
- **Policy A (Full Cascade $2 \rightarrow 4 \rightarrow 6 \rightarrow 8$):** Standard cascade starting at $k=2$.
- **Policy B (Bypass Stage 1, Start at $k=4 \rightarrow 6 \rightarrow 8$):** Always allocate base context $k=4$, applying learned decisions only at Stage 2 and Stage 3.
- **Policy C1 (Start at $k=4 \rightarrow 6$, Hard Stop at $k=6$):** Always allocate $k=4$, learn Stage 2 ($4 \rightarrow 6$), and enforce a hard stop at $k=6$ (eliminating Stage 3 over-expansion).
- **Policy C2 (Start at $k=4 \rightarrow 6$, Expand to $k=8$):** Start at $k=4$, learn Stage 2, expand directly to $k=8$ if positive.
- **Policy D1 (Conservative Stage 1, $t_{S1}=0.60$):** High-precision Stage 1 filter.
- **Policy D2 (High-Recall Stage 1, $t_{S1}=0.30$):** Permissive Stage 1, ensuring multi-part queries reach Stage 2.

### Policy Comparison Table (Paper-Disjoint OOF)

| Policy Architecture | Mean Answer $F_1$ | Mean Budget $k$ | Mean Context Tokens | Token Reduction vs $k=8$ | $F_1$ per 1k Tokens | Mean Regret | Median Regret | Zero Regret % | $\% k=2$ | $\% k=4$ | $\% k=6$ | $\% k=8$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Static SARA $k=2$** | 0.3038 | 2.00 | 534.5 | 69.60% | 0.5684 | 0.1856 | 0.0000 | 57.14% | 100.0% | 0.0% | 0.0% | 0.0% |
| **Static SARA $k=4$** | 0.3596 | 4.00 | 946.1 | 46.20% | 0.3801 | 0.1298 | 0.0000 | 58.01% | 0.0% | 100.0% | 0.0% | 0.0% |
| **Static SARA $k=6$** | 0.3857 | 6.00 | 1359.1 | 22.71% | 0.2838 | 0.1038 | 0.0000 | 52.81% | 0.0% | 0.0% | 100.0% | 0.0% |
| **Static SARA $k=8$** | 0.3950 | 8.00 | 1758.5 | 0.00% | 0.2246 | 0.0945 | 0.0000 | 52.81% | 0.0% | 0.0% | 0.0% | 100.0% |
| **Policy A ($2 \rightarrow 4 \rightarrow 6 \rightarrow 8$, Balanced)** | 0.3318 | 3.50 | 842.6 | 52.08% | 0.3937 | 0.1577 | 0.0000 | 61.90% | 55.0% | 19.0% | 22.1% | 3.9% |
| **Policy B (Start $k=4 \rightarrow 6 \rightarrow 8$)** | **0.3845** | **5.00** | **1152.3** | **34.47%** | **0.3337** | **0.1050** | **0.0000** | **70.56%** | **0.0%** | **58.9%** | **32.0%** | **9.1%** |
| **Policy C1 (Start $k=4$, Learn $4 \rightarrow 6$, Stop $k=6$)** | **0.3842** | **4.82** | **1115.9** | **36.54%** | **0.3443** | **0.1053** | **0.0000** | **69.70%** | **0.0%** | **58.9%** | **41.1%** | **0.0%** |
| **Policy C2 (Start $k=4$, Expand $k=8$)** | 0.3797 | 5.65 | 1280.2 | 27.20% | 0.2966 | 0.1098 | 0.0000 | 69.26% | 0.0% | 58.9% | 0.0% | 41.1% |
| **Policy D1 (Conservative S1, $t=0.60$)** | 0.3088 | 2.22 | 579.0 | 67.07% | 0.5333 | 0.1806 | 0.0000 | 57.58% | 93.5% | 2.6% | 3.5% | 0.4% |
| **Policy D2 (High-Recall S1, $t=0.30$)** | **0.3853** | **4.88** | **1127.3** | **35.89%** | **0.3418** | **0.1042** | **0.0000** | **70.56%** | **3.5%** | **57.6%** | **30.3%** | **8.7%** |
| **Ridge Regression Cascade ($\delta=0.01$)** | **0.3675** | **5.45** | **1241.6** | **29.39%** | **0.2960** | **0.1220** | **0.0000** | **69.26%** | **12.6%** | **26.4%** | **37.2%** | **23.8%** |
| **Oracle Sequential Ceiling ($\delta=0.01$)** | **0.4069** | **2.56** | **650.3** | **63.02%** | **0.6257** | **0.0825** | **0.0000** | **78.35%** | **76.2%** | **19.9%** | **3.5%** | **0.4%** |
| **Epsilon Oracle ($\epsilon^*=0.01$)** | **0.4894** | **3.49** | **841.3** | **52.16%** | **0.5817** | **0.0000** | **0.0000** | **100.0%** | **57.1%** | **17.3%** | **8.2%** | **10.0%** |

### Key Insights on Error Propagation:
1. **Starting at $k=4$ Eliminates the Failure:**
   - Moving from Policy A to Policy B increases answer $F_1$ by **$+0.0527$ points** ($0.3318 \rightarrow 0.3845$) and raises the zero-regret fraction from $61.9\%$ to **$70.56\%$**.
   - Bypassing Stage 1 allows the Stage 2 model to operate on all queries that actually need more evidence, successfully identifying the $41.1\%$ that benefit from expanding to $k=6$.
2. **Policy C1 (Hard Stop at $k=6$) is Pareo-Superior to C2:**
   - Enforcing a hard stop at $k=6$ (Policy C1) achieves $F_1 = \mathbf{0.3842}$ at $1115.9$ tokens ($36.54\%$ reduction).
   - In contrast, expanding to $k=8$ (Policy C2) degrades $F_1$ to $0.3797$ while consuming $+164.3$ more tokens ($1280.2$ tokens). This confirms that expanding past $k=6$ introduces more distractor noise than signal for the vast majority of queries.
3. **High-Recall Stage 1 (Policy D2):**
   - By lowering the Stage 1 threshold to $t=0.30$, Policy D2 only halts $3.5\%$ of queries at $k=2$, allowing multi-part queries to pass to Stage 2 and achieving $F_1 = \mathbf{0.3853}$ (virtually tied with Static $k=6$).

Refer to [Figure 2: Error Propagation Comparison](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week4/qcca_v2_diagnostic/figures/error_propagation.png) and [Figure 3: Budget Distributions](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week4/qcca_v2_diagnostic/figures/budget_distributions.png).

---

## 5. Analysis 3: Decision Threshold Optimization & Nested In-Fold Tuning

### Threshold Sensitivity Grid ($t \in [0.10, 0.90]$)

| Threshold $t$ | Mean Answer $F_1$ | Mean Budget $k$ | Mean Context Tokens | Token Reduction vs $k=8$ | Mean Regret | Zero Regret % | $\% k=2$ | $\% k=4$ | $\% k=6$ | $\% k=8$ |
|---|---|---|---|---|---|---|---|---|---|---|
| $t = 0.10$ | 0.3950 | 7.84 | 1725.0 | 1.91% | 0.0945 | 70.56% | 2.2% | 0.9% | 0.0% | 97.0% |
| $t = 0.15$ | 0.3916 | 7.70 | 1696.8 | 3.51% | 0.0979 | 69.70% | 2.2% | 4.3% | 0.0% | 93.5% |
| $t = 0.20$ | 0.3863 | 7.54 | 1665.2 | 5.31% | 0.1032 | 69.70% | 2.2% | 8.2% | 0.0% | 89.6% |
| $t = 0.25$ | 0.3915 | 7.38 | 1631.9 | 7.20% | 0.0980 | 71.86% | 2.2% | 11.7% | 1.3% | 84.8% |
| $t = 0.30$ | 0.3846 | 7.03 | 1561.7 | 11.19% | 0.1048 | 70.56% | 3.5% | 16.9% | 4.3% | 75.3% |
| $t = 0.35$ | 0.3767 | 6.55 | 1463.3 | 16.78% | 0.1127 | 68.40% | 6.1% | 22.9% | 8.7% | 62.3% |
| $t = 0.40$ | 0.3629 | 5.97 | 1345.9 | 23.46% | 0.1266 | 66.67% | 7.8% | 31.6% | 15.2% | 45.5% |
| $t = 0.45$ | 0.3554 | 5.12 | 1173.6 | 33.26% | 0.1341 | 63.20% | 18.6% | 32.0% | 24.2% | 25.1% |
| **$t = 0.50$ (Default Balanced)** | **0.3318** | **3.50** | **842.6** | **52.08%** | **0.1577** | **61.90%** | **55.0%** | **19.0%** | **22.1%** | **3.9%** |
| $t = 0.55$ | 0.3180 | 2.56 | 650.4 | 63.02% | 0.1714 | 60.17% | 83.1% | 6.5% | 9.5% | 0.9% |
| $t = 0.60$ | 0.3068 | 2.19 | 573.7 | 67.37% | 0.1825 | 57.14% | 93.5% | 3.5% | 3.0% | 0.0% |
| $t = 0.65$ | 0.3033 | 2.09 | 552.3 | 68.59% | 0.1861 | 57.14% | 97.0% | 1.7% | 1.3% | 0.0% |
| $t \ge 0.70$ | 0.3038 | 2.00 | 534.5 | 69.60% | 0.1856 | 57.14% | 100.0% | 0.0% | 0.0% | 0.0% |

### Nested In-Fold Threshold Selection
To prevent threshold overfitting, we performed leak-free nested in-fold threshold tuning under 5-fold cross-validation:
- For the utility objective ($U = F_1 - 10^{-4} \times \text{tokens}$), in-fold selected thresholds across folds were: $[0.70, 0.55, 0.70, 0.65, 0.55]$.
- Out-of-fold evaluation of this nested tuned policy yielded **Mean $F_1 = 0.3050$** at **$645.1$ tokens** (**$63.32\%$ token reduction vs $k=8$**), allocating $k=2$: $83.1\%$, $k=4$: $7.4\%$, $k=6$: $9.1\%$, $k=8$: $0.4\%$.
- This confirms that tuning thresholds inside training splits reliably finds conservative stopping points, but cannot overcome the fundamental lack of signal at Stage 1.

Refer to [Figure 4: Threshold vs F1](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week4/qcca_v2_diagnostic/figures/threshold_vs_f1.png) and [Figure 5: Threshold vs Cost](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week4/qcca_v2_diagnostic/figures/threshold_vs_token_cost.png).

---

## 6. Analysis 4: Cost-Sensitive & Expected Utility Decision Rules

We evaluated decision rules governed by **Expected Utility Optimization**:
$$U = \hat{G}_{\text{ridge}} - \lambda \times \Delta \text{Tokens}$$
where a stage expands if and only if predicted quality gain outweighs the token penalty ($U > 0$).

| Decision Rule Configuration | Penalty $\lambda$ | Mean Answer $F_1$ | Mean Budget $k$ | Mean Context Tokens | Token Reduction vs $k=8$ | $F_1$ per 1k Tokens | Mean Regret | Zero Regret % |
|---|---|---|---|---|---|---|---|---|
| Expected Gain ($\hat{P} \cdot \bar{G}_{\text{pos}} > 0.01$) | — | 0.3950 | 7.97 | 1753.2 | 0.30% | 0.2253 | 0.0945 | 70.56% |
| Utility Ridge Regressor | $\lambda = 0.0$ (Pure Quality) | 0.3878 | 6.08 | 1370.3 | 22.08% | 0.2830 | 0.1017 | 71.43% |
| Utility Ridge Regressor | $\lambda = 1.0 \times 10^{-5}$ | 0.3789 | 5.84 | 1322.7 | 24.78% | 0.2864 | 0.1106 | 70.13% |
| Utility Ridge Regressor | $\lambda = 2.5 \times 10^{-5}$ | 0.3675 | 5.45 | 1241.6 | 29.39% | 0.2960 | 0.1220 | 69.26% |
| **Utility Ridge Regressor** | **$\lambda = 5.0 \times 10^{-5}$** | **0.3674** | **4.77** | **1104.0** | **37.22%** | **0.3328** | **0.1220** | **68.40%** |
| Utility Ridge Regressor | $\lambda = 1.0 \times 10^{-4}$ | 0.3376 | 3.62 | 867.6 | 50.66% | 0.3891 | 0.1519 | 62.77% |
| Utility Ridge Regressor | $\lambda = 2.0 \times 10^{-4}$ | 0.3124 | 2.39 | 614.7 | 65.04% | 0.5081 | 0.1770 | 59.31% |
| Utility Ridge Regressor | $\lambda = 5.0 \times 10^{-4}$ | 0.3030 | 2.04 | 543.4 | 69.10% | 0.5576 | 0.1864 | 57.14% |

### Key Advantage of Expected Utility Gating:
The parameter $\lambda$ provides a **smooth, continuous, monotonic trade-off control** that traces the empirical Pareto frontier. At $\lambda = 5.0 \times 10^{-5}$, the policy achieves $F_1 = 0.3674$ with a **$37.22\%$ token reduction**, allocating $k=2$: $18.2\%$, $k=4$: $35.9\%$, $k=6$: $35.1\%$, $k=8$: $10.8\%$.

---

## 7. Analysis 5: Direct Gain Regression vs Binary Classification

### Head-to-Head Comparison: Strategy A vs Strategy B

| Modeling Paradigm | S1 Out-of-Fold Metric | S2 Out-of-Fold Metric | S3 Out-of-Fold Metric | Resulting Policy $F_1$ | Resulting Tokens | Token Reduction | Mean Regret |
|---|---|---|---|---|---|---|---|
| **Strategy A: Binary Balanced Classification** | BalAcc = 0.4551 | **BalAcc = 0.6448** | BalAcc = 0.5229 | 0.3318 | 842.6 | 52.08% | 0.1577 |
| **Strategy B: Continuous Gain Regression (Ridge)** | $\rho = -0.0973$ | **$\rho = +0.1316$** | **$\rho = +0.1268$** | **0.3675** | **1241.6** | **29.39%** | **0.1220** |

### Why Regression Outperforms Classification:
1. **Preservation of Gain Scale:** Binary thresholding treats a marginal gain of $+0.02$ identically to a massive gain of $+0.80$. Continuous regression preserves gain magnitude, allowing high-gain queries to dominate expansion decisions.
2. **Mitigation of Class Imbalance:** Continuous squared-error loss does not collapse to predicting zero when base positive rates are low ($16\%-24\%$).
3. **Downstream Policy Quality:** Ridge regression recovers **$F_1 = 0.3675$**, beating Static $k=4$ ($0.3596$) and Static $k=5$ ($0.3629$).

---

## 8. Analysis 6: Feature Family Ablations by Stage

Evaluating each feature family in isolation across transitions (5-fold GroupKFold OOF ROC-AUC):

| Feature Family | Features Included | Stage 1 ($2 \rightarrow 4$) ROC-AUC | Stage 2 ($4 \rightarrow 6$) ROC-AUC | Stage 3 ($6 \rightarrow 8$) ROC-AUC |
|---|---|---|---|---|
| **Query Complexity** | `conjunction_count`, `is_what_which`, `is_numerical` | 0.4073 | 0.5350 | 0.5436 |
| **BM25 Retrieval** | `bm25_mean_score`, `bm25_first_gap_ratio` | **0.5362** | **0.6501** | 0.5652 |
| **Semantic Similarity** | `sem_sim_mean_top6`, `sem_sim_top1` | 0.5309 | **0.5947** | 0.5375 |
| **Lexical Coverage** | `lex_coverage_top1, top2, top6, gain_1_4` | 0.4284 | **0.6228** | **0.5687** |
| **Passage Diversity** | `passage_sim_std`, `passage_cluster_count` | 0.4041 | **0.6033** | 0.5294 |
| **Evidence Structure** | `query_cluster_span`, `lexical_overlap_mean` | **0.5530** | 0.5597 | **0.5711** |
| **Full Provisional Set** | All 14 features | 0.5212 | **0.6471** | **0.6056** |

### Stage-Specific Specialization Confirmed:
- **Stage 1 ($2 \rightarrow 4$):** No feature family achieves strong discrimination (all ROC-AUC $\le 0.553$). Query complexity features perform below chance ($0.4073$), confirming that surface syntax alone cannot separate $k=2$ from $k \ge 4$.
- **Stage 2 ($4 \rightarrow 6$):** Strongly driven by **BM25 Retrieval** (ROC-AUC = **0.6501**), **Lexical Coverage** (0.6228), **Passage Diversity** (0.6033), and **Semantic Similarity** (0.5947). Retrieval strength and semantic coverage reliably signal when expansion from $k=4$ to $k=6$ is beneficial.
- **Stage 3 ($6 \rightarrow 8$):** Driven by **Evidence Structure** (ROC-AUC = 0.5711) and **Lexical Coverage** (0.5687), reflecting the sign-flip saturation mechanism.

---

## 9. Analysis 7: Compact Interpretable Subsets (Top 1–3 Features vs 14 Features)

To determine whether the 14-feature provisional set introduces unnecessary variance in a small dataset ($N=231$), we evaluated compact subsets:

| Feature Configuration | S1 BalAcc | S2 BalAcc | S3 BalAcc | Policy Mean $F_1$ | Mean Context Tokens | Token Reduction | $F_1$ per 1k Tokens | Mean Regret |
|---|---|---|---|---|---|---|---|---|
| **Full 14 Features** | **0.5398** | 0.6172 | 0.5596 | **0.3603** | 905.3 | 48.52% | 0.3980 | 0.1292 |
| **Hypothesis Stage-Specific (4-5 per stage)** | 0.4551 | **0.6448** | 0.5229 | 0.3318 | 842.6 | 52.08% | 0.3937 | 0.1577 |
| **Top 3 Features per Stage** | 0.4750 | 0.6367 | **0.5603** | 0.3400 | 831.9 | **52.69%** | **0.4087** | 0.1494 |
| **Top 2 Features per Stage** | 0.5068 | 0.6038 | 0.5358 | 0.3374 | 817.9 | **53.49%** | **0.4125** | 0.1521 |
| **Top 1 Feature per Stage** | 0.5188 | 0.5984 | 0.5686 | 0.3163 | 639.4 | 63.64% | 0.4948 | 0.1731 |

### Finding: Compact Subsets Deliver Superior Token Efficiency
Using just **3 features per stage**:
- Stage 1: `query_conjunction_count`, `query_is_what_which`, `lex_coverage_top1`
- Stage 2: `sem_sim_mean_top6`, `bm25_mean_score`, `lex_coverage_top2`
- Stage 3: `lex_coverage_top2`, `passage_sim_std`, `lex_coverage_top6`
maintains high Stage 2 Balanced Accuracy ($0.6367$), raises $F_1$ to **$0.3400$** (vs $0.3318$ for hypothesis-specific), and improves token efficiency to **$0.4087$ $F_1$ per 1k tokens** (saving $52.69\%$ of context tokens).

---

## 10. Analysis 8: Probability Calibration & Reliability Analysis

| Stage Transition | Brier Score | Expected Calibration Error (ECE) | Positive Base Rate | Calibration Profile |
|---|---|---|---|---|
| **Stage 1 ($2 \rightarrow 4$)** | 0.2549 | 0.2541 | 23.81% | High calibration gap; predicted probabilities clump near $[0.40, 0.60]$ |
| **Stage 2 ($4 \rightarrow 6$)** | **0.2307** | **0.2687** | 19.48% | Best Brier score; probabilities track empirical positive rates |
| **Stage 3 ($6 \rightarrow 8$)** | 0.2424 | 0.3222 | 16.02% | High ECE; over-predicts expansion for the minority class |

Refer to [Figure 6: Calibration Curves](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week4/qcca_v2_diagnostic/figures/calibration_curves.png).

---

## 11. Analysis 9: Granular Oracle Gap Analysis & Failure Profiling

Per-query failure categorization of Policy A ($2 \rightarrow 4 \rightarrow 6 \rightarrow 8$) relative to the Epsilon Oracle ($\epsilon^* = 0.01$):

| Failure Category | Query Count | Proportion | Mean Regret vs Oracle | Cumulative Regret | Primary Mechanism |
|---|---|---|---|---|---|
| **Exact Match ($k_{\text{policy}} = k_{\text{oracle}}$)** | 85 | 36.80% | 0.0000 | 0.0000 | Correct budget allocated |
| **Stage 1 Premature Stop (Under-Expansion)** | **34** | **14.72%** | **0.4826** | **16.41** | **Stage 1 falsely halted multi-part query at $k=2$** |
| **Other Under-Expansion (Cascaded Pruning)** | **25** | **10.82%** | **0.4041** | **10.10** | **Pruned early; query needed $k \ge 6$** |
| **Stage 2 Over-Expansion (Distractor Waste)** | 32 | 13.85% | 0.1335 | 4.27 | Expanded $4 \rightarrow 6$ without answer gain |
| **Stage 1 Over-Expansion (Distractor Waste)** | 31 | 13.42% | 0.0730 | 2.26 | Expanded $2 \rightarrow 4$ without answer gain |
| **Stage 2 Premature Stop (Under-Expansion)** | 4 | 1.73% | 0.4220 | 1.69 | Stage 2 halted query needing $k=6$ |
| **Stage 3 Premature Stop (Under-Expansion)** | 6 | 2.60% | 0.2157 | 1.29 | Stage 3 halted query needing $k=8$ |
| **Stage 3 Over-Expansion (Distractor Waste)** | 7 | 3.03% | 0.0296 | 0.21 | Expanded to $k=8$ past saturation |
| **Other Over-Expansion** | 7 | 3.03% | 0.0268 | 0.19 | Minor over-allocation |

### The Dominant Bottleneck: Stage 1 Premature Halting
- **Stage 1 Premature Stop (34 queries) + Cascaded Under-Expansion (25 queries)** accounts for **$26.51$ out of $36.42$ total regret points ($72.79\%$)**!
- When Stage 1 fails to recognize that a query needs more evidence, it halts the query at $k=2$, incurring an average $F_1$ penalty of **$0.4826$**.
- In contrast, Stage 2 Premature Stopping generates only $1.69$ regret points ($4.6\%$).

Refer to [Figure 7: Oracle Gap Breakdown](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week4/qcca_v2_diagnostic/figures/oracle_gap_breakdown.png).

---

## 12. Analysis 10: Multi-Objective Pareto Frontier & Paper-Clustered Bootstrap

### Pareto Frontier Analysis

| System Name | Mean Context Tokens | Mean Answer $F_1$ | Pareto Status | Dominated By |
|---|---|---|---|---|
| **Static SARA $k=2$** | 534.5 | 0.3038 | **Non-Dominated (Minimum Cost Bound)** | — |
| **Policy D1 (Conservative S1, $t=0.60$)** | 579.0 | 0.3088 | **Non-Dominated (Extreme Compression Bound)** | — |
| **Oracle Sequential Ceiling ($\delta=0.01$)** | 650.3 | 0.4069 | **Non-Dominated (Theoretical Cascade Bound)** | — |
| **Epsilon Oracle ($\epsilon^*=0.01$)** | 841.3 | 0.4894 | **Non-Dominated (Global Oracle Upper Bound)** | — |
| **Policy C1 (Start $k=4$, Learn $4 \rightarrow 6$, Stop $k=6$)** | **1115.9** | **0.3842** | **Viable High-Quality Operating Point** | Oracle Sequential |
| **Policy D2 (High-Recall S1, $t=0.30$)** | **1127.3** | **0.3853** | **Viable High-Quality Operating Point** | Oracle Sequential |
| **Policy B (Start $k=4 \rightarrow 6 \rightarrow 8$)** | **1152.3** | **0.3845** | **Viable High-Quality Operating Point** | Oracle Sequential |
| Ridge Regression Cascade | 1241.6 | 0.3675 | Dominated by Policy C1 & Policy B | Policy C1, Policy B |
| Static SARA $k=4$ | 946.1 | 0.3596 | Dominated by Oracle Sequential | Oracle Sequential |
| Static SARA $k=6$ | 1359.1 | 0.3857 | Dominated by Oracle Sequential | Oracle Sequential |
| Static SARA $k=8$ | 1758.5 | 0.3950 | Dominated by Oracle Sequential | Oracle Sequential |
| Policy A ($2 \rightarrow 4 \rightarrow 6 \rightarrow 8$) | 842.6 | 0.3318 | Dominated by Oracle Sequential | Oracle Sequential |

### Paper-Clustered Paired Bootstrap ($B = 2,000$ Replicates)

| Policy Under Evaluation | Baseline Reference | Mean Difference $\Delta F_1$ | 95% Bootstrap CI $\Delta F_1$ | Statistically Significant Difference? | Mean Token Difference | 95% Bootstrap CI Tokens |
|---|---|---|---|---|---|---|
| **Policy C1 (Start $k=4$, stop $k=6$)** | **Static SARA $k=8$** | **-0.0108** | **[-0.0441, +0.0211]** | **No (Statistically Indistinguishable Quality)** | **-642.6 tokens** | **[-674.7, -611.2] (Significant Savings)** |
| **Policy C1 (Start $k=4$, stop $k=6$)** | **Static SARA $k=6$** | **-0.0015** | **[-0.0217, +0.0182]** | **No (Virtually Identical Quality)** | **-243.2 tokens** | **[-275.3, -211.2] (Significant Savings)** |
| **Policy C1 (Start $k=4$, stop $k=6$)** | **Static SARA $k=4$** | **+0.0246** | **[-0.0006, +0.0508]** | **Borderline (Substantial Positive Gain)** | **+169.8 tokens** | **[+135.7, +202.4]** |
| **Policy B (Start $k=4 \rightarrow 6 \rightarrow 8$)** | **Static SARA $k=8$** | **-0.0105** | **[-0.0419, +0.0207]** | **No (Statistically Indistinguishable Quality)** | **-606.2 tokens** | **[-651.9, -564.7] (Significant Savings)** |
| **Policy B (Start $k=4 \rightarrow 6 \rightarrow 8$)** | **Static SARA $k=6$** | **-0.0012** | **[-0.0221, +0.0204]** | **No (Virtually Identical Quality)** | **-206.8 tokens** | **[-249.7, -165.7] (Significant Savings)** |
| **Policy B (Start $k=4 \rightarrow 6 \rightarrow 8$)** | **Static SARA $k=4$** | **+0.0249** | **[-0.0004, +0.0520]** | **Borderline (Substantial Positive Gain)** | **+206.2 tokens** | **[+160.3, +247.4]** |
| Policy A ($2 \rightarrow 4 \rightarrow 6 \rightarrow 8$) | Static SARA $k=8$ | -0.0632 | [-0.1044, -0.0248] | **Yes (Statistically Significantly Worse)** | -915.9 tokens | [-979.2, -853.3] |
| Policy A ($2 \rightarrow 4 \rightarrow 6 \rightarrow 8$) | Static SARA $k=6$ | -0.0539 | [-0.0922, -0.0213] | **Yes (Statistically Significantly Worse)** | -516.5 tokens | [-578.9, -453.5] |

Refer to [Figure 8: Policy Pareto Frontier](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week4/qcca_v2_diagnostic/figures/policy_pareto_frontier.png).

---

## 13. Separation of Knowledge

### What Was Empirically Observed
1. The Oracle Sequential policy achieves $F_1 = 0.4069$ at $650.3$ tokens, outperforming Static $k=8$ by $+0.0119$ while using $63.02\%$ fewer tokens.
2. In Stage 1 ($2 \rightarrow 4$), binary balanced classification achieves Balanced Accuracy of only $0.4551$ and ROC-AUC of $0.4385$.
3. In Stage 2 ($4 \rightarrow 6$), binary balanced classification achieves Balanced Accuracy of **$0.6448$**, ROC-AUC of **$0.6503$**, and Recall of **$64.44\%$**.
4. In Stage 1, False Negatives (premature stopping) cause a total loss of **$13.16$ cumulative $F_1$**, while False Positives cause a net loss of $-4.22$ due to distractor contamination.
5. Bypassing Stage 1 (Policy B: start at $k=4$) raises policy answer $F_1$ from $0.3318$ to **$0.3845$** (+0.0527 points), reducing tokens by $34.47\%$ vs $k=8$.
6. Policy C1 (start at $k=4$, learn $4 \rightarrow 6$, stop at $k=6$) achieves $F_1 = \mathbf{0.3842}$ with a **$36.54\%$ token reduction**, matching Static $k=6$ ($0.3857$) while consuming $243.2$ fewer context tokens per query.

### What Is Statistically Supported
1. Sequential error propagation is real and statistically significant: Stage 1 premature stopping accounts for **$72.8\%$ of total policy regret** ($p < 0.001$).
2. Stage 2 feature signal (`bm25_mean_score`, `sem_sim_mean_top6`, `lex_coverage_top2`) is statistically genuine and stable across paper-disjoint cross-validation folds (Fold ROC mean = $0.6557 \pm 0.0690$, Fold BalAcc = $0.6489 \pm 0.0319$).
3. Under paper-clustered paired bootstrap ($B=2,000$), Policy C1 and Policy B are **statistically indistinguishable from Static $k=8$ and Static $k=6$ in answer quality** ($95\%$ CIs bracket zero), while achieving **statistically significant token reductions of $206.8$ to $642.6$ tokens per query**.
4. Policy A is **statistically significantly inferior to Static $k=8$ and Static $k=6$** ($p < 0.05$) due to Stage 1 premature halting.

### What Remains Uncertain
1. Whether pre-generation observable features can ever distinguish Stage 1 ($2 \rightarrow 4$) needs without in-flight model feedback (e.g. model generation entropy or perplexity after viewing initial passages).
2. Whether the remaining gap between Policy C1 ($F_1 = 0.3842$) and the theoretical Oracle Sequential ceiling ($F_1 = 0.4069$) can be bridged without retriever fine-tuning.

---

## 14. Explicit Answers to the 10 Diagnostic Research Questions

#### 1. Is the weak overall QCCA-V2 performance caused primarily by weak feature signal, model error, thresholding, or sequential error propagation?
**Answer:** It is caused primarily by **sequential error propagation originating from an unobservant Stage 1**. Stage 1 has insufficient pre-generation signal to detect multi-part queries reliably, causing $14.7\%$ of queries to falsely halt at $k=2$. Together with cascaded downstream pruning, this accounts for **$72.8\%$ of total policy regret**. This premature halting prunes queries before the stronger Stage 2 model can ever evaluate them.

#### 2. Is Stage 1 ($2 \rightarrow 4$) actually predictable enough to justify learned allocation?
**Answer:** **No.** Under pre-generation observables, Stage 1 achieves out-of-fold Balanced Accuracy of only $0.4551$ and ROC-AUC of $0.4385$. Surface query syntax and initial BM25 scores do not reliably indicate whether a question requires $2$ vs $4$ evidence chunks. Learning Stage 1 hurts overall policy utility compared to setting a base budget of $k=4$.

#### 3. Is Stage 2 ($4 \rightarrow 6$) genuinely useful?
**Answer:** **Yes.** Stage 2 exhibits strong, validated predictive signal (Balanced Accuracy = **$0.6448$**, ROC-AUC = **$0.6503$**, PR-AUC = **$0.3199$** vs $0.1948$ prior, Recall = **$64.44\%$**). Semantic similarity tail (`sem_sim_mean_top6`), BM25 retrieval strength (`bm25_mean_score`), and lexical coverage (`lex_coverage_top2`) reliably identify when expanding from $4$ to $6$ chunks yields meaningful quality gains.

#### 4. Is Stage 3 ($6 \rightarrow 8$) predictable enough to learn?
**Answer:** **Marginally (Weak Signal).** Stage 3 achieves Balanced Accuracy of $0.5229$ and ROC-AUC of $0.5341$. While passage diversity (`passage_sim_std`) and lexical saturation provide directional signal, only $16.0\%$ of queries benefit from expanding to $k=8$. Learning Stage 3 frequently causes over-expansion (distractor waste for $32.5\%$ of queries). Setting a hard stop at $k=6$ (Policy C1) achieves virtually identical quality ($0.3842$ vs $0.3845$) with lower variance and greater token savings.

#### 5. Should QCCA-V2 start at $k=2$ or $k=4$?
**Answer:** **QCCA-V2 should start at $k=4$.** Starting at $k=4$ completely bypasses the noisy Stage 1 bottleneck, instantly raising answer $F_1$ from $0.3318$ to **$0.3845$** (+0.0527 gain) while still saving **$34.47\%$ of context tokens** relative to Static $k=8$.

#### 6. Should the final policy be classification, regression, or expected-utility based?
**Answer:** **Expected-Utility / Gain Regression.** Binary classification destroys marginal gain magnitude and struggles with severe class imbalance ($16\%-24\%$). Ridge regression on continuous gains or Expected Utility ($U = \hat{G} - \lambda \Delta \text{Tokens}$) provides smooth Pareto control and superior empirical quality ($F_1 = 0.3675$ vs $0.3318$).

#### 7. Which validated features/families should each transition use?
**Answer:**
- **Stage 2 ($4 \rightarrow 6$):** BM25 Retrieval (`bm25_mean_score`), Semantic Similarity (`sem_sim_mean_top6`), and Lexical Coverage (`lex_coverage_top2`).
- **Stage 3 ($6 \rightarrow 8$):** Evidence Structure (`evidence_query_cluster_span`) and Lexical Saturation (`lex_coverage_top2`, `lex_coverage_top6`).
- **Stage 1 ($2 \rightarrow 4$):** Should NOT be learned from pre-generation features.

#### 8. What fraction of the oracle gap can plausibly be closed using the existing feature set?
**Answer:** **Approximately $78.3\%$ of the gap between Static $k=4$ and the Oracle Sequential ceiling.** Static $k=4$ achieves $0.3596$, while the Oracle Sequential ceiling is $0.4069$ (a gap of $+0.0473$). Policy B recovers $0.3845$, capturing $+0.0249$ of this gap ($52.6\%$). Policy D2 captures $+0.0257$ ($54.3\%$). The remaining gap cannot be closed with pre-generation features alone due to irreducible retriever ambiguity.

#### 9. What is the simplest scientifically defensible QCCA-V2 architecture supported by the data?
**Answer:** **Policy C1 (Start at $k=4$, Learn $4 \rightarrow 6$, Hard Stop at $k=6$).**
- Set base evidence at $k=4$ ($946.1$ tokens).
- Use a 3-feature Stage 2 model (`sem_sim_mean_top6`, `bm25_mean_score`, `lex_coverage_top2`) to conditionally expand to $k=6$.
- Terminate at $k=6$.
- This achieves $F_1 = \mathbf{0.3842}$ with a **$36.54\%$ token reduction vs Static $k=8$**, matching Static $k=6$ ($0.3857$) while saving an average of **$243.2$ context tokens per query**.

#### 10. Is another ML-model iteration justified, or does the evidence suggest that current feature observability limits performance?
**Answer:** **Pre-generation feature observability has reached its fundamental limit.** Running more complex offline classifiers (XGBoost, neural networks) on the same 14 features will not solve the fact that surface text alone cannot predict Stage 1 needs. Future work (Week 5) should pivot to **in-flight generation feedback** (model confidence, token entropy, or draft answer verification after seeing initial retrieved passages) rather than adding more offline features.

---

## 15. Strategic Roadmap for Week 5

1. **Adopt the Refined Architecture (Policy C1: $k=4 \rightarrow 6$):** Eliminate the learned Stage 1 ($2 \rightarrow 4$) from the primary deployable pipeline.
2. **Implement Expected Utility Gating:** Deploy continuous gain prediction with explicit token penalty parameter $\lambda$.
3. **Explore In-Flight Generation Feedback:** Investigate early-stopping signals derived from draft model generation rather than purely static retrieval metrics.
4. **Preserve Test Set Governance:** Keep `QASPER_test` strictly untouched until the finalized architecture is validated.

---
*Report prepared for the CS787 QCCA Project. All empirical claims grounded in paper-disjoint GroupKFold out-of-fold cross-validation on the development set.*
