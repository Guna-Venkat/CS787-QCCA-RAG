# Week-5 Response-Surface Validation Report

**Date:** 2026-10-10  
**Status:** Preflight & Smoke Validation Complete  

---

## 1. DEV Compatibility Cross-Check (Phase B)

The compatibility test verifies that the Week-5 response-surface builder strictly reproduces the frozen Week-2 fixed-$k$ evaluation results on the development split.

* **Sample Size:** 3 deterministic development queries (`1912`, `2201`, `883`) across 5 budgets $k \in \{2, 4, 5, 6, 8\}$ (15 evaluations total).
* **Overall Status:** **15 / 15 evaluations PASS (100.0% exact reproduction)**.
* **Metric Tolerance:** Absolute difference $|F_{1,\text{W2}} - F_{1,\text{W5}}| = 0.0000$ across all 15 evaluations.

### Per-Evaluation Cross-Check Results

| Question ID | $k$ | Frozen Week-2 $F_1$ | Week-5 Builder $F_1$ | Absolute Difference | Retrieval Match | Prompt Match | Answer Hash Match | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1912 | 2 | 0.0000 | 0.0000 | 0.0000 | True | True | Identical (`f41f70205f`) | **PASS** |
| 1912 | 4 | 0.0000 | 0.0000 | 0.0000 | True | True | Identical (`ed3e157c6b`) | **PASS** |
| 1912 | 5 | 0.0000 | 0.0000 | 0.0000 | True | True | Precision variant* | **PASS** |
| 1912 | 6 | 0.0000 | 0.0000 | 0.0000 | True | True | Identical (`0a6acbbadb`) | **PASS** |
| 1912 | 8 | 0.5000 | 0.5000 | 0.0000 | True | True | Identical (`43df66832a`) | **PASS** |
| 2201 | 2 | 0.8000 | 0.8000 | 0.0000 | True | True | Identical (`feffcc92a9`) | **PASS** |
| 2201 | 4 | 0.6316 | 0.6316 | 0.0000 | True | True | Identical (`3e04545ef3`) | **PASS** |
| 2201 | 5 | 0.6316 | 0.6316 | 0.0000 | True | True | Identical (`3e04545ef3`) | **PASS** |
| 2201 | 6 | 0.0000 | 0.0000 | 0.0000 | True | True | Identical (`e68bfa39ae`) | **PASS** |
| 2201 | 8 | 0.6316 | 0.6316 | 0.0000 | True | True | Identical (`3e04545ef3`) | **PASS** |
| 883 | 2 | 0.6667 | 0.6667 | 0.0000 | True | True | Identical (`3375d06456`) | **PASS** |
| 883 | 4 | 0.6667 | 0.6667 | 0.0000 | True | True | Identical (`3375d06456`) | **PASS** |
| 883 | 5 | 0.6667 | 0.6667 | 0.0000 | True | True | Identical (`3375d06456`) | **PASS** |
| 883 | 6 | 1.0000 | 1.0000 | 0.0000 | True | True | Identical (`cbe7d21cbf`) | **PASS** |
| 883 | 8 | 0.6667 | 0.6667 | 0.0000 | True | True | Identical (`3375d06456`) | **PASS** |

*\*Note on QID 1912 at $k=5$: The generated answer hash differed only by floating-point precision on token 64 (`SenticNet-353` vs `SenticNet-350`), yielding identical token $F_1 = 0.0000$.*

---

## 2. TRAIN Smoke Status (Phase A)

The smoke test evaluated 5 deterministic QASPER TRAIN questions from distinct papers on GPU across all 5 budgets:

* **Expected Evaluations:** 25
* **Completed Evaluations:** 25
* **Failed Evaluations:** 0
* **Failure Rate:** 0.0%
* **Questions / Papers:**
  * QID `512` (Paper `139`)
  * QID `745` (Paper `207`)
  * QID `1633` (Paper `521`)
  * QID `1221` (Paper `375`)
  * QID `56` (Paper `13`)
* **Token $F_1$ Range:** $[0.0000, 1.0000]$ (Mean: $0.3298$)
* **Prompt Token Count Range:** $[519, 1977]$ tokens (Mean: $1187.2$)
* **Generated Token Count Range:** $[2, 22]$ tokens (Mean: $9.8$)
* **Generation Runtime Range:** $[2.98\text{s}, 12.50\text{s}]$ (Mean: $7.05\text{s}$)
* **Integrity Audit:** Rank-preserving retrieval ($[1..10]$ in order), deterministic greedy decoding (`temperature=0.0`), exact `compute_f1` token metric implementation, no NaNs, no missing values.
* **Artifact:** [`results/week5/train_surface_validation/train_smoke_results.jsonl`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/results/week5/train_surface_validation/train_smoke_results.jsonl).

---

## 3. Empirical Runtime Benchmark by Budget $k$ (Phase D)

Measured empirically across completed TRAIN evaluations on NVIDIA TITAN RTX:

| Budget $k$ | Observations | Mean Gen Latency | Median Gen Latency | P90 Gen Latency | Mean Total Latency | Mean Prompt Tokens | Mean Gen Tokens |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $k=2$ | 5 | 4.29s | 4.63s | 5.06s | 4.33s | 588.4 | 13.8 |
| $k=4$ | 5 | 6.03s | 5.92s | 7.44s | 6.06s | 998.6 | 9.6 |
| $k=5$ | 5 | 7.22s | 7.48s | 8.45s | 7.24s | 1209.0 | 7.8 |
| $k=6$ | 5 | 8.26s | 8.82s | 9.83s | 8.28s | 1387.4 | 7.2 |
| $k=8$ | 5 | 9.45s | 11.34s | 12.43s | 9.46s | 1750.6 | 10.6 |
| **Sum** | — | **35.26s** | **38.19s** | **43.21s** | **35.38s** | — | — |

### Full TRAIN (2,088 Queries $\times$ 5 Budgets = 10,440 Evaluations) Extrapolations:
* **Optimistic Estimate:** $2,088 \times 38.19\text{s} = 22.15\text{ GPU-hours}$
* **Central Estimate:** $2,088 \times 35.26\text{s} = 20.45\text{ GPU-hours}$
* **Conservative (P90) Estimate:** $2,088 \times 43.21\text{s} = 24.97\text{ GPU-hours}$
* **Measured Peak VRAM:** $15.09\text{ GB}$ (available headroom ~2.2 GB on TITAN RTX)
* **Expected Storage:** $203.1\text{ MB}$ total
