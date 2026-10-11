# Week 5-B Content-Aware Allocator Live Tracking

**Status:** 🏁 **COMPLETE** (12 / 12 Models Finished)  
**Current Activity:** All runs, bootstrap, and figures generated  
**Last Updated:** 2026-10-11 13:39:14 IST  

---

## 1. Overall Progress

| Metric | Target | Current Value | Status |
| :--- | :---: | :---: | :---: |
| **Models Completed** | 12 | **12 / 12** | **100.0%** 🎯 |
| **Elapsed Time** | — | 0m 42s | Active |
| **Estimated Remaining Time** | — | **0s (Complete)** | — |
| **GPU Status** | NVIDIA TITAN RTX | NVIDIA TITAN RTX | Util: 0 % | VRAM: 7089 MiB / 24576 MiB | Temp: 50°C | Power: 67.23 W | Monitoring |

---

## 2. Model Execution Log

| Condition | Seed | Selected $\tau$ | DEV Policy $F_1$ | Context Tokens | Mean $k$ | Fallback Rate | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $I1$-T3 | 42 | 0.70 | **0.3771** | 1033.0 | 4.46 | 17.8% | ✅ Done |
| $I1$-T3 | 123 | 0.70 | **0.3573** | 975.0 | 4.16 | 7.8% | ✅ Done |
| $I1$-T3 | 2026 | 0.70 | **0.3825** | 1210.0 | 5.34 | 18.2% | ✅ Done |
| $I2$-T3 | 42 | 0.70 | **0.3568** | 1166.0 | 5.15 | 20.4% | ✅ Done |
| $I2$-T3 | 123 | 0.70 | **0.3568** | 1164.4 | 5.13 | 27.3% | ✅ Done |
| $I2$-T3 | 2026 | 0.70 | **0.3381** | 962.7 | 4.18 | 13.8% | ✅ Done |
| $I3$-T3 | 42 | 0.70 | **0.3815** | 1301.0 | 5.77 | 32.0% | ✅ Done |
| $I3$-T3 | 123 | 0.70 | **0.3718** | 1065.9 | 4.63 | 11.3% | ✅ Done |
| $I3$-T3 | 2026 | 0.70 | **0.3898** | 1407.8 | 6.30 | 14.7% | ✅ Done |
| $I4$-T3 | 42 | 0.70 | **0.3420** | 884.7 | 3.79 | 0.0% | ✅ Done |
| $I4$-T3 | 123 | 0.70 | **0.3660** | 1044.6 | 4.57 | 1.7% | ✅ Done |
| $I4$-T3 | 2026 | 0.70 | **0.3386** | 844.2 | 3.56 | 0.0% | ✅ Done |

---

## 3. Reference Baseline Anchors

* **Static $k=8$:** $F_1 = 0.3950$ | 1,750.5 context tokens | Mean $k = 8.00$
* **W4 Best Allocator (Balanced LogReg):** $F_1 = 0.3842$ | 1,115.9 context tokens | Mean $k = 4.82$
* **Static $k=6$:** $F_1 = 0.3857$ | 1,351.1 context tokens
* **Cost-Aware $\epsilon=0.01$ Oracle:** $F_1 = 0.4894$ | 839.4 context tokens | Mean $k = 3.49$

