# Week-2 Scientific Validation Checkpoint Report

**Checkpoint Identifier:** `week2_dev_fixed_k_frozen_v1`  
**Timestamp:** `2026-09-30 16:47:40`  
**Dataset:** QASPER Development Set (231 questions across 86 papers)  
**Evaluated Grid:** k in {0, 2, 4, 5, 6, 8, 10} (1,617 evaluation points)  

---

## 1. Scientific & Data Integrity Verification

- [x] **No Question / Paper Leakage:** 86 development papers are strictly paper-disjoint from training papers. The historical QASPER test set remains **100% untouched**.
- [x] **Pre-Generation Feature Availability:** All allocator features (rho_1, Delta_12, H_norm, N_high, query_token_length) are derived strictly from BM25 scores s_1...s_10 and query text before generation. Zero model predictions or generation targets are accessible.
- [x] **Oracle Labeling Integrity:** The per-question oracle k*_sara(q) is constructed purely from the offline fixed-k matrix with deterministic smaller-k tie-breaking.
- [x] **ROUGE-L Audit:** Verified and recomputed across all 1,617 records using `rouge==1.0.1`.

---

## 2. Oracle Gain Quantification & Category Attribution

- **Static SARA Baseline (k=8):** Mean F1 = **0.3950**
- **Oracle Adaptive SARA (k*):** Mean F1 = **0.5008**
- **Net Oracle Improvement:** **+0.1058 Token F1 (+26.78% relative gain)**

### Category Gain Attribution Table

| Optimal Budget k* | Question Count | Dataset Share | Total F1 Gain Sum | Share of Oracle Gain (%) | Mean Gain per Query |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **k* = 0** (Pure Compression) | 81 | 35.06% | +2.8771 | 11.77% | +0.0355 |
| **k* = 2** | 55 | 23.81% | +8.2437 | **33.73%** | +0.1499 |
| **k* = 4** | 37 | 16.02% | +6.3090 | **25.82%** | +0.1705 |
| **k* = 5** (Parent SARA) | 15 | 6.49% | +3.4256 | **14.02%** | **+0.2284** |
| **k* = 6** | 21 | 9.09% | +3.5813 | **14.66%** | +0.1705 |
| **k* = 8** (Static Baseline) | 22 | 9.52% | +0.0000 | 0.00% | +0.0000 |
| **TOTAL** | **231** | **100.0%** | **+24.4367** | **100.0%** | **+0.1058** |

> [!IMPORTANT]
> **Key Insight**: Over **73.5% of the total Oracle gain** stems from queries requiring intermediate budgets (k* in {2, 4, 5}). Static k=8 hurts performance on these queries by loading redundant text that introduces distractor tokens.

---

## 3. Qualitative Inspection across k* Categories

### Representative Examples for Each k* Budget

| Optimal k* | QID | Question Text | F1(k=0) | F1(k=2) | F1(k=4) | F1(k=5) | F1(k=6) | F1(k=8) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **k*=0** | 1028 | *"Will these findings be robust through different datasets..."* | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **k*=0** | 1315 | *"What are the effects of extracting features of multigranular..."* | **0.33** | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 |
| **k*=2** | 1134 | *"Which training objectives do they combine?"* | 0.00 | **1.00** | 1.00 | 1.00 | 1.00 | 0.67 |
| **k*=2** | 1065 | *"What crowdsourcing platform is used for data collection..."* | 0.00 | **0.50** | 0.44 | 0.44 | 0.44 | 0.44 |
| **k*=4** | 1000 | *"What is English mixed with in the TRAC dataset?"* | 0.00 | 0.40 | **1.00** | 1.00 | 1.00 | 1.00 |
| **k*=5** | 1004 | *"What is the baseline?"* | 0.11 | 0.00 | 0.46 | **0.89** | 0.44 | 0.00 |
| **k*=6** | 1001 | *"Which psycholinguistic and basic linguistic features are used?"* | 0.13 | 0.14 | 0.14 | 0.14 | **0.60** | 0.42 |
| **k*=8** | 1005 | *"What datasets did they use?"* | 0.00 | 0.67 | 0.67 | 0.75 | 0.75 | **1.00** |

---

## 4. Final Performance Hierarchy & Freezing Confirmation

1. **Vanilla RAG Reference Ceiling:** k=10, F1 = **0.4090**
2. **Best Static SARA Baseline (k_dev*):** k=8, F1 = **0.3950**
3. **Parent SARA Configuration:** k=5, F1 = **0.3629**
4. **Pure Compression Baseline:** k=0, F1 = **0.1388**
5. **Oracle Adaptive SARA Limit:** k*(q) in {0,2,4,5,6,8}, F1 = **0.5008**

All Week-2 evaluation matrix records (1,617 rows) and processed datasets are now **frozen**. Week 3 can safely proceed with allocator model development.
