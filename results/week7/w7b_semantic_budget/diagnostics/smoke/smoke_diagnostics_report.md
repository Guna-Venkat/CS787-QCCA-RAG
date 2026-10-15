# Week 7-B Smoke Test Diagnostics Report

**STATUS:** SMOKE_ONLY_NOT_FOR_REPORTING  
**DATE:** October 2026  
**DESCRIPTION:** Preflight verification of Week 7-B pipeline modules on 3 TRAIN and 2 DEV queries.

---

## 1. Split Isolation & Safety Audit
- **TEST Access Guard:** Verified `assert_test_disabled('test')` strictly blocks execution.
- **TRAIN Queries Sampled:** ['512', '513', '745']
- **DEV Queries Sampled:** ['1912', '2201']
- **Data Leakage:** 0 overlap across sampled splits.

---

## 2. Semantic Routing Invariants
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (frozen bi-encoder).
- **Tested Budgets:** k in [2, 4, 5, 6, 8].
- **Invariant Audit:**
  - Exactly k FULL passages selected: PASSED.
  - Exactly 10 - k COMPRESSED passages: PASSED.
  - All 10 candidate passages represented once: PASSED.
  - Original BM25 order preserved in final presentation: PASSED.

---

## 3. Response Surface Cache & Matrix
- **Cache File:** `results/week7/w7b_semantic_budget/diagnostics/smoke/smoke_semantic_fixed_k_results.jsonl` (25 verified entries).
- **Matrix File:** `results/week7/w7b_semantic_budget/diagnostics/smoke/smoke_semantic_fixed_k_matrix.csv` (5 queries x 5 budgets).
- **Oracle File:** `results/week7/w7b_semantic_budget/diagnostics/smoke/smoke_semantic_oracle.csv`.
- **Mean Oracle k:** 8.00

---

## 4. Feature Extraction & Alignment
- **Features Extracted:** ['bm25_mean_score', 'lex_coverage_top2', 'sem_sim_mean_top6', 'evidence_query_cluster_span', 'evidence_lexical_overlap_mean']
- **Completeness:** 100% of 5 core features extracted without missing values.

---

`W7B_SMOKE_STATUS = SUCCESS_SMOKE_ONLY_NOT_FOR_REPORTING`
