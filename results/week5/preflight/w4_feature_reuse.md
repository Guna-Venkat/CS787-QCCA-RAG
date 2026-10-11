# Week 5 Preflight: Week 4 Feature Reuse & Pre-Generation Audit

**Date:** October 2026  
**Auditor:** SARA Research Engineering Agent  
**Artifact:** `results/week5/preflight/w4_feature_reuse.md`  

---

## 1. Overview & Pre-Generation Guarantee

Week 5 studies whether directly reading retrieved passage natural-language text provides evidence allocation signals beyond cheap pre-generation features. To ensure fair information ablations:
- **Condition A:** Query Only ($q$)
- **Condition B:** Query + W4 Retrieval/Features ($q + \mathbf{x}_{\text{W4}}$)
- **Condition C:** Query + Actual Retrieved Passages ($q + D_1 \dots D_{10}$)
- **Condition D:** Query + Actual Passages + W4 Features ($q + D_1 \dots D_{10} + \mathbf{x}_{\text{W4}}$)

**Invariant:** All W4 features must satisfy the strict **pre-generation invariant**. No feature may use:
- Generated candidate or final answers
- Gold answers or annotations
- Downstream task metric values ($F_1$, EM, ROUGE)
- Oracle allocation budgets
- Attention weights into future tokens

Every feature documented below was verified against `results/week4/feature_discovery/feature_metadata.csv` as `available_pre_generation: True` and `leakage_status: VERIFIED_PRE_GEN`.

---

## 2. The 6 Feature Families (64 Features Total)

| Family | Count | Description | Primary Inputs |
| :--- | :---: | :--- | :--- |
| **A: BM25/Retrieval** | 17 | Retrieval score magnitude, drop-off, entropy, concentration ratios (`rho_1`, `delta_12`, `bm25_mean_score`, etc.) | BM25 retrieval scores |
| **B: Query Complexity** | 12 | Query syntax, word/char counts, clause conjunctions, question category flags (`query_conjunction_count`, `query_is_what_which`, `query_is_numerical`, etc.) | Raw query text $q$ |
| **C: Semantic Structure** | 11 | Query-passage cosine similarities via SFR-Embedding-Mistral (`sem_sim_top1`, `sem_sim_mean_top6`, saturation ratios) | Query & SFR passage embeddings |
| **D: Passage Redundancy/Diversity** | 10 | Pairwise passage embedding distances, clustering, nearest-neighbor separation (`passage_sim_std`, `passage_cluster_count`) | SFR passage embeddings |
| **E: Lexical Coverage** | 8 | Exact term overlap and cumulative vocabulary coverage across top-$k$ documents (`lex_coverage_top1`, `lex_coverage_top2`, etc.) | Query text & passage text |
| **F: Evidence Structure** | 6 | Evidence length variance, Jaccard token overlap, multi-aspect cluster spans (`evidence_query_cluster_span`, `evidence_lexical_overlap_mean`) | Passage text & cluster assignments |

---

## 3. Strongest Baseline Feature Set: The Locked 5-Feature Core

In Week 4, comprehensive feature family ablations, FDR pruning, and GroupKFold stability analysis identified the optimal, non-redundant 5-feature subset for Stage-2 ($4 \rightarrow 6$) transition allocation:

1. `bm25_mean_score` (Family A) — Average retrieval strength across the candidate set.
2. `lex_coverage_top2` (Family E) — Fraction of query content words covered by top-2 passages (saturation signal).
3. `sem_sim_mean_top6` (Family C) — Mean semantic similarity of deep candidate passages.
4. `evidence_query_cluster_span` (Family F) — Dispersion of query aspects across distinct passage clusters.
5. `evidence_lexical_overlap_mean` (Family F) — Pairwise Jaccard overlap indicating chunk redundancy.

A minimal 2-feature subset was also isolated:
- `bm25_mean_score` + `lex_coverage_top2`

---

## 4. Re-use Plan for Week 5 Information Ablations

In Condition B and Condition D, the model will ingest:
1. **Primary Comparison Vector:** The locked 5-feature core (`STAGE_2_CORE_FEATURES`), which yielded the state-of-the-art Week 4 Balanced Logistic Regression policy ($F_1 = 0.3842$, context reduction $= 36.54\%$).
2. **Extended Feature Vector (Ablation Control):** The 15 FDR-validated provisional features from `results/week4/feature_validation/provisional_feature_set.csv`.
3. **Full Vector (Exploratory):** The complete 64-feature discovery matrix.

All feature values will be standardized using train-split statistics only, preventing cross-split data leakage.
