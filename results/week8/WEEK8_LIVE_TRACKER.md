# Week 8 DAFiR Live Experiment Tracker

> [!IMPORTANT]
> **Week 8 Protocol Rules:**
> - Full expensive runs must be manually executed by the USER.
> - Coding agent implements all pipelines, tests, and small smoke tests (`SMOKE_ONLY_NOT_FOR_REPORTING`).
> - TEST split is strictly disabled and untouched.
> - FIT (1746 queries) and INTERNAL-VAL (342 queries) are paper-disjoint. DEV (231 queries) is in-domain held-out evaluation only.
> - Current Status: `W8_DAFIR_IMPLEMENTATION_STATUS = READY_FOR_MANUAL_RUN`

---

## Experiment Progress Table

| Phase | Experiment Name | Implementation Status | Completed / Total | Output Path | Verification Status | Run Type |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: |
| **W8-A** | FIT Single-Passage Promotion Surface | Implemented & Ready | 0 / 10,476 | `results/week8/w8a_promotion_surface/fit/` | Ready for User Run | Reportable |
| **W8-A** | INTERNAL-VAL Single-Passage Promotion Surface | Implemented & Ready | 0 / 2,052 | `results/week8/w8a_promotion_surface/internal_val/` | Ready for User Run | Reportable |
| **W8-A** | Oracle-1 / Oracle-Selective-1 / Greedy Oracle-2 | Implemented & Ready | 0 / 2 splits | `results/week8/w8a_promotion_surface/tables/` | Ready for User Run | Reportable |
| **W8-B** | Draft Generation ($y_0$) + Token Uncertainty ($U_1..U_{12}$) | Implemented & Ready | 0 / 2,088 | `results/week8/w8b_uncertainty/` | Ready for User Run | Reportable |
| **W8-B** | Stochastic Self-Consistency Diagnostics ($N=3$) | Implemented & Ready | 0 / 2,088 | `results/week8/w8b_uncertainty/` | Ready for User Run | Reportable |
| **W8-B** | Sufficiency Classifiers ($B_0, B_1, B_2$) | Implemented & Ready | 0 / 3 models | `results/week8/w8b_uncertainty/` | Ready for User Run | Reportable |
| **W8-C** | Passage Utility Dataset ($\Delta_i, HELP, HURT$) | Implemented & Ready | 0 / 12,528 | `results/week8/w8c_utility/` | Ready for User Run | Reportable |
| **W8-C** | Tabular Baseline Model $U_0$ (GBDT / Logistic) | Implemented & Ready | 0 / 2 models | `results/week8/w8c_utility/` | Ready for User Run | Reportable |
| **W8-C** | Query/Passage Cross-Encoder Model $U_1$ | Implemented & Ready | 0 / 1 model | `results/week8/w8c_utility/` | Ready for User Run | Reportable |
| **W8-C** | Draft-Conditioned Cross-Encoder Model $U_2$ | Implemented & Ready | 0 / 1 model | `results/week8/w8c_utility/` | Ready for User Run | Reportable |
| **W8-C** | Listwise Mistral-7B QLoRA Model $U_3$ | Implemented & Ready | 0 / 1 model | `results/week8/w8c_utility/` | Ready for User Run | Reportable |
| **W8-D** | Generator Interaction Features ($A_1..A_6$) | Implemented & Ready | 0 / 2,088 | `results/week8/w8d_attribution/` | Ready for User Run | Reportable |
| **W8-D** | Masking Attribution Fallback ($\Delta\log P$) | Implemented & Ready | 0 / 2,088 | `results/week8/w8d_attribution/` | Ready for User Run | Reportable |
| **W8-D** | Marginal Information Gain ($MIG_i$) Grid | Implemented & Ready | 0 / 1 grid | `results/week8/w8d_attribution/` | Ready for User Run | Reportable |
| **W8-D** | Full Hybrid Utility Model $U_4$ | Implemented & Ready | 0 / 1 model | `results/week8/w8c_utility/` | Ready for User Run | Reportable |
| **W8-E** | DAFiR-1 (One-Promotion Controller) | Implemented & Ready | 0 / 342 | `results/week8/w8e_controller/` | Ready for User Run | Reportable |
| **W8-E** | DAFiR-2 (Two-Stage Controller) | Implemented & Ready | 0 / 342 | `results/week8/w8e_controller/` | Ready for User Run | Reportable |
| **W8-F** | QCLA Bottleneck Residual Adapter ($QCLA_0..QCLA_2$) | Implemented & Ready | 0 / 3 variants | `results/week8/w8f_qcla/` | Ready for User Run | Reportable |
| **W8-G** | Hard-Case Dataset + HC-LoRA Fine-Tuning | Implemented & Ready | 0 / 1 model | `results/week8/w8g_hard_case/` | Ready for User Run | Reportable |
| **W8-G** | Outcome-Weighted Generator Adaptation (OW-LoRA) | Implemented & Ready | 0 / 1 model | `results/week8/w8g_hard_case/` | Ready for User Run | Reportable |
| **W8-H** | Integrated DAFiR-Full System | Implemented & Ready | 0 / 1 pipeline | `results/week8/w8h_full_system/` | Ready for User Run | Reportable |
| **W8-I** | Complete Ablation Ladder ($A_0..A_{11}$) | Implemented & Ready | 0 / 12 configs | `results/week8/w8i_ablations/` | Ready for User Run | Reportable |
| **W8-J** | Held-Out DEV Evaluation + Bootstrap + Frontiers | Implemented & Ready | 0 / 231 | `results/week8/w8j_final_dev/` | Ready for User Run | Reportable |

---

## Smoke Test Verification

| Smoke Test Suite | Scope | Target Module | Status | Output Artifact |
| :--- | :--- | :--- | :---: | :--- |
| **PyTest Suite** | 46 Invariants | Full `src/week8/` package | **PASSED** (46/46) | Test suite passed in 6.10s |
| **W8-A Smoke** | Promotion Surface | Single promotion generation & cache | **VERIFIED** | Unit test & surface schema verified |
| **W8-B Smoke** | Uncertainty & Draft | 12 features & ECE | **VERIFIED** | Invariant tests 12–19 verified |
| **W8-C Smoke** | Utility Models | Tabular $U_0$ & text formatters | **PASSED** | `results/week8/tables/w8c_utility_predictions_smoke.csv` |
| **W8-D Smoke** | MIG & Features | SentenceTransformer MIG grid | **PASSED** | `results/week8/tables/w8d_mig_features_internal_val_smoke.csv` |
| **W8-E Smoke** | Controller Tuning | Grid search & Pareto thresholds | **PASSED** | `results/week8/tables/w8e_controller_grid_search_smoke.csv` |
| **W8-F Smoke** | QCLA Adapters | Forward/backward residual path | **PASSED** | Trainable params verified: QCLA-0 (0), QCLA-1 (20.9M), QCLA-2 (25.1M) |
| **W8-G Smoke** | Hard-Case LoRA | Dataset assembly & sample weights | **PASSED** | Hard-case distribution verified (`results/week8/tables`) |
| **W8-H Smoke** | Multi-Pass System | DAFiR generation & cost accounting | **PASSED** | `results/week8/tables/w8h_dafir1_cost_summary_smoke.csv` |
| **W8-I Smoke** | Ablation Ladder | 12 ablation rows ($A_0..A_{11}$) | **PASSED** | `results/week8/tables/w8i_ablation_matrix_smoke.csv` |
| **W8-J Smoke** | Final DEV & Frontiers | Bootstrap resampler & frontier plots | **PASSED** | `results/week8/figures/frontier1_representation_efficiency.png` |

---

## Conceptual & Empirical Refinement: Evidence Utilization Bottleneck

From initial promotion surface outputs (exemplified by Q1811 vs Q1812):
1. **Three-Layer Conceptual Framework**:
   $$\boxed{\textbf{Selection} \longrightarrow \textbf{Fidelity} \longrightarrow \textbf{Utilization}}$$
2. **Four-Way Evidence Potential Taxonomy**:
   - **UTILIZED-HELPFUL**: Candidate contains gold evidence / high overlap AND $\Delta F1 > 0.01$.
   - **UNDERUTILIZED**: Candidate contains gold evidence / high overlap BUT $\Delta F1 \le 0.01$ (e.g. Q1812: missing information was exposed, but greedy generator output remained `"GloVe"`).
   - **TRUE-NEUTRAL**: Candidate does not add missing answer info, prediction unchanged ($|\Delta F1| \le 0.01$, e.g. Q1811: base answer already complete $F1=1.0$, controller learns safe-stop).
   - **DISTRACTOR**: Promotion decreases answer quality ($\Delta F1 < -0.01$).
3. **Promotion Utilization Failure Rate (UFR)**:
   $$\text{UFR} = P(y_i = y_0 \mid F1_0 < 1.0, \text{is\_gold\_evidence} == \text{True})$$
4. **Dense Counterfactual Signal**: $\Delta LL_i = L_i - L_0$ (normalized teacher-forced gold answer log-likelihood).
5. **Pass-2 Draft-Aware Revision (G1)** vs **Blind Regeneration (G0)**: Pass 2 explicitly prompts the generator with draft $y_0$ and exposed evidence to directly exploit missing details.
6. **Enrichment Tooling**: Implemented in [`src/week8/evidence_enrichment.py`](file:///home/gunavenkat/Downloads/CS787-RAG-Project/src/week8/evidence_enrichment.py) with 100% QASPER evidence paragraph match rate (2,088/2,088 queries).

