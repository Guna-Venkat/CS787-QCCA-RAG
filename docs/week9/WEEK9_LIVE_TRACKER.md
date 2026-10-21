# Week 9 live tracker

BRANCH = gagan. Implementation only; no experimental claim or frozen winner.

## W9-PRE

Status: IMPLEMENTED; official caches and bounded GPU smoke only (see preflight manifests).
Full FIT screen: NOT RUN. Model panel: NOT FROZEN.
Roles: canonical trained SARA first pass; alternative extractors/revisers are textual only.
Candidates: Mistral, Llama3.1, Qwen2.5, DeepSeek-R1-distill; secondary Qwen3/Gemma3.
Measured: M0 FP16 numeric/memory PASS, but 4/4 extraction outputs malformed/truncated; M2 int8 completes all 10 task calls; M3 NF4/FP32 diagnostic has valid scores but 5/5 empty final answers at the declared caps. M1/M5 access blocked; M4 disk deferred.
Next: manually screen M0/M2 on the fixed 49-query, 14-paper FIT subset; inspect extraction failures and qualitative evidence before any panel freeze. No model is selected by loading or smoke success. See W9PRE_MODEL_AUDIT.md.

## W9-A

Status: COMPLETE FIT, AUDITED AND FROZEN (2026-10-21).
Saved results: 298,566 rows = 1,746 queries × 171 configurations; 656 papers.
Frozen primary: FIXED-5 / SEMANTIC @20, using the existing global .003 recall-tolerance/token-cost rule.
All-gold recall: 0.878716 on 1,682 evidence-annotated queries. Mean retrieved tokens: 5,600.47.
Artifacts: results/week9/frozen/w9a_frozen_config.json; results/week9/retrieval/fit/freeze_audit/.
Audit: SEM-5/7/8 saturate at W20 for 97–98% of queries. SEM-4/RRF@20 is a Pareto alternative, not the preregistered primary winner. Nominal median matching does not equal full length-distribution matching.
Decision: No retrospective primary change. Runtime source hashes and cached outcomes remain unchanged.
Next: W9-B manual FIT run. Confirmation and TEST remain unrun by this audit.

## W9-B

Status: READY FOR MANUAL FIT RUN; not executed.
Frozen retrieval: W9-A FIXED-5 / SEMANTIC @20. Selector: DraftSemantic.
Grid: K_E = 0,1,2,4. K=0 reuses the canonical draft; positive budgets use raw chunks at this stage.
Analysis: exact argmax (smallest tied K) and epsilon=.01 smallest near-best K, without predictor training.
Interpretation: median unit is 1,022 tokens; 256/1,746 FIT queries have fewer than four candidates. Notebook reports requested/actual counts and a full-four-candidate subgroup.
Verification: frozen A and B dependency/protocol checks pass; B notebook runs safely with full/smoke flags disabled.
Next: manually execute notebooks/week9/week9b_evidence_k_surface.ipynb or the B CLI from the runbook; inspect complete results before freezing B.

## W9-C

Status: IMPLEMENTED; manual execution pending.
Hypothesis: DraftSemantic is the simple baseline to beat.
Prerequisites: see WEEK9_RUNBOOK.md and enforced phase dependencies.
Implementation status: source, CLI, YAML, notebook and invariant checks present.
Full-run status: NOT RUN
Frozen/not frozen: NOT FROZEN
Key artifact paths: results/week9/selection/fit/; results/week9/frozen/w9c_frozen_config.json
Key results: none; full experiment not executed.
Audit notes: No generator experiment run. Gold restricted to evaluation/training targets.
Decision: Await complete FIT evidence and manual review.
Next action: Finish and freeze prerequisite phases, then run this phase manually.

## W9-D

Status: IMPLEMENTED; manual execution pending.
Hypothesis: Draft conditioning may improve compact evidence extraction.
Prerequisites: see WEEK9_RUNBOOK.md and enforced phase dependencies.
Implementation status: source, CLI, YAML, notebook and invariant checks present.
Full-run status: NOT RUN
Frozen/not frozen: NOT FROZEN
Key artifact paths: results/week9/extraction/fit/; results/week9/frozen/w9d_frozen_config.json
Key results: none; full experiment not executed.
Audit notes: No generator experiment run. Gold restricted to evaluation/training targets.
Decision: Await complete FIT evidence and manual review.
Next action: Finish and freeze prerequisite phases, then run this phase manually.

## W9-E

Status: IMPLEMENTED; manual execution pending.
Hypothesis: Structured evidence may improve support and compression.
Prerequisites: see WEEK9_RUNBOOK.md and enforced phase dependencies.
Implementation status: source, CLI, YAML, notebook and invariant checks present.
Full-run status: NOT RUN
Frozen/not frozen: NOT FROZEN
Key artifact paths: results/week9/faithfulness/fit/; results/week9/frozen/w9e_frozen_config.json
Key results: none; full experiment not executed.
Audit notes: No generator experiment run. Gold restricted to evaluation/training targets.
Decision: Await complete FIT evidence and manual review.
Next action: Finish and freeze prerequisite phases, then run this phase manually.

## W9-F

Status: IMPLEMENTED; manual execution pending.
Hypothesis: Compare unchanged F0/F1 with evidence-anchored F2 on identical evidence; FIT freezes the winning mode. The single predeclared F2 prompt specification is already saved with its hash; no winning revision mode has been selected.
Prerequisites: see WEEK9_RUNBOOK.md and enforced phase dependencies.
Implementation status: source, CLI, YAML, notebook and invariant checks present.
Full-run status: NOT RUN
Frozen/not frozen: NOT FROZEN
Key artifact paths: results/week9/revision/fit/; results/week9/frozen/w9f_frozen_config.json
Key results: none; full experiment not executed.
Audit notes: No generator experiment run. Gold restricted to evaluation/training targets.
Decision: Await complete FIT evidence and manual review.
Next action: Finish and freeze prerequisite phases, then run this phase manually.

## W9-G

Status: IMPLEMENTED; manual execution pending.
Hypothesis: Candidate-specific stopping is worthwhile only with measured headroom under the actual FIT-frozen F0/F1/F2 mode.
Prerequisites: see WEEK9_RUNBOOK.md and enforced phase dependencies.
Implementation status: source, CLI, YAML, notebook and invariant checks present.
Full-run status: NOT RUN
Frozen/not frozen: NOT FROZEN
Key artifact paths: results/week9/controller/fit/; results/week9/frozen/w9g_frozen_config.json
Key results: none; full experiment not executed.
Audit notes: No generator experiment run. Gold restricted to evaluation/training targets.
Decision: Gate thresholds preregistered; may SKIP. Global gate requires verified outer FIT OOF.
Next action: Finish and freeze prerequisite phases, then run this phase manually.

## W9-H

Status: IMPLEMENTED; manual execution pending.
Hypothesis: FIT OOF selects the canonical-generator method frontier; expanded same-split baselines and generator robustness are reported in separate views.
Prerequisites: see WEEK9_RUNBOOK.md and enforced phase dependencies.
Implementation status: source, CLI, YAML, notebook and invariant checks present.
Full-run status: NOT RUN
Frozen/not frozen: NOT FROZEN
Key artifact paths: results/week9/frontier/fit/; results/week9/frozen/w9h_frozen_config.json
Key results: none; full experiment not executed.
Audit notes: No generator experiment run. Gold restricted to evaluation/training targets.
Decision: QCLA without FIT-OOF is reporting-only. Final W8 G0/HC predictions are registered through checksummed read-only imports from committed main history; direct HC k4 remains INTERNAL-VAL only. No substitution or automatic winner from another generator.
Next action: Finish and freeze prerequisite phases, then run this phase manually.
