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

Status: COMPLETE AND FROZEN (FIT).
Coverage: 1,746 queries x four budgets = 6,984 validated rows; canonical K=0 and FP16 positive-budget runs.
Frozen primary choice: K_E=0, policy K_E=0. Frozen A remains FIXED-5 / SEMANTIC @20.
FIT mean F1: K0 .385631; K1 .380967; K2 .385004; K4 .380388. Raw evidence does not improve the aggregate fixed-budget baseline.
Artifact: results/week9/frozen/w9b_frozen_config.json.
Decision: Preserve the primary K=0 result. User authorized a separately labelled K=2 selector diagnostic; this does not replace B's cost-aware decision.

## W9-C

Status: COMPLETE AND FROZEN FOR THE K=2 DIAGNOSTIC ONLY.
Coverage: 1,746 FIT queries x five selectors = 8,730 verified rows.
FIT F1: BM25 .382157; QuerySemantic .383490; DraftSemantic .385004; QueryDraft .384095; SANE-adapted .386336.
Cost-aware controlled selector: DraftSemantic. SANE's +.001332 F1 is below .003 and its paper-bootstrap CI [-.009723, .012181] includes zero; no reliable selector breakthrough established.
Validation: DraftSemantic reproduces B K2 selections and predictions on all 1,746 queries.
Artifact: results/week9/diagnostics/w9c_k2/frozen/w9c_frozen_config.json, labelled fit_diagnostic_fixed_k and diagnostic_only.
Primary policy: B K=0 remains frozen and unchanged; no primary C selector was substituted.
Next: controlled D representation diagnostic at K=2 with DraftSemantic, explicitly authorized by user.

## W9-D

Status: READY FOR MANUAL FIT DIAGNOSTIC; bounded GPU check passed, full run not launched.
Design: D0 saved canonical draft; D1 raw chunks; D2 query-only joint extraction; D3 draft-aware joint extraction. Positive arms hold K=2, DraftSemantic, selected chunk IDs and canonical SARA generation fixed.
Extraction: existing SANE blueprint, max 1,024 generated tokens per joint extraction; answer max 128. Compression and factual support must be measured, not assumed.
Configuration: configs/week9/w9d_k2_diagnostic.yaml.
Notebook: notebooks/week9/week9d_representation.ipynb.
Outputs: results/week9/diagnostics/w9c_k2/extraction/fit/; bounded smoke stays in smoke/.
Frozen state: diagnostic C prerequisites frozen; D NOT FROZEN. Primary K=0 remains intact.
Verification: one real query/all four arms passed at 14.37 GiB peak; identical positive-arm chunks and D1 reproduces C. One extraction hit its token cap and D3 had zero exact-span support; notebook reports these model-quality diagnostics. 80 code tests and safe notebook execution passed. Evidence: results/week9/patch_verification/w9d_readiness/. No production Python source change or previous experiment rerun required.
Execution: RUN_FULL=True, RUN_SMOKE=False, FREEZE_FIT=False, SPLIT='fit'. No automatic promotion to primary E-H.

## W9-E

Status: GROUNDED COMPACT V1 IMPLEMENTED; full run NOT STARTED. Preflight quality warnings require review.
Scope: E-only nested package leaves existing A-D code, results, freezes and global source fingerprints untouched. Dedicated E hashes cover the new code.
Controlled inputs: complete D0-D3 are pinned read-only in an E-local input seal. FIXED-5 / SEMANTIC@20, DraftSemantic, K=2, canonical answer generation remain fixed; primary B stays K=0.
Policies: E0 deterministic compact verbatim spans; E1-Q query-only supported structured facts; E1-QD draft-aware equivalent; E2 minimal supported blueprint. Facts use a conservative copied-subspan variant, not unconstrained paraphrases.
Budgets: final whole evidence object 128 tokens, optional separate 64 config; intermediate generation 256 tokens; answer generation 128. No cap increases or raw/synthetic fallbacks.
Fail-closed: unusable evidence returns y0 without an answer call; extraction cost remains charged.
Preflight: FIT queries 2038, 1583, 567; 12 rows, 13 new model calls, 14.21 GiB peak. Accepted spans verified exact and final evidence max 110 tokens. E0 accepted 3/3; E1-Q 1/3; E1-QD/E2 0/3. Eight failures retained y0. Do not equate code readiness with successful structured extraction.
Gate: default full disabled; warnings (including malformed/unsupported output) require deliberate acknowledgment after review. No method/prompt selected using F1; no prompt repair attempted.
Notebook: notebooks/week9/week9e_extraction_faithfulness.ipynb (existing notebook patched).
Config: configs/week9/w9e_grounded.yaml; alternate configs/week9/w9e_grounded_64.yaml requires its own preflight.
Artifacts: results/week9/diagnostics/w9c_k2/grounded_faithfulness/v1/budget_128/.
Verification: 103 passed, 1 existing global safe-default test deselected for user-enabled B/C/D flags; dedicated E safe path passes. Independent source/budget/cache checks passed.
Details/manual instructions: docs/week9/W9E_GROUNDED_IMPLEMENTATION.md. Review quality warnings before investing in a full run; no F/G or TEST work authorized here.

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

## W9-E3 — appended semantic sentence suppression

Status: IMPLEMENTED AND SMOKE-VERIFIED; full surface NOT RUN.
This is a separate E3-SentTopK experiment; no earlier E0/E1-Q/E1-QD/E2 policy/result is changed.
Inputs: identical frozen FIXED-5 / SEMANTIC@20 / DraftSemantic requested K=2 chunks; primary K=0 stays fixed. D/E query/paper/chunk/hash identities are hard-checked read-only.
Method: academic-offset-rules-v1 sentence inventory; canonical MiniLM query-only cosine; top-k selection then source-order presentation. Grid {1,2,3,4,6,8}; no extraction LM, MMR or new model/revision mode.
Config: configs/week9/w9e3_senttopk.yaml. Package: src/week9/senttopk_e3/.
Notebook: appended W9-E3 section in notebooks/week9/week9e_extraction_faithfulness.ipynb. Old and new execution flags default False.
Artifacts: results/week9/w9e3_senttopk/v1/ (separate from every earlier E artifact).
Smoke: FIT queries 594/1204, 151 candidate sentences, 12 answers, zero extraction calls, 18–196 evidence tokens, 13.88 GiB peak. Source, rank and presentation checks passed; old artifact hashes remain unchanged.
Tests: 119 passed, one existing global notebook-default test deselected due to user-enabled B/C/D flags; dedicated E3 safe notebook path passed.
Next: manual full six-k surface. Only complete FIT results select/freeze k, compute paired CIs and intervention/full-action oracles, and recommend dynamic GO/NO-GO. No k frozen or dynamic controller built/run from smoke.
Protocol, exact revision, tests, limitations and manual command: docs/week9/W9E3_SENTTOPK.md.
