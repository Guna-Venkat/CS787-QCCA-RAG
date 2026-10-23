# W9-D readiness

**Ready for the manually executed FIT K=2 representation diagnostic.**

Preparation verifies all 8,730 completed C rows, current source/input/config/row hashes and full query-selector coverage. DraftSemantic reproduces B K2 selections and predictions for all 1,746 queries. C is frozen only inside the diagnostic branch under its original .003 criterion, explicitly labelled diagnostic-only. The primary B K=0 artifact is unchanged. No production Python source, model checkpoint or completed experiment outputs were modified.

Open notebooks/week9/week9d_representation.ipynb. It uses configs/week9/w9d_k2_diagnostic.yaml. For full execution: RUN_FULL=True, RUN_SMOKE=False, FREEZE_FIT=False, SPLIT='fit'. D outputs belong in results/week9/diagnostics/w9c_k2/extraction/fit/. Smoke outputs are separate; no full experiment was launched or D decision frozen. The main configs/week9/w9d.yaml is preserved.

The real trained-SARA smoke ran one FIT query, all four arms and both full-budget extraction calls. Peak allocated memory was 14.37 GiB. D0 reproduces the draft; D1 reproduces C; D1/D2/D3 have identical selected chunks. The notebook passed all code cells with execution flags disabled. 80 W9 tests passed; one notebook-default check was excluded because the user enabled B/C full-run flags.

Readiness is about execution, not a claim of extraction quality. In this one query, D2 generated 1,024 tokens, hit its cap and ended with a partial table row; D3 generated 208 tokens with zero exact quote support. Their raw tables, audit details and downstream answers remain saved. D intentionally measures extraction as produced under the existing SANE blueprint; faithfulness-format interventions belong to later E. The notebook now reports cap-hit rates, zero-support tables, invalid/empty extractions, compression, costs and paired answer comparisons. The same max extraction budget applies to both variants. Do not infer draft-aware superiority from a hypothesis or this single-query smoke.

Reproducibility: prepare.py verifies C and creates diagnostic configuration/freeze; verify_smoke.py validates actual smoke rows and controls. preparation.json, readiness.json, notebook_safe_check.json and gpu_smoke.log provide evidence. No full D run, W8 rerun, TEST access or primary selector/representation promotion occurred.
