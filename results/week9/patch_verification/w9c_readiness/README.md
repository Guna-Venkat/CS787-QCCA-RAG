# W9-C readiness audit

**Ready for manual FIT K=2 diagnostic, not primary selector selection.**

B has exactly 6,984 valid rows (1,746 queries x K=0/1/2/4), unchanged canonical K0 predictions and consistent FP16 positive-budget calls. Its primary cost-aware winner is K=0. The primary B artifact is now frozen at that decision; A's frozen SHA is unchanged. `audit.json` records the initial pre-setup blockers.

The user explicitly authorized a separately labelled K=2 selector diagnostic. `prepare_diagnostic.py` prepares separate provenance and outputs, pins the diagnostic to the primary B artifact and configuration, and shares completed A/B inputs through relative links without duplication. Primary C/D-H decisions are not changed. No W9 production Python source was modified in this audit.

`readiness.json` records final checks: one real FIT query, five selector outcomes, FP16 efficient SDPA, peak allocation 14.33 GiB; DraftSemantic reproduces the saved B K2 selection and prediction. Three distinct LM calls served five policies through the existing identity-based cache, including the SANE selector call. The smoke manifest correctly remains incomplete/partial. No full C experiment or primary C freeze was launched. 80 W9 tests passed; one notebook-default test was excluded because the user enabled B's execution flag.

Open notebooks/week9/week9c_selector.ipynb. For a deliberate full run: RUN_FULL=True, RUN_SMOKE=False, FREEZE_FIT=False, SPLIT='fit'. The notebook checks frozen diagnostic prerequisites before loading a model. Results go only to results/week9/diagnostics/w9c_k2/selection/fit/; smoke results are separate. Inspect results before making any downstream decision. The one-query check does not establish correctness for every FIT prompt or held-out superiority.
