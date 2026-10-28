# Week 11 progress

Branch `gagan`. Setup, A, B and C are audited and frozen. **D is running**, reusing C evidence selections and cached Qwen answers. The cohort remains 50 development queries per dataset.

C finished 5,550 available text rows, with 3,506 physical generation calls. All checksums, exact source spans, budgets and same-reader baseline pairs passed. 308 empty strict-prefix evidence rows are flagged; no invalid SciFact labels or generation-cap hits occurred. Optional LLMLingua token conditions remain pending.

Reports: [A](multigranular/v1/retrieval/INTERPRETATION.md), [B](multigranular/v1/passage_representation/INTERPRETATION.md), [C](multigranular/v1/granularity_budget/INTERPRETATION.md).

The passive local monitor records every 15 minutes; the agent reviews and advances phases. Completed results are resumed, not overwritten. Executed notebook copies are under `executed_notebooks/`; user-edited original notebooks are preserved. D/E/F/G/H/I remain to finish.
