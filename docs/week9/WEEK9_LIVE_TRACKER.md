# Week 9 live tracker

BRANCH = gagan. Implementation only; no experimental claim or frozen winner.

## W9-A

Status: IMPLEMENTED; manual execution pending.
Hypothesis: Semantic boundaries can improve evidence acquisition.
Prerequisites: see WEEK9_RUNBOOK.md and enforced phase dependencies.
Implementation status: source, CLI, YAML, notebook and invariant checks present.
Full-run status: NOT RUN
Frozen/not frozen: NOT FROZEN
Key artifact paths: results/week9/retrieval/fit/; results/week9/frozen/w9a_frozen_config.json
Key results: none; full experiment not executed.
Audit notes: Final real two-query smoke and resume passed: 342 retrieval rows, 38 chunk manifests, 707 unique embeddings. Final source/row checksums verified under results/week9/implementation_verification/. No configuration frozen.
Decision: Await complete FIT evidence and manual review.
Next action: Run the first A command in the runbook.

## W9-B

Status: IMPLEMENTED; manual execution pending.
Hypothesis: Small evidence budgets may preserve factual specificity.
Prerequisites: see WEEK9_RUNBOOK.md and enforced phase dependencies.
Implementation status: source, CLI, YAML, notebook and invariant checks present.
Full-run status: NOT RUN
Frozen/not frozen: NOT FROZEN
Key artifact paths: results/week9/ke_surface/fit/; results/week9/frozen/w9b_frozen_config.json
Key results: none; full experiment not executed.
Audit notes: No generator experiment run. Gold restricted to evaluation/training targets.
Decision: Await complete FIT evidence and manual review.
Next action: Finish and freeze prerequisite phases, then run this phase manually.

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
Hypothesis: Explicit revision may improve utilization.
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
Hypothesis: Candidate-specific stopping is worthwhile only with measured headroom.
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
Hypothesis: FIT OOF selects the quality/cumulative-cost frontier.
Prerequisites: see WEEK9_RUNBOOK.md and enforced phase dependencies.
Implementation status: source, CLI, YAML, notebook and invariant checks present.
Full-run status: NOT RUN
Frozen/not frozen: NOT FROZEN
Key artifact paths: results/week9/frontier/fit/; results/week9/frozen/w9h_frozen_config.json
Key results: none; full experiment not executed.
Audit notes: No generator experiment run. Gold restricted to evaluation/training targets.
Decision: QCLA-1 FIT-OOF artifact missing; final frontier refuses substitution.
Next action: Finish and freeze prerequisite phases, then run this phase manually.
