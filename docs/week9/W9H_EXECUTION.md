# W9-H execution and remaining prerequisites

W9-H closes the study; it does not select methods. The new notebook is
`notebooks/week9/week9h_final_evaluation.ipynb`. Its defaults execute only H0.

## Systems and fixed criteria

- Primary: original canonical semantic-k4 SARA first pass, FIXED-5/SEMANTIC@20,
  DraftSemantic coarse K=2, query-only MiniLM exact top-3 sentences presented
  in source order, then unchanged F0 fresh regeneration.
- Safety ablation: identical evidence, frozen F2-A.
- Verifier ablation: identical F0 candidate, all-FIT audited logistic pipeline,
  original 32 features, threshold 0.70. KEEP still pays F0.
- Strong comparators: BOTH semantic k8 and original SARA/G0 from W8.
- Quality hit: positive lower paired 95% F1 CI against both strong comparators.
- Efficiency hit: point F1 loss at most .003 AND cumulative-prompt saving at
  least 15%, against both strong comparators.
- Separate mechanistic hit: at least 90% evidence suppression vs D1 AND its
  complete paired F1 difference CI contained within [-.003,.003]. A CI merely
  crossing zero is not evidence of equivalence.
- Otherwise mixed/negative. No missing-comparator substitution is allowed.

## Current readiness

H0 is expected to finish with BLOCKED readiness. The current repository has
canonical TEST reformatted data but lacks explicit TEST retrieval, canonical
first-pass and SFR paths in its W9 configuration. The W8 full-system runner is
hard-coded for INTERNAL-VAL; it cannot honestly be called a TEST producer.

The historical `test_draft.jsonl` was confirmed synthetic by the user in this
task. It was not opened. The TEST lock conclusion relies on project provenance
and that attestation, not an exhaustive external access log.

The already fitted audited G2 pipeline was exported with zero additional fits
to `results/week9/w9g_action_verification/v1/final_fit`. Its packaging manifest
preserves its original model checksum, FIT membership and audited lineage.

## Run the safe audit

```bash
sara_env/bin/python -m src.week9.final_h.runner --config configs/week9/w9h_final.json --stage h0
```

It writes protocol, historical registry, compatibility matrix, audit report and
read-only pretest manifest under `results/week9/w9h_final/v1`. Repeating it
validates existing immutable artifacts; it never updates the freeze in place.

## Required before confirmation

Prepare outcome-independent TEST adapters and their metadata in a NEW pretest
version. Do this before reading TEST examples/labels or generating metrics.
Do not edit the frozen v1 configuration or bypass H0. No additional scientific
choices are needed: only faithful split-specific materialization.

The materialization spec is a JSON protocol, not a prediction file. Schema:

```json
{
  "schema": "W9-H-materialization-v1",
  "split": "test",
  "test_inputs": {"retrieval": "<real path>", "canonical": "<real path>", "sfr": "<real path>", "raw": "<real path>"},
  "membership_path": "<producer output JSONL with question_id,paper_id>",
  "labels_path": "<producer output JSONL with question_id,paper_id,evaluation>",
  "provenance": "Outcome-independent adapters for exactly the frozen methods",
  "systems": {
    "D0": {
      "command": ["sara_env/bin/python", "<frozen producer.py>", "<arguments>"],
      "producer_source_hashes": {"<frozen producer.py>": "<sha256>"},
      "predictions_path": "<fresh output JSONL>",
      "receipt_path": "<fresh output JSON>",
      "method_config": {"<all frozen method choices>": "<exact values>"}
    }
  }
}
```

Include every system in `protocol.REQUIRED`, plus all legitimate available
secondary baselines. No TEST output may predate the pretest freeze. Producers
must write a receipt containing `complete`, `split`, `system`, `freeze_digest`,
`method_config`, `producer_source_hashes`, and `predictions_sha256`.
The code identity must cover producer imports, model and prompt definitions;
it is not sufficient to hash a thin wrapper while allowing its imports to vary.

Predictions require `system,split,question_id,paper_id,initial_draft,prediction`
and real stage/cost/source metadata. Missing cost observations remain null.
Evaluation objects use the canonical `answers` and `answers_reformatted`.
Producer-reported F1 is ignored and recomputed centrally. W9 rows additionally
carry exact sentence/evidence/chunk identities, generator/decoding and frozen
prompt hashes. `final_h.runtime.generate_w9` reuses E3/F/G implementations;
it does not invent missing first-pass/SFR or W8 baseline inputs.

## Final commands — only after a new complete pretest audit passes

```bash
sara_env/bin/python -m src.week9.final_h.runner --config <complete-pretest-config.json> --stage h1
sara_env/bin/python -m src.week9.final_h.runner --config <complete-pretest-config.json> --stage h2 --run-final-confirmation
sara_env/bin/python -m src.week9.final_h.runner --config <complete-pretest-config.json> --stage h3
```

Notebook equivalent: use that passing version's CONFIG, then enable
`RUN_FINAL_CONFIRMATION=True`. Currently this MUST remain False because H0
is blocked. `--resume` only continues an interrupted identical frozen run;
sealed H2 output cannot be regenerated. A material bug needs preserved invalid
artifacts and an explicitly documented new version.

H3 creates common quality, cost, safety, primary/extended tables, 10,000-resample
paper-cluster intervals, cost/safety Pareto plots, compression accounting,
sealed aggregate metrics, deterministic explanatory examples, final verdict,
18-section report and checksummed manifest in `report/`. No verdict or
confirmation result is created by H0. No legitimate frozen robustness panel
was found; new generator selection and unregistered strata are skipped.
