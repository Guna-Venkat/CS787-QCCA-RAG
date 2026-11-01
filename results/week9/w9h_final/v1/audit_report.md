# W9-H H0 audit

**BLOCKED — audit completed; confirmation is not ready**

Primary: E3-k3/F0. Safety: E3-k3/F2-A. Verifier: E3-k3/F0 then frozen 32-feature logistic at 0.70.

## Frozen baseline registry

104 provenance entries across earlier inspected splits and frozen methods. Historical scores are never TEST scores.
Original Standard RAG is checkpoint-distinct from SARA. W8 original SARA/G0 remains mandatory.

## Split audit

{'fit': 'Extensively developed', 'dev': 'Inspected; not untouched confirmation', 'internal_val': 'Inspected; not untouched confirmation', 'test': 'Locked according to config/code/history and human provenance; no claim of exhaustive external access logging', 'suspicious_filename': 'User confirmed synthetic fixture in this task; file not opened', 'test_data_opened_by_h0': False, 'test_labels_opened_by_agent': False, 'evidence': ['configs/week8/w8_dafir.yaml: test_disabled=true', 'src/week9/config.py: explicit TEST authorization and frozen config guard', 'Existing final split paths absent from W9 data configuration', 'User explicitly confirmed historical test_draft is synthetic; not inspected'], 'known_test_query_count': None}

## Blocking dependencies

- TEST materialization spec absent: retrieval/first-pass/SFR inputs and same-split baseline producers are not configured
- Historical W8 runtime source absent from current tree: full_system.py, final_dev.py, final_reporting.py; preserve/recover the exact audited implementation for the TEST adapter

## Success criteria

```json
{
  "primary": "W9-PRIMARY",
  "strong_comparators": [
    "original_SARA_G0",
    "semantic_k8"
  ],
  "quality": "Positive lower 95% paired paper-cluster CI against BOTH W8 original_SARA/G0 and semantic k8; no major protocol issue.",
  "efficiency": "Point F1 loss <= .003 against BOTH strong comparators AND mean cumulative query prompt reduction >= .15 against BOTH.",
  "mechanistic": "Against D1: exposed evidence reduction >= .90 AND the entire paired 95% F1 difference CI lies within [-.003,.003].",
  "quality_tolerance": 0.003,
  "minimum_cumulative_saving": 0.15,
  "minimum_evidence_suppression": 0.9,
  "compatibility_is_not_equivalence": true,
  "order": [
    "QUALITY_HIT",
    "EFFICIENCY_OR_MECHANISTIC_HIT",
    "MIXED_OR_NEGATIVE"
  ],
  "missing_required_comparator": "No final classification; evaluation is incomplete, never drop a comparator",
  "note": "Both strong comparators designated before confirmation. Conservative interval containment operationalizes quality compatibility; CI crossing zero alone is insufficient."
}
```

## Freeze

`/home/gunavenkat/Downloads/CS787-RAG-Project/results/week9/w9h_final/v1/w9h_pretest_freeze_manifest.json`
File SHA256: `0df18e811072a72a1fca5fe862a5d90fe1d45a2617c3f34af3142f23eaab4c08`

## Execution limits

No TEST files, examples or labels were opened. No confirmation generation, new training or threshold selection ran.
H1 prepares a metadata plan. H2 executes only pre-frozen producer commands and ingests their complete query-level exports.
Readiness gaps are material blockers for a fair confirmation, not newly demonstrated scoring bugs in preserved historical results.
TEST-specific adapters and first-pass/SFR materialization are not supplied by legacy FIT/DEV-only runners. Until specified, H2 fails closed.
Do not simply turn the final flag on with the current BLOCKED audit. Do not edit v1 after freezing; complete missing metadata/adapters in a new version before TEST.

## Claims hierarchy

Confirmed: pending; no confirmation verdict exists.
Development: prior FIT/DEV diagnostics remain development findings, including E3 compression and G2 nested OOF.
Negative/inconclusive: generative extraction brittleness and inconclusive verifier gains are retained.
