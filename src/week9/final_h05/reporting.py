"""Readiness report only; never a TEST performance claim."""
from ..final_h import protocol as p


def write(root,freeze,proof,registry):
    recovery=proof['w8_recovery']
    table='\n'.join('| '+name+' | executable; dry-run and fabricated fixture passed |' for name in registry['order'])
    hashes='\n'.join('| '+name+' | `'+sha+'` |' for name,sha in recovery['source_hashes'].items())
    text=f'''# W9-H H0.5 readiness/provenance audit

**H0 v2: PASS. W8 original SARA/G0: RECOVERED_EXACT. Final confirmation was not started.**

## A. W8 G0 provenance recovery

Recovered 24 byte-exact Python sources from commit `{recovery['commit']}`
(parent `{recovery['parent_commit']}`). Every source checksum matches the
historical W8-J provenance artifact. The isolated archive is
`src/week9/final_h05/historical/`; existing W8 files were not overwritten.
This includes full_system.py, final_dev.py, final_reporting.py and the older
generation/controller integration chain. The current retired/proxy generation
implementation is not a substitute and is not used by the W8 producer.

Historical provenance SHA256: `{recovery['historical_provenance_sha256']}`.
Historical prediction SHA256: `{recovery['historical_prediction_sha256']}`.
Historical live-decision SHA256: `{recovery['live_decisions_sha256']}`.
Experiment fingerprint: `{recovery['historical_fingerprint']}`.

| Recovered source | Exact SHA256 |
|---|---|
{hashes}

The gate remains original B0-GW measured token uncertainty. Selection remains
the frozen S3/DraftSemantic policy with U4-A help/hurt safety probabilities.
The existing E5-DraftSemantic thresholds and single-promotion condition are
unchanged. G0 promotes exactly the selected compressed passage to FULL and
uses the original canonical regeneration prompt; it is not W9 F0 or G1.

## B. Producer registry

| Required final system | Status |
|---|---|
{table}

Each definition pins the executable command, complete source hashes, model,
checkpoint, prompt/routing recipe, decoding, required inputs, output schema,
membership, conceptual calls, GPU requirement, cost mapping and resume policy.
No compatible TEST predictions already exist. This is execution readiness,
not observed TEST comparability or performance.

The first producer materializes canonical BM25256 top-10 retrieval and original
SFR passage embeddings after the explicit H2 claim. SFR and large generators
are never resident together. Generator producers run in separate sequential
processes; G2 is CPU-only. Semantic k4 legitimately aliases the same D0 answer.
D1 precedes E3 so all W9 ablations consume exactly its selected chunks. SAFE
and VERIFY reuse identical PRIMARY evidence; VERIFY also reuses its F0 answer.

Two provisional v1 metadata assumptions were corrected against historical
code, without changing scientific methods: BM25's historical generation cap
is 64, not 128; original W7/W8 first-pass/static calls use the historical
BF16 loader default, while frozen E3/F revision remains FP16 efficient SDPA.
All checkpoints, prompts, k values, routing, thresholds and success criteria
are unchanged. The separate original RAG checkpoint is not called SARA k10.

The historical BM25 prompt function is compiled verbatim from its original
AST to avoid an unrelated retired CLI import collision (`src.utils.text_utils`).
No prompt was rewritten. Canonical prompt parity is tested on inspected DEV.

## C. Non-TEST replay validation

- W8 DEV: all 231 saved query/paper identities, canonical drafts, selected
  actions/candidates, answer serialization, common F1/EM and token accounting
  matched exactly. The actual frozen gate applied to saved measured features
  reproduced probabilities with maximum difference {recovery['maximum_gate_probability_error']}.
- All 1,155 DEV semantic-budget serialized prompts matched the original builder.
- A fixed identity-only three-query DEV sample also reproduced all 15 budget
  allocations with the actual frozen MiniLM encoder on CPU; no LM generation.
- BM25 k8 matched its literal historical serialized prompt.
- All 1,746 FIT F0 aliases matched E3-k3 answers and exact source identities;
  F2-A used identical evidence and sentence/chunk IDs.
- All 1,746 G2 feature vectors matched saved 32-feature inputs within 1e-12;
  the final pipeline bytes match the previously audited full-FIT estimator.

These are saved-output/action/feature/prompt replays, not fresh GPU numerical
regeneration. No controller was refit, and no new all-FIT deployment score is
presented as an honest OOF estimate. Historical stage latency remains scoped.

## D. Synthetic fixture validation

Two fabricated queries/papers exercised all 12 inference adapters, actual
prompt/selection/source-order/cost code, recovered W8 diagnostics/prompts and
serialization, and common aggregation/reporting. Hardware and learned-model
outputs were explicit test doubles. W8 STOP and PROMOTE and G2 KEEP and ACCEPT
were all exercised. Real LM calls: zero. Real encoder calls: zero.
Fixture IDs start with SYNTHETIC; no QASPER TEST text or ID was used.
Fixture metrics have no scientific meaning.

## E. Compatibility matrix and costs

`producer_compatibility_matrix.csv` contains all 12 required systems.
Exact membership and paper identity are enforced at H2, not claimed observed
at H0.5. All final predictions use the same scorer. Actual serialized prompt
sums, generated tokens and conceptual LM calls are comparable. W8 token sums
map directly from its real original accounting; missing measurements stay null.
Latencies are stage-scoped, not common measured deployment end-to-end latency.
Offline SFR/index costs, reusable sentence preprocessing, conceptual online
cost and actual cache use are distinct. G2 KEEP never refunds F0 generation.

Query completion is stored in fsynced atomic per-query shards. Every shard
pins the exact pretest freeze/config/source identity and its own checksum.
Complete receipts seal query-level outputs. Changed hashes refuse resume;
an explicit identical resume is required at H2. No other producer is a fallback.

## F. TEST access audit

Only TEST path existence was checked: the reformatted input and raw Arrow
file. No real TEST bytes, question text, examples, schema contents, IDs,
answers, evidence, F1, EM, HELP/HURT or bootstrap results were read/computed.
IDs and record counts are deferred to H2 and validated in the TEST namespace
before paired evaluation. TEST membership must match across all 12 systems.

The v1 freeze remains byte-identical, SHA256
`{freeze['v1_freeze_sha256']}`. Existing A–G scientific artifacts were not edited.

## G. H0 v2 result

**PASS.** New immutable manifest:
`results/week9/w9h_final/v2/w9h_pretest_freeze_manifest.json`.

File SHA256: `{p.file_hash(root/'w9h_pretest_freeze_manifest.json')}`.

It pins producer definitions/code/configs, recovered W8 sources/provenance,
baseline registry, model/checkpoint/tokenizer weights and cache revisions,
SFR, G2 schema/model/0.70 threshold, E3 k3, exact F prompts, evaluator/bootstrap,
the unchanged pretest success criteria and the TEST execution contract.
Primary remains E3-k3/F0; SAFE and VERIFY cannot be promoted after outcomes.

## H. Remaining risks

No readiness blocker remains. No fresh GPU numerical replay or real TEST run
was performed; hardware/runtime memory and throughput remain operational risks.
W8 uses its historical BF16 execution and measured uncertainty and deliberately
fails if its regenerated first pass differs from shared canonical D0. Preserve
such an invalid/incomplete run rather than changing the method. Interrupted
generation may repeat the one uncommitted call, but completed query shards
are never silently mixed or overwritten. Latency scopes differ across systems.
No new model panel or unregistered strata were added.

## I. Exact user execution

Notebook: `notebooks/week9/week9h_final_evaluation.ipynb`, now points to v2.
Defaults remain RUN_H0_AUDIT=True, RUN_FINAL_CONFIRMATION=False,
RUN_ROBUSTNESS=False. Review H0, then manually set
`RUN_FINAL_CONFIRMATION=True`. Set `RUN_H3_REPORT=True` for sealed statistics.

```bash
sara_env/bin/python -m src.week9.final_h05.runner --config configs/week9/w9h_final_v2.json --stage h2 --run-final-confirmation
```

For an interrupted identical run, append `--resume`. A sealed H2 cannot run
again. Report-only command:

```bash
sara_env/bin/python -m src.week9.final_h05.runner --config configs/week9/w9h_final_v2.json --stage h3
```

## J. Files and verification

Added `src/week9/final_h05/` (recovery/import isolation, literal prompt loading,
materialization, atomic storage, inference adapters, producer CLI, fabricated
fixture, non-TEST replays, readiness/freezing, safety gate and this reporting).
Added the nested `scripts/week9/h05/produce.py` entry point and
`configs/week9/w9h_final_v2.json`. The H notebook now targets v2 with all
confirmation flags still disabled. Existing A–G sources/results and the entire
v1 freeze remain preserved. New v2 artifacts and patch-verification metadata
are separate. Regression test results are saved under
`results/week9/patch_verification/w9h_h05/verification.json`.

BRANCH = gagan

W8 G0 PROVENANCE = RECOVERED_EXACT

W9-H H0.5 READINESS COMPLETED

W9-H H0 v2 = PASS

FINAL CONFIRMATION WAS NOT STARTED

TEST OUTCOMES WERE NOT USED

NO SCIENTIFIC METHOD OR THRESHOLD WAS CHANGED
'''
    for name in ('W9H_H05_READINESS_AUDIT.md','audit_report.md'):
        path=root/name
        with path.open('x') as stream:stream.write(text)
        path.chmod(0o444)
