# W9-H H0.5 readiness/provenance audit

**H0 v2: PASS. W8 original SARA/G0: RECOVERED_EXACT. Final confirmation was not started.**

## A. W8 G0 provenance recovery

Recovered 24 byte-exact Python sources from commit `fe6d1bdf08f8e9960f8a22432a98ce6fc923ea8d`
(parent `6b39cd26c56085f96c7a9711ed4a3a84d634bd33`). Every source checksum matches the
historical W8-J provenance artifact. The isolated archive is
`src/week9/final_h05/historical/`; existing W8 files were not overwritten.
This includes full_system.py, final_dev.py, final_reporting.py and the older
generation/controller integration chain. The current retired/proxy generation
implementation is not a substitute and is not used by the W8 producer.

Historical provenance SHA256: `69e781ce950ddf0d8788d7085632ee7c1d0ee34e153fe9493ba35c420376b3eb`.
Historical prediction SHA256: `d2e7a000d072a975214d6534b7985a822d7c891927041ff7bb03118e53568fde`.
Historical live-decision SHA256: `2718818a351c3d91deed14af76682696316fd5dfb4406fb71fa283c1f5e7f57c`.
Experiment fingerprint: `3d6e2bf7fc33526316abd5f1ad118c0e03a302538f3f43e9b4a30dfac1707241`.

| Recovered source | Exact SHA256 |
|---|---|
| src/week8/hard_case_dataset.py | `099c3a306f56d696bd36f4f01f3ae5bf249c06a40c5d69fd5602115e4ce4ae03` |
| src/week8/hard_case_lora.py | `7775f5295d765773fc168af21f60ca5b31286e37528d25e7b03bacedde64f6a2` |
| src/week8/outcome_weighted_training.py | `c8d3aeaa6f4ece3a4fbad649e5cc4905352d9c3e713dad5a8ceb534dec85760b` |
| scripts/week8/run_w8g_lora_training.py | `cd67e395806475e8ee7acce3eab7d438b82454be91a6b05655a475c2455fd827` |
| src/week8/qcla_training.py | `33cf31e92c35069c98305ee51d7ee954e6a215e02dc40501f09316adc1093ee3` |
| src/week8/qcla_experiment.py | `a44788efd3351732336407deb2e73066ec837311d9a11c6f44b90cdf2aa1cc03` |
| src/week8/generator_features.py | `8391e82ad0dc432d641c5fc57191eececabe0e19251dd08e615df12d9c5bee84` |
| src/week8/utility_training.py | `2c66e967b7ad89610940b8e8c96a3123fa940a73d68238d5ba3b251dd1180b88` |
| src/week7/w7e/prompt_builder.py | `c0477b051721663efacf7a9c7e3f5c9ed694f4b9d957d288160c7ec3c58c2283` |
| src/week6/evaluation.py | `45e9636fa25d0595b1c331003ea56f625a86511ec21d36207d388c6f8a8fab16` |
| Baselines/SARA-main/src/model/loader.py | `eaee0e790ab8307ed2c0e639bcfe3b60519f2d2527c3c5e11c8f088e9087be24` |
| Baselines/SARA-main/src/model/xMistral/modeling_xmistral.py | `85aabecd09eb2aef670c17b47f218a99f3da006a90074441fe63ebaf16f15604` |
| src/week8/generation.py | `975bd50dd8ca6439aafb9704c78d9969303500c3b8293a87237bf568368a4f78` |
| src/week8/full_system.py | `52d807ada78e61133a51f98b7010e3f67b7f781d745c2bbbb101896f43b77d17` |
| scripts/week8/run_w8h_full_evaluation.py | `66398276a03406650f8716dd4e49fb91705703ff45f481427659385f2cddaa02` |
| src/week8/gate_diagnostics.py | `f6fab42df474e52979db082ee4a83436b111eae0fbe3cb9176006efeb1428d22` |
| src/week8/selective_controller.py | `9e2a1fa5dd27f0b57e225c61b2463d9efe27f3a9e304bc6e8924b443d83d21f3` |
| src/week8/semantic.py | `89b0f34620c8da0a058a1c0d689ef00b45b20afab8381853c848a3061f480b23` |
| src/week8/marginal_information.py | `51d69ee19d2621160cd6b55f37027f5fbe72e649f05b1069eff172f39d2f85fb` |
| src/week8/utility_models.py | `71a6939b803b1b3210f2fd8cc57293ba6fb117cd030f3a6219a93512787015fc` |
| src/week8/utility_dataset.py | `9427041d09eac38a4ea3e7e0091bcd75835c0449e73daa845a6badeb11d4bbd1` |
| src/week8/final_dev.py | `8a77014a2bff4859d27df6b9b094517e9e1d8b78473434232cf441d9980c4121` |
| src/week8/final_reporting.py | `6bf1bdaa6add645263a9555493eee279400a8260b8c3d7d9ac7fe59b489c3931` |
| scripts/week8/run_w8j_final_dev_benchmarks.py | `1f1c381c3a7ea276370974697b9ebe0823cc1e452708c99c4812a8475a3e5b39` |

The gate remains original B0-GW measured token uncertainty. Selection remains
the frozen S3/DraftSemantic policy with U4-A help/hurt safety probabilities.
The existing E5-DraftSemantic thresholds and single-promotion condition are
unchanged. G0 promotes exactly the selected compressed passage to FULL and
uses the original canonical regeneration prompt; it is not W9 F0 or G1.

## B. Producer registry

| Required final system | Status |
|---|---|
| D0 | executable; dry-run and fabricated fixture passed |
| semantic_k4 | executable; dry-run and fabricated fixture passed |
| semantic_k2 | executable; dry-run and fabricated fixture passed |
| semantic_k5 | executable; dry-run and fabricated fixture passed |
| semantic_k6 | executable; dry-run and fabricated fixture passed |
| semantic_k8 | executable; dry-run and fabricated fixture passed |
| BM25_k8 | executable; dry-run and fabricated fixture passed |
| D1 | executable; dry-run and fabricated fixture passed |
| W9-PRIMARY | executable; dry-run and fabricated fixture passed |
| W9-SAFE | executable; dry-run and fabricated fixture passed |
| W9-VERIFY | executable; dry-run and fabricated fixture passed |
| original_SARA_G0 | executable; dry-run and fabricated fixture passed |

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
  reproduced probabilities with maximum difference 0.0.
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
`0df18e811072a72a1fca5fe862a5d90fe1d45a2617c3f34af3142f23eaab4c08`. Existing A–G scientific artifacts were not edited.

## G. H0 v2 result

**PASS.** New immutable manifest:
`results/week9/w9h_final/v2/w9h_pretest_freeze_manifest.json`.

File SHA256: `376d1ad7c331c71e7bec01d9086b4e1c99ea5a2fb643266ae9602044da9767e6`.

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
