# Week12 corrected final development protocol — implemented

Implementation commit: `f1c07b536c9e370ebafcfda3a560c8258752d132`. Active branch: gagan. Active schema: W12-QBGF-v3. Runtime code hash: `a8b00f0998bfa458367d6fca43874aa9c227c8b4c4715681c3caddb986b6e6e0`. Split hash: `78f5ace11f9080d05dda12716e634020711182f721ac562dc81d19bad6c1c117`. This report supersedes v1 readiness for final-science decisions; old artifacts remain intact. v2 was a metadata-only audit draft, superseded before any model run because its QASPER confirmation would have had only three papers.

## Exact eligible membership

|   CONFIRM |   CONFIRM_groups |   TRAIN |   TRAIN_groups |   VAL |   VAL_groups | dataset   |
|----------:|-----------------:|--------:|---------------:|------:|-------------:|:----------|
|         0 |                0 |    2178 |            790 |  1005 |          281 | QASPER    |
|       200 |              200 |   90447 |          90439 |  7155 |         7155 | HotpotQA  |
|        50 |               43 |     531 |            325 |   177 |          157 | SciFact   |

Official source queries: QASPER TRAIN 2,593 / validation 1,005; HotpotQA TRAIN 90,447 / DEV 7,405; SciFact TRAIN 809 / DEV 300. QASPER official TEST metadata reports 416 papers, but TEST payloads were not opened. Saved W9 membership proves 1,309 historical TEST queries were consumed. TEST labels/predictions already exist historically; they are permanently excluded from this work. No locally configured official TEST scorer/labels are used for HotpotQA/SciFact; their held-out confirmation comes from DEV.

## Exposure exclusions and limitations

The audit enumerates 86,563 local structured historical artifacts, exact matching query IDs per week, W11 pilot IDs, file hashes and saved QASPER TEST membership. Two large W11 per-query CSVs were separately checked: they add no previously unknown exposure IDs. Remaining large unparsed surfaces are QASPER only and are covered conservatively by historical source-paper exclusion, not treated as evidence of freshness. Older project-local numeric IDs can refer to different source splits: QASPER papers are translated through the raw source indices, rather than trusting bare query numbers. Unavailable/off-machine history cannot be proven fresh by this audit.

858 QASPER TRAIN papers occur in historical project source pools. Only 39 unused queries on 30 papers remain. The pre-outcome v3 threshold requires 200 unused papers, allowing at least 20 confirmation paper clusters at 10%. Therefore **QASPER_FRESH_CONFIRMATION_AVAILABLE = FALSE**. No confirmation split is fabricated. Eligible official TRAIN excludes 98 historical validation papers (415 queries); the remaining 2,178 queries on 790 papers can supply gradients. The 1,005 official validation questions on 281 distinct papers supply validation, never confirmation. Historical training-label reuse is allowed and is not called unseen evaluation.

HotpotQA confirmation excludes historical pilot/matching question groups; duplicate normalized questions are kept together. All 90,447 eligible TRAIN questions remain in the sampling pool. Background documents can repeat between provided contexts; this is query-disjoint validation, not a claim of novel-document generalization.

SciFact grouping joins claims sharing cited documents transitively across TRAIN and DEV. Historical exposed claim/document groups and held-out groups are excluded from gradients. This leaves 531 eligible TRAIN claims, 177 VAL, 50 CONFIRM; the reduction from 809 TRAIN is leakage prevention, not a cap. Indexed corpus abstracts remain common background for retrieval.

All query-ID, normalized-question and grouping intersections between TRAIN/VAL/CONFIRM are zero. Full lists, exact source content revisions (SHA-256), creation timestamp, seed42, exclusions, split hash and git commit are in the immutable freeze. Compression is lossless storage only: tracked `.json.gz` copies restore byte-identical 108 MiB JSON; completed splits are never regenerated based on outcomes.

## Training and checkpoint selection

Every eligible TRAIN query enters exactly one corpus row. A deterministic hash of seed/dataset/query ID chooses one of RAW_SKIP, FIXED_SENTENCE or QBGF at 256/1024. No sixfold multiplication and no oracle evidence. All frozen VAL checkpoint rows use fixed QBGF-256; validation is never used for gradients. TRAIN corpus validation demands the exact full frozen TRAIN IDs, so accidental caps (including 500) fail before model loading.

Dataset-balanced sampling gives each dataset probability1/3 and samples uniformly within that dataset with replacement. An epoch is93,156 draws (approximately31,052/dataset), with5,823 optimizer updates at accumulation16. All source queries are eligible; replacement balancing does not guarantee every distinct Hotpot query is visited in one epoch. Raw counts, groups, mode/token statistics, expected/planned and completed draw/update contributions are recorded. Mixed accumulation steps may touch several datasets, so touched-step counts overlap.

Checkpoint selection minimizes equal-dataset mean query-mean answer NLL on VAL only, ties earliest. Task scores and SciFact compliance are recorded at validation. One epoch is the preregistered stopping rule; no large search or confirmation-based selection. Training loss/history, chosen checkpoint and actual completed schedule are saved. G keeps the matched RAW/QBGF × base/adapted NF4 2×2; retention requires paired VAL support and noninferiority, not a training-loss decrease.

## Notebook and confirmation guards

0 displays official counts, historical exposure, available pools, TRAIN/VAL/CONFIRM group counts, overlap matrix and QASPER status, without generation/training. C is now `week12_C_validation_surface.ipynb`; G `week12_G_lora_validation.ipynb`; H `week12_H_validation_analysis.ipynb`. Old named versions are archived. All ten defaults remain smoke-only. F displays counts, probabilities and corpus/step/length plans before user-enabled training. Complete identity-checked rows and corpus shards resume.

Week12 loaders permit only TRAIN/VAL. C/D/E/G use frozen VAL; H rejects results outside frozen VAL IDs; F validates TRAIN/VAL roles before gradients. QASPER TEST guard runs before input reads. I records the final reader/retriever/budgets/parameters, adapter hash, parser/prompt/code/config identities, frozen confirmation IDs, exact applicable method list, metrics and commit. It does not generate confirmation.

Week13 has a separate readiness-only default entrypoint. Full user opt-in requires a final freeze, unchanged scientific identity, unchanged adapter and identical immutable confirmation IDs. It materializes only those IDs; interrupted confirmation resumes completed shards without rerunning them. QASPER has zero confirmation rows and its historical TEST is never rerun. No final freeze exists yet, so Week13 is currently blocked. Confirmation statistics/plots remain Week13 work after the single campaign; they must not feed back into selection.

## Tests and smoke evidence

**107 tests passed, zero skips;10/10 notebook default executions passed.** Added checks cover every cross-role overlap, QASPER paper grouping/TEST guard, duplicate-question groups, historical pilot exclusions, uncapped all-TRAIN completeness, dataset balancing, VAL-only checkpoint objective, source-role refusal, immutable split hashes and frozen-only Week13 IDs. A tiny synthetic validation test exercises balanced NLL/task/compliance recording.

Bounded real smoke: three unique VAL queries total; QBGF source/budget checks passed. Canonical BM25 k5/n10 SARA and its separate adapted RAG reference each ran on the same QASPER smoke query. One real Qwen NF4 optimizer step used synthetic text only; no adapter was saved. Resume kept completed answer shards byte-identical. These are execution checks, not quality evidence. No LLMLingua result was fabricated: dependency/revision qualification remains unavailable.

Historical preservation: **2,671 tracked W0–W11 files byte-identical** to the pre-task working tree, including pre-existing user edits. No full validation, full training, TEST or confirmation run occurred.

## Manual execution and remaining concerns

See `notebooks/week12/README.md` for exact flags. Week12:0 → A → B → C → D → optional E → F → G → H → I. C/D/G require manual RUN_FULL=True; F requires RUN_TINY_LORA_SMOKE=False, RUN_FULL_LORA_TRAIN=True. H/I read real results with RUN_SMOKE=False. RESUME=True and OVERWRITE=False remain enabled.

Week13: finish/review VAL, freeze I, inspect `scripts/week13/run_confirmation.py` without flags, then manually opt in once with `--full --explicit`. Analyze the saved frozen campaign; never tune after its outcomes. QASPER contributes validation/historical evidence only, not a new final-confirmation claim.

This is substantially larger than the pilot: C133,392 logical portable rows plus2,010 specialist QASPER rows; D116,718 rows; optional E16,674; G66,696. Cache reuse affects physical calls; these are not independent questions. About18 GiB free remains: full all-TRAIN preparation may need substantially more embedding/cache/checkpoint storage and time. Smoke does not certify full-run feasibility. The agent did not start any full workload.
