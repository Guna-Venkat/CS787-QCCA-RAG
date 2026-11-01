# Week 12 — corrected TRAIN → VAL → confirmation protocol

Active protocol: **v3**, unchanged splits/config; see `results/week12/W12_SCHEDULER_CORE_REPORT.md` for the execution/scheduler patch. Active protocol file (`configs/week12/protocol.yaml`). Existing v1 results are historical development; v2 is a metadata-only superseded split audit. Neither contributes final method-selection outcomes. The original 50-query W11 cohort is **HISTORICAL_DEVELOPMENT / HYPOTHESIS_GENERATION**.

| Dataset | All eligible TRAIN | Frozen VAL | Frozen CONFIRM |
|---|---:|---:|---:|
| QASPER | 2,178 | 1,005 | 0 |
| HotpotQA | 90,447 | 7,155 | 200 |
| SciFact | 531 | 177 | 50 |

QASPER has no credible fresh confirmation pool: only 39 unused TRAIN questions on 30 papers remain outside the historical source-paper union. The prospectively specified requirement is 200 unused papers, giving at least 20 independent confirmation papers at 10%. QASPER TEST is permanently blocked. QASPER uses eligible official TRAIN and official validation for development only. HotpotQA/SciFact use group-disjoint held-out official DEV subsets because no official TEST scorer/labels are locally configured. No confirmation example was evaluated.

TRAIN may reuse historical training data; historical QASPER validation papers are excluded from gradients. SciFact grouping joins shared cited documents across official TRAIN and DEV; Hotpot grouping joins normalized duplicate questions. A shared indexed background document is not equated with an exposed labeled question. See the exposure audit and frozen membership for precise exclusions.

## Run All and manual flags

**CORE_FINAL=True is the default.** C runs NONE, RAW_SKIP, FIXED_SENTENCE, H3 and QBGF at 256/1024, plus C2 at its preregistered primary budget 256. D runs QBGF_FULL, QBGF_NO_QUERY, QBGF_SENTENCE_ONLY and QBGF_PASSAGE_ONLY at 256/1024, reusing compatible C answers. NO_MASS, GREEDY and BM25 remain secondary. Natural-budget references, RAW_STRICT and other ablation/sensitivity arms remain available with explicit CORE_FINAL=False (`--include-secondary` in the CLI); they are not automatically launched. The original 39-contrast primary family remains intact, with optional E/G arms pending until explicitly run. Twelve prospective fusion diagnostics compare QBGF with the sentence-only/passage-only controls; analysis also provides joint Holm correction over all 51 contrasts.

Every notebook defaults to synthetic smoke or metadata, **RUN_FULL=False**, **RESUME=True**, **OVERWRITE=False**. C/D/E/G use only frozen VAL; F uses TRAIN gradients and VAL checkpoint selection; H/I read saved VAL results. No notebook evaluates confirmation.

1. `week12_0_protocol_and_split_audit.ipynb`: Run All. Inspect counts, exposure, group overlaps and QASPER status. No generation/training.
2. `week12_A_qbgf_algorithm.ipynb`: algorithm/synthetic smoke. Optional `RUN_REAL_MODEL_SMOKE=True`: at most three VAL queries total.
3. `week12_B_baseline_qualification.ipynb`: qualification/smoke only. Optional `RUN_CANONICAL_SMOKE=True` qualifies original adapted Mistral references on one QASPER VAL query each. Both remain separate specialist references.
4. `week12_C_validation_surface.ipynb`: manually set **RUN_FULL=True** for the entire frozen VAL surface. Full canonical references require B qualification.
5. `week12_D_qbgf_ablations.ipynb`: manually set **RUN_FULL=True**, VAL only.
6. Optional E: the official smaller multilingual-BERT LLMLingua-2 checkpoint is **QUALIFIED** in an isolated environment and pinned by a separate runtime lock. The original frozen protocol YAML and splits remain unchanged. Full work requires **RUN_LLMLINGUA_FULL=True**. Downloads remain explicitly opt-in. Changing frozen scientific config requires a new output version.
7. F: first inspect counts, available disk and workload. For real training set **RUN_TINY_LORA_SMOKE=False**, **RUN_FULL_LORA_TRAIN=True**. Corpus preparation resumes by source query. No 500-query cap. The notebook shows group, evidence-mode, sampling, token length and optimizer-step tables before training.
8. `week12_G_lora_validation.ipynb`: after full training, manually set **RUN_FULL=True** for RAW_SKIP/QBGF × BASE_NF4/LORA_NF4 on VAL.
9. `week12_H_validation_analysis.ipynb`: set **RUN_SMOKE=False** to read real VAL results. No generation.
10. I: set **RUN_SMOKE=False**, supply **FINAL_METHOD**, **FINAL_REASON**, optional **RETAIN_LORA**. Paired conservative gates apply. Writes immutable final freeze; never evaluates confirmation.

Changing RUN_FULL automatically disables effective synthetic smoke for C/D/G. Keep resume enabled. A changed scientific identity fails instead of overwriting completed rows. Complete checksum-validated query/action rows are reused.

## LoRA policy

All eligible TRAIN queries enter the pool, with exactly one deterministic evidence mode/budget per original TRAIN query: hash(seed,dataset,query_id) modulo the six RAW_SKIP/FIXED_SENTENCE/QBGF × 256/1024 conditions. VAL checkpoint examples use fixed QBGF-256. No oracle evidence is used.

Final training uses a seeded full permutation **without replacement**, visiting every eligible TRAIN query exactly once in the registered one epoch. Dataset imbalance is handled by loss weights `w_d = N / (D * N_d)` applied to query-mean answer NLL, with `N=93,156` and `D=3`. Each dataset therefore has equal nominal epoch loss mass, while no Hotpot example is omitted by replacement sampling. The small balanced replacement sampler remains available for smoke only.

`coverage_events.jsonl` durably records each successful training forward and applied/skipped optimizer step. `sampling_actual.json` reports unique examples seen, repeats across interrupted attempts, missing examples, applied loss weights, update touches and loss-coefficient mass. Unfinished forward windows count as seen but not as applied optimizer updates. Gradient clipping and Adam mean loss coefficients are not gradient-norm or parameter-change attribution. Before full training, actual TRAIN consumption is zero; planned coverage must not be presented as observed.

Checkpoint objective: minimum equal-dataset mean query-mean answer NLL on frozen VAL; ties retain the earlier checkpoint. One epoch is the preregistered stopping rule. VAL task scores and SciFact format compliance are also saved. Both base and adapted G readers use identical NF4 precision/evidence. LoRA retention requires paired VAL support, not training loss.

This is a much larger workload than v1. No full runtime or disk-feasibility claim follows from three-query smoke. About 18 GiB free was measured after data preparation; arrange additional cache/checkpoint capacity before full corpus preparation/training if needed.

## Metadata and fresh checkout

Full immutable IDs/groups are in `results/week12/W12_SPLIT_FREEZE.json` and the matching active protocol file. Lossless `.json.gz` copies are tracked to avoid committing 108 MiB raw JSON files. Restore with:

```bash
sara_env/bin/python scripts/week12/restore_metadata.py
# Only when official TRAIN files are absent:
sara_env/bin/python scripts/week12/prepare_training_data.py --download
```

Sources are pinned by exact content SHA-256 (including Arrow files and dataset metadata), not by a mutable dataset name alone. Repeated notebook 0 execution reads the existing freeze and verifies hashes; it never resplits after outcomes. A changed split/config requires a new version.

## Week 13 — after final freeze only

1. Finish all desired VAL experiments, review H, freeze the final method/LoRA decision in I.
2. Run the readiness-only command (no generation):
   `sara_env/bin/python scripts/week13/run_confirmation.py`
3. Only after review, the user manually runs:
   `sara_env/bin/python scripts/week13/run_confirmation.py --full --explicit`
4. One frozen confirmation campaign, resuming unfinished shards only; no tuning or configuration edits afterward. All applicable methods see identical frozen confirmation IDs. QASPER has no confirmation rows; historical QASPER TEST is never rerun.
5. Analyze saved confirmation results with dataset-specific metrics, paired grouped bootstrap and the frozen primary family. Do not mix VAL and CONFIRM or compare specialist SARA as a controlled Qwen ablation.

Full flags remain disabled. The agent has run tests and bounded smoke only.

Read actual TRAIN coverage without launching a model: `sara_env/bin/python scripts/week12/training_coverage.py`. Before training this reports zero actual consumption; during interrupted work the durable ledger retains the physical repeats.

Final pre-run CORE D patch and revised cost: `results/week12/W12_CORE_D_FUSION_PATCH.md`. No split, retriever, QBGF definition or confirmation identity changed.

### C-EXT — additive baseline completion

After the existing C completes, open `week12_C_EXT_baseline_extension.ipynb`, set
`RUN_C_EXT_FULL=True`, then Run All. Keep `RESUME=True`, `OVERWRITE=False`.
This runs budget-matched BM25/SFR RAW_SKIP passage RAG followed by a real one-query
canonical BM25-SARA qualification and its QASPER-only specialist reference.
`RUN_SFR_SARA=True` additionally requests qualification/full execution of SFR-SARA
k5/n10; it is disabled until requested. Qualification failure stops that route,
never substitutes another checkpoint or retriever. For qualification only, leave
full disabled and set `RUN_SPECIALIST_QUALIFICATION=True`.

H/I display merged C/C-EXT metrics plus separately labelled specialist references.
I requires the two complete matched controls and canonical BM25 SARA before final
freeze. Historical W10 SFR-SARA k6 remains historical, never a newly paired score.
Supplemental extension comparisons have a separate 12-test Holm correction and
are exploratory additions; original primary registries and frozen science stay intact.

### After D: H, then optional attribution diagnostic J/K/L

E/F/G full experiments are skipped by user decision; qualification/synthetic smoke
is not a replacement quality result. H defaults to read-only saved analysis and
shows `optional_phase_decisions.json`. Original multiplicity families stay unchanged.

J: set `RUN_SMOKE=False` and Run All to inspect/reuse the frozen 50-query-per-dataset
VAL diagnostic and six full-VAL-selected comparators. Cached setup does not call an LM.
K: set `RUN_FULL=True` only when ready for real perturbations; completed immutable
actions resume. L: after K, set `RUN_SMOKE=False` and Run All for real saved analysis.
J/K/L default to synthetic smoke, never full generation. The diagnostic is causal
input ablation, not internal-model explanation; it does not inspect confirmation.
See `results/week12/W12_ATTRIBUTION_READINESS.md` for IDs, call budget, limits and tests.
