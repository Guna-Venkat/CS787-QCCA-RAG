# Week 12 — corrected TRAIN → VAL → confirmation protocol

Active protocol: **v3** (`configs/week12/protocol.yaml`). Existing v1 results are historical development; v2 is a metadata-only superseded split audit. Neither contributes final method-selection outcomes. The original 50-query W11 cohort is **HISTORICAL_DEVELOPMENT / HYPOTHESIS_GENERATION**.

| Dataset | All eligible TRAIN | Frozen VAL | Frozen CONFIRM |
|---|---:|---:|---:|
| QASPER | 2,178 | 1,005 | 0 |
| HotpotQA | 90,447 | 7,155 | 200 |
| SciFact | 531 | 177 | 50 |

QASPER has no credible fresh confirmation pool: only 39 unused TRAIN questions on 30 papers remain outside the historical source-paper union. The prospectively specified requirement is 200 unused papers, giving at least 20 independent confirmation papers at 10%. QASPER TEST is permanently blocked. QASPER uses eligible official TRAIN and official validation for development only. HotpotQA/SciFact use group-disjoint held-out official DEV subsets because no official TEST scorer/labels are locally configured. No confirmation example was evaluated.

TRAIN may reuse historical training data; historical QASPER validation papers are excluded from gradients. SciFact grouping joins shared cited documents across official TRAIN and DEV; Hotpot grouping joins normalized duplicate questions. A shared indexed background document is not equated with an exposed labeled question. See the exposure audit and frozen membership for precise exclusions.

## Run All and manual flags

Every notebook defaults to synthetic smoke or metadata, **RUN_FULL=False**, **RESUME=True**, **OVERWRITE=False**. C/D/E/G use only frozen VAL; F uses TRAIN gradients and VAL checkpoint selection; H/I read saved VAL results. No notebook evaluates confirmation.

1. `week12_0_protocol_and_split_audit.ipynb`: Run All. Inspect counts, exposure, group overlaps and QASPER status. No generation/training.
2. `week12_A_qbgf_algorithm.ipynb`: algorithm/synthetic smoke. Optional `RUN_REAL_MODEL_SMOKE=True`: at most three VAL queries total.
3. `week12_B_baseline_qualification.ipynb`: qualification/smoke only. Optional `RUN_CANONICAL_SMOKE=True` qualifies original adapted Mistral references on one QASPER VAL query each. Both remain separate specialist references.
4. `week12_C_validation_surface.ipynb`: manually set **RUN_FULL=True** for the entire frozen VAL surface. Full canonical references require B qualification.
5. `week12_D_qbgf_ablations.ipynb`: manually set **RUN_FULL=True**, VAL only.
6. Optional E: LLMLingua remains **NOT_QUALIFIED** until official dependency + pinned model revision qualify. Full work requires **RUN_LLMLINGUA_FULL=True**. Downloads remain explicitly opt-in. Changing frozen scientific config requires a new output version.
7. F: first inspect counts, available disk and workload. For real training set **RUN_TINY_LORA_SMOKE=False**, **RUN_FULL_LORA_TRAIN=True**. Corpus preparation resumes by source query. No 500-query cap. The notebook shows group, evidence-mode, sampling, token length and optimizer-step tables before training.
8. `week12_G_lora_validation.ipynb`: after full training, manually set **RUN_FULL=True** for RAW_SKIP/QBGF × BASE_NF4/LORA_NF4 on VAL.
9. `week12_H_validation_analysis.ipynb`: set **RUN_SMOKE=False** to read real VAL results. No generation.
10. I: set **RUN_SMOKE=False**, supply **FINAL_METHOD**, **FINAL_REASON**, optional **RETAIN_LORA**. Paired conservative gates apply. Writes immutable final freeze; never evaluates confirmation.

Changing RUN_FULL automatically disables effective synthetic smoke for C/D/G. Keep resume enabled. A changed scientific identity fails instead of overwriting completed rows. Complete checksum-validated query/action rows are reused.

## LoRA policy

All eligible TRAIN queries enter the pool, with exactly one deterministic evidence mode/budget per original TRAIN query: hash(seed,dataset,query_id) modulo the six RAW_SKIP/FIXED_SENTENCE/QBGF × 256/1024 conditions. VAL checkpoint examples use fixed QBGF-256. No oracle evidence is used.

Dataset probability is 1/3 each; within a dataset sample uniformly with replacement. One epoch is 93,156 draws, so approximately 31,052 draws/dataset and 5,823 optimizer updates at accumulation 16. All queries are eligible; balanced replacement does **not** guarantee every unique Hotpot query is visited in one epoch. Planned and completed schedule contributions are recorded separately. Mixed accumulation steps can contain several datasets.

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
