# Week12 M — implementation and untouched-confirmation readiness

Part M is implemented on `gagan`. No real confirmation question, gold answer, prediction or outcome was inspected during implementation. No real confirmation generation or scoring was executed. Default notebook execution completed metadata checks only. Part K's process and scientific implementation were left unchanged.

## Frozen contract verified

- Both Part-I freeze files remain byte-identical: SHA256 `d0188a3582d6aead0a3a31f9f15bad9b51a88a6b475b05ac79ae38e58f05366b`.
- Split SHA256: `78f5ace11f9080d05dda12716e634020711182f721ac562dc81d19bad6c1c117`.
- Captured source implementation commit: `93992932e29252bcedbc56bda02e0c8ae3926001`.
- Confirmation-ID hash: `c35e3ac9b3855688a50787b8efafa0b49a002cd1ffd8098ced10f195ad3bb0a4`.
- Frozen final policy: SFR top-10 → FIXED_SENTENCE → pinned Qwen/Qwen3-8B BASE_FP16. No LoRA/adaptation.
- Systems: NONE, RAW_SKIP, FIXED_SENTENCE, H3, QBGF_FULL; budgets 256/1024.

| Dataset | Queries | Frozen groups |
|---|---:|---:|
| HotpotQA | 200 | 200 |
| SciFact | 50 | 43 |
| QASPER | 0 | 0 |

Exactly 2,500 logical rows. Conservative bound: 2,250 distinct generations before further exact-evidence reuse. NONE's identical answer is reused across budget labels. QASPER TEST stays permanently closed. No split was regenerated.

## Implementation

- `src/week12_confirmation/`: metadata/freeze guards, gated cohort loader, immutable preparation and answer caches, completion validation, task-specific summaries, predefined statistical analysis and report writer, isolated synthetic fixtures.
- `scripts/week12/run_final_confirmation.py`: metadata-only default; CPU synthetic mode; explicitly gated full execution in separate SFR preparation / Qwen generation / analysis processes.
- `notebooks/week12/week12_M_final_confirmation.ipynb`: safe Run All defaults, manual full-run unlock and durable resume.
- `tests/week12/test_confirmation_m.py`: 31 tests; existing notebook registry test updated for M.
- `results/week12/.gitignore`: excludes only synthetic caches and detailed M resume shards; compact protocol, registry, summaries and reports remain versionable.

Per-query shards preserve evidence/source IDs and hashes, exact token IDs, prompt hash, source spans, reader/tokenizer/generation identities, predictions, task scores and group. Incompatible caches/results fail loudly. Completed answers are reused; no result replacement or outcome-driven exclusion is supported. Completion requires the exact grid, evidence budgets, source validation, frozen prompt/model/parser identities and checksums.

Predefined analysis uses 10,000 paired frozen-group bootstrap resamples, seed 42, nominal 95% intervals. Eight primary contrasts compare FIXED_SENTENCE against NONE/RAW_SKIP; four secondary contrasts compare QBGF_FULL. Separate Holm families are declared before outcomes. H3 stays descriptive. SciFact macro-F1 is recomputed from labels within every bootstrap sample; invalid outputs remain incorrect. VAL/confirmation sample shifts are descriptive, never paired tests. No pooled dataset score or retrospective method promotion.

## Checks performed

- Full Week12 unit suite: **153 tests passed** (including 31 new M tests).
- CPU synthetic fixture: **20/20 logical rows**, five systems × two budgets × two fabricated examples; preparation, generation stub, completion checks, statistics and report plumbing passed.
- Resume test: **zero additional synthetic reader calls**, unchanged completed shard timestamps.
- Default notebook Run All: **passed**, metadata-only; no real cohort access.
- No real confirmation smoke, full validation, model training or confirmation execution.

The synthetic report is explicitly labelled synthetic and is not scientific evidence. The real `week12_M_final_confirmation_report.md` is generated only after authorized real execution and complete result validation.

## Manual execution later

Open `notebooks/week12/week12_M_final_confirmation.ipynb`. Keep defaults for readiness only. To launch the frozen confirmation after attribution completes, change only:

```python
RUN_FULL = True
ALLOW_CONFIRMATION_LABEL_ACCESS = True
RUN_CONFIRMATION_ANALYSIS = True
CONFIRM_FINAL_FREEZE_SHA256 = "d0188a3582d6aead0a3a31f9f15bad9b51a88a6b475b05ac79ae38e58f05366b"
```

Keep `CORE_FINAL=True`, `RUN_SMOKE=False`, `RESUME=True`, `OVERWRITE=False`, `FINAL_METHOD="FIXED_SENTENCE"`, `RETAIN_LORA=False`. Run All. Re-running uses immutable completed selections/answers. Full execution refuses to start while K's full generation process is present.

Equivalent command:

```bash
sara_env/bin/python scripts/week12/run_final_confirmation.py --stage full --full --explicit --allow-confirmation-label-access --run-confirmation-analysis --freeze-sha d0188a3582d6aead0a3a31f9f15bad9b51a88a6b475b05ac79ae38e58f05366b
```

## Resources and limits

Latest observation: approximately **9.15 GiB free disk**, GPU idle (0% utilization; about 2.3 GiB resident memory). K had completed before the final observation. No caches were deleted or weights downloaded by M. Full execution checks disk again and refuses below 2 GiB; this threshold is a guard, not a guarantee against external disk use.

Only 250 confirmation examples, two datasets, no fresh QASPER confirmation; SciFact uncertainty may be substantial. D's query-conditioning result remains VAL-only; J/K/L is post-hoc attribution. LLMLingua and LoRA are excluded. Any technical failure must resume the exact frozen configuration; confirmation outcomes cannot justify tuning, replacements or promotion.
