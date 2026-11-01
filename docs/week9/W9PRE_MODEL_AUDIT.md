# W9-PRE measured feasibility audit

BRANCH = gagan. Audit date: 2026-10-20. This is bounded FIT feasibility evidence, not a model-quality benchmark or a frozen panel.

## Environment and scope

NVIDIA TITAN RTX, compute capability 7.5, 24 GiB. Strict ceiling: peak allocated VRAM <20,000,000,000 bytes. Reserved memory is separately recorded and is not the pass criterion. Software: torch 2.14.0, Transformers 4.57.6, bitsandbytes 0.50.2, huggingface-hub 0.36.2. One generator is loaded at a time, released before the next. Eager attention and explicit precision avoid assuming native BF16 support on Turing.

The fixed screen set is 49 FIT queries from 14 whole papers (nominal 48; paper overshoot allowed), seed 42. Type/base-correctness strata are heuristics based only on FIT. Smoke uses its first two distinct queries for M0/M2 and one distinct query for the M3 FP32 diagnostic. Repeated precision/resume probes reuse these queries. INTERNAL-VAL, DEV and TEST did not select models.

## Cache audit

| ID | Official checkpoint | Cache/access outcome |
|---|---|---|
| M0 | mistralai/Mistral-7B-Instruct-v0.2 | COMPLETE; 14,483,498,016 weight bytes; existing snapshot verified |
| M1 | meta-llama/Llama-3.1-8B-Instruct | GatedRepoError; no usable access in this session |
| M2 | Qwen/Qwen2.5-7B-Instruct | COMPLETE; 15,231,271,888 weight bytes |
| M3 | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | COMPLETE; 15,231,271,850 weight bytes |
| M4 | Qwen/Qwen3-8B | Metadata accessible; 16,381,516,776 weight bytes; deferred to retain the configured 8 GB disk reserve |
| M5 | google/gemma-3-12b-it | GatedRepoError; no usable access in this session |

Exact immutable commits, paths and access errors are in `results/week9/preflight/model_cache_manifest.json`. Weight-byte sizes exclude ancillary tokenizer/config metadata. No user model cache was deleted. Complete pinned snapshots are recognized even when `refs/main` is absent. Access gates were not accepted automatically.

## Measured precision results

| Model | Weights / compute | Peak allocated bytes | Numerical/memory outcome | Interface outcome |
|---|---|---:|---|---|
| M0 | FP16 | 14,631,406,592 | PASS | 10 calls; no empty outputs; all 4 T1/T2 extraction outputs malformed and capped |
| M2 | FP16 | 15,325,298,176 | FAIL: non-finite generation scores | No quality result inferred |
| M2 | LLM.int8, FP16 unquantized modules | 8,962,191,360 | PASS | 10 calls; no empty, truncated or malformed outputs |
| M3 | FP16 | 15,342,163,968 | FAIL: non-finite generation scores | No quality result inferred |
| M3 | LLM.int8, FP16 unquantized modules | 8,941,492,224 | FAIL: non-finite generation scores | No quality result inferred |
| M3 | NF4 double quant, FP16 compute | 5,742,134,272 | FAIL: non-finite generation scores | No quality result inferred |
| M3 | NF4 double quant, FP32 compute/unquantized modules | 11,612,920,320 | PASS | 5/5 final answers empty and truncated; NOT interface-qualified |

M0 peak reserved: 14,860,419,072 bytes. M2 int8 peak reserved: 14,313,062,400 bytes. M3 NF4/FP32 peak reserved: 11,945,377,792 bytes. Main status and per-task public results retain measured prompt/generation lengths and timing; failed numerical probes retain their failure and memory diagnostics rather than substitute outputs.

M0's greedy extraction generated a partial table header/whitespace to the 256-token cap. This is a real format failure, not a parser success. M0 is retained as the mandatory canonical reference, with this caveat visible. PRE screen P1/P2 use fixed rule-based evidence shared by every model, so they evaluate reading/revision and do not establish that a model is a qualified extractor. The optional D/E extractor experiment must still check actual structured extraction.

M3's explicitly labelled diagnostic uses `configs/week9/w9pre_deepseek_nf4_fp32.yaml` and a distinct output/cache directory. Answer caps are 128 tokens, extraction caps 256. Reasoning consumed every budget without an identifiable final-answer boundary. No reasoning-only prefix is counted as a final answer. M3 is deferred under this configuration; changing its budget would require an explicit new FIT configuration/cache and cost comparison, not a hidden allowance.

## Scientific and publication limits

Two queries cannot establish model superiority. Tiny-sample F1/HELP/HURT in smoke summaries are diagnostics only. The evidence-utilization field is a nonempty-output/evidence proxy, not semantic grounding. Extraction faithfulness measures source quote support where available; answer entailment is unmeasured. The 0.003 F1 tolerance does not override mandatory qualitative review before panel freezing.

All generated tokens, including reasoning, enter compute totals. Raw outputs remain in permission-0600 ignored private caches; public results contain parsed final content and cost/failure metadata. Fixed source spans are shared across models. Initial canonical SARA draft tokens are included when the draft is used; full deployment end-to-end latency remains unmeasured.

No alternative receives the trained Mistral projector. PRE textual M0 is also not a fresh evaluation of the complete trained SARA first-pass pipeline. Runtime hard checks validate both configured and actual checkpoint base-model identity before latent injection.

## Completed checks and next action

Actual smoke resume succeeded for M0/M2 and the separate M3 diagnostic. Resume can retry a failed earlier precision before reusing the complete successful precision; it does not imply zero model loads. No screen, model-panel freeze, alternative projector training, full W9 phase or TEST run was launched.

Run manually from the repository root:

```bash
./sara_env/bin/python scripts/week9/run_w9pre_models.py --stage screen --model M0 --model M2 --resume
```

Then inspect the fixed qualitative sample and documented faithfulness/format caveats. Follow `WEEK9_RUNBOOK.md` to record review and freeze a FIT-only panel. The currently recommended smoke/screen IDs are execution candidates, not an already selected panel. Cache/prompt/source identities reject incompatible reuse; do not erase stale caches to force acceptance.
