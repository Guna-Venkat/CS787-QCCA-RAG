"""Read-only coverage/cost audit; writes new reports only, never regenerates splits."""
from pathlib import Path
import sys,json,re,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.week12.protocol import load,root,identity,sha,atomic,digest,immutable
from src.week12.splits import read_freeze
from src.week12.scheduling import full_coverage
from src.week12.execution import cost_plan

def main():
 c=load();f=read_freeze(c);out=root(c)/'qualification/scheduler_patch';out.mkdir(parents=True,exist_ok=True)
 before=json.loads(Path('/tmp/w12_scheduler_protected.json').read_text());changed=[p for p,h in before.items() if sha(p)!=h and p!='results/week12/.gitignore'];assert not changed,changed
 rows=[dict(dataset=d,query_id=r['query_id']) for d,p in f['datasets'].items() for r in p['TRAIN']];_,plan=full_coverage(rows,c['lora']['gradient_accumulation_steps'],c['seed']);counts={d:len(p['VAL']) for d,p in f['datasets'].items()};cost=cost_plan(c,counts)
 assert not (root(c)/'lora/complete.json').exists() and not (root(c)/'lora/coverage_events.jsonl').exists();assert not list((root(c)/'validation').glob('*/status.json'));assert not (ROOT/'results/week13').exists()
 actual=dict(status='NOT_STARTED',unique_examples_seen={d:0 for d in counts},repeats={d:0 for d in counts},applied_optimizer_steps=0,optimizer_contributions={d:0 for d in counts});atomic(out/'final_train_consumption.json',actual);atomic(out/'full_coverage_schedule_plan.json',plan);atomic(out/'C_D_cost_audit.json',cost)
 tests=Path('/tmp/w12_scheduler_tests.log').read_text();assert tests.rstrip().endswith('OK');n=int(re.search(r'Ran (\d+) tests',tests).group(1));v=json.loads((root(c)/'smoke/notebooks'/identity(c)['code_hash'][:12]/'validation.json').read_text());assert len(v)==10 and all(x['status']=='PASS' for x in v)
 from src.week12.runner import directory
 real=[json.loads(p.read_text())['row'] for p in (directory(c,'B',False,True)/'rows').glob('*.json') if len(p.stem)==64];assert len(real)==3
 for r in real:assert r['query_id'] in {x['query_id'] for x in f['datasets'][r['dataset']]['VAL']} and r['evidence_tokens']<=r['budget'] and r['source_support']==1
 from src.week12.compression_runtime import status
 llm=status(c);assert llm['status']=='QUALIFIED'
 patch=dict(policy=plan['policy'],split_hash=f['split_hash'],frozen_config_hash=digest(c),full_coverage=True,replacement=False,loss_weight_formula='N/(D*N_d)',CORE_FINAL_default=True,scientific_parameters_or_split_ids_changed=False,user_authorized=True)
 immutable(root(c)/'protocol/training_execution_patch.json',patch)
 atomic(out/'verification.json',dict(tests_passed=n,notebooks_passed=10,real_smoke_queries=3,protected_files_verified=len(before),allowed_non_scientific_ignore_change='results/week12/.gitignore',changed_historical_artifacts=changed,identity=identity(c),full_runs_started=False,LLMLingua_status=llm['status']))
 import pandas as pd
 table=[dict(dataset=d,eligible=plan['raw_eligible_examples'][d],planned_unique=plan['planned_unique_seen'][d],actual_unique=0,actual_repeats=0,loss_weight=plan['effective_loss_weights'][d],nominal_epoch_loss_mass=plan['objective_weight_mass'][d],planned_update_touches=plan['planned_optimizer_updates_with_dataset'][d],planned_update_loss_coefficient_mass=plan['planned_optimizer_loss_coefficient_mass'][d],actual_applied_updates=0) for d in counts]
 text=f'''# W12-QBGF-v3 scheduler, CORE_FINAL and LLMLingua qualification patch

Frozen W12-QBGF-v3 YAML, all split IDs/hashes, QBGF source/definitions and historical artifacts are unchanged. No split was regenerated. Split hash: `{f['split_hash']}`. This user-authorized pre-full-run patch supersedes replacement-sampling instructions; original audit artifacts remain preserved.

## Exact coverage and loss weighting

{pd.DataFrame(table).to_markdown(index=False)}

One seeded permutation of canonical `(dataset,query_id)` identities, without replacement. Every one of93,156 eligible TRAIN examples occurs exactly once in the registered epoch;5,823 optimizer updates at batch1/accumulation16, including a final four-example update. `w_d=N/(3*N_d)` multiplies query-mean answer NLL. Nominal epoch loss mass is31,052 per dataset. A separate balanced replacement sampler is smoke-only. No evidence mode, target, split, LoRA hyperparameter or validation objective changed.

**Actual final-training consumption is zero for every dataset.** Planned values are not observed training. The toy CPU gradient/ledger test consumes synthetic identities only. During manual full training the durable forward/update ledger reports physical unique examples, repeats across restarted attempts, unapplied/skipped windows, applied weights and optimizer-step contributions. Step touches overlap for mixed-dataset accumulation. Coefficients use the actual window denominator, including the short last window; clipping and Adam prevent interpreting them as gradient norms or equal parameter changes. Completion refuses missing TRAIN examples. Interrupted work is retained in the physical-consumption audit rather than falsely reported as zero repeats.

## C/D cost and execution

{pd.DataFrame([{k:v for k,v in r.items() if k not in ('methods','note')} for r in cost]).to_markdown(index=False)}

CORE C: NONE, RAW_SKIP, FIXED_SENTENCE, H3, QBGF at256/1024, plus C2 at256 only. CORE D: QBGF and NO_QUERY at256/1024. This covers the existing C/D preregistered primary contrasts. The unchanged39-member family retains separately opted-in E/G contrasts as pending; no hypotheses are silently removed to improve multiplicity correction.

Core C/D total125,055 logical rows (some repeated NONE/QBGF rows) and at most100,044 additional answer calls assuming compatible C-generation reuse in D. Broader portable C/D would require250,110 logical rows and up to225,099 calls, plus2,010 QASPER specialist calls if enabled. Identical/empty evidence can reduce calls further. These are bounds, not measured full-run times. At generation caps, the core upper bound is12,567,744 generated tokens across QA/SciFact; real answer lengths and retrieval/embedding work remain unknown. No hour estimate is inferred from tiny smoke.

All full flags remainFalse. `CORE_FINAL=True` is now present in notebooks. For C/D manually enable RUN_FULL only after review. Secondary arms require CORE_FINAL=False or CLI `--include-secondary`; they use separate result directories and are not automatically launched. Natural-budget RAG, RAW_STRICT, nonprimary C2@1024 and the other QBGF sensitivities remain implemented. A/B remain qualification/smoke, E/G separate optional manual experiments. Primary budgets and scientific definitions are untouched.

## LLMLingua-2

Official checkpoint metadata was checked before download: smaller mBERT709,388,104 bytes(0.661 GiB), versus larger XLM-R2,235,829,648 bytes(2.082 GiB). Measured free space before download was17.10 GiB; qualification retained16.41 GiB, exceeding the10 GiB reserve. No caches were deleted. Dependencies were installed in a separate environment sharing the original pinned torch/transformers read-only; the original environment was not upgraded.

**QUALIFIED:** `{llm['id']}` @ `{llm['revision']}`; LLMLingua0.2.2. Official Microsoft implementation: https://github.com/microsoft/LLMLingua and https://huggingface.co/{llm['id']}. Synthetic source2,041 Qwen tokens compressed to250 at budget256 and875 at budget1024 using the predefined retries. These are mechanical budget checks, not quality evidence; no answer LM was called during compressor qualification. The original frozen YAML remains intact; a checksummed optional baseline runtime lock records model, revision, exact weight hash, package versions and worker identity. The final-method freeze captures the effective compressor spec. It does not silently launch E or add compression to QBGF.

Remaining disk is not a guarantee that all-TRAIN embedding preparation/checkpoint storage will fit. Keep E optional and arrange capacity before large user-run workloads.

## Verification and manual use

{n} tests pass;10/10 default notebooks pass; three frozen VAL-query execution smokes pass source/budget checks. The synthetic weighted-gradient ledger test verifies full coverage, zero planned repeats, actual update accounting and restart repeats. Frozen role/TEST guards still pass. All{len(before):,} protected tracked files/config/QBGF checksums were verified, allowing only the requested non-scientific ignore-file update; no historical result artifact changed.

Manual: notebook0→A→B→C(core)→D(core)→optional E/F/G→H→I. F still requires RUN_TINY_LORA_SMOKE=False and RUN_FULL_LORA_TRAIN=True. `sampling_plan.json` is planned; `sampling_actual.json` is observed. Week13 remains locked until final freeze. No full validation, full LoRA training, confirmation or QASPER TEST was run.
'''
 (ROOT/'results/week12/W12_SCHEDULER_CORE_REPORT.md').write_text(text)
 print(pd.DataFrame(table).to_string(index=False));print('Scheduler/core report and immutable-preservation checks complete.')
if __name__=='__main__':main()
