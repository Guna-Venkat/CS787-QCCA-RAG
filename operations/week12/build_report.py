"""Build the implementation report from actual qualification evidence, no experiment work."""
from pathlib import Path
import sys,json,hashlib,subprocess,re
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.week12.protocol import load,root,identity,sha,atomic
from src.week12.qualification import inventory
from src.week12.runner import directory
import pandas as pd

def main():
 c=load();out=root(c);qual=out/'qualification';qual.mkdir(parents=True,exist_ok=True)
 before=json.loads(Path('/tmp/w12_history_before.json').read_text());changed=[p for p,h in before.items() if not (ROOT/p).exists() or sha(ROOT/p)!=h];assert not changed,changed
 atomic(qual/'historical_integrity.json',dict(passed=True,tracked_historical_files_verified=len(before),before_manifest_hash=sha('/tmp/w12_history_before.json'),changed_files=changed,scope='Byte comparison against actual working-tree contents at task start, including pre-existing user notebook edits. No W0-W11 source/results/notebook file was modified.'))
 suite=json.loads((qual/'smoke_suite.json').read_text());assert suite['identity']==identity(c),'Smoke suite identity differs from current implementation';assert all(s['status']=='PASS' for s in suite['steps']);assert len(suite['steps'])==4
 code=identity(c)['code_hash'];validation_path=out/'smoke/notebooks'/code[:12]/'validation.json'
 assert validation_path.exists(),'Validate notebook defaults against the final code first'
 notebooks=json.loads(validation_path.read_text())
 assert len(notebooks)==10 and all(r['status']=='PASS' for r in notebooks)
 tests=Path('/tmp/week12_tests_final.log').read_text();assert tests.rstrip().endswith('OK');count=int(re.search(r'Ran (\d+) tests',tests).group(1));atomic(qual/'unit_tests.json',dict(passed=True,tests=count,command='sara_env/bin/python -m unittest discover -s tests/week12 -v'))
 immutable_notebooks=ROOT/'notebooks/week12';assert all('RUN_FULL = False' in __import__('nbformat').read(p,as_version=4).cells[1].source for p in immutable_notebooks.glob('*.ipynb'))
 assert not list((out/'development').glob('*/status.json')),'Unexpected full development status'
 assert not (out/'lora/complete.json').exists(),'Unexpected full training completion'
 llm=json.loads((qual/'llmlingua.json').read_text());assert llm['status']=='NOT_QUALIFIED'
 rows_path=directory(c,'B',False,True)/'rows'
 assert rows_path.exists(),'Run bounded real smoke against the final code first'
 rows=[json.loads(p.read_text())['row'] for p in rows_path.glob('*.json') if len(p.stem)==64];assert len(rows)==3
 smoke=[]
 for r in rows:smoke.append(dict(dataset=r['dataset'],query_id=r['query_id'],evidence_tokens=r['evidence_tokens'],budget=r['budget'],unused=r['unused_budget'],source_support=r['source_support'],prompt_tokens=r['prompt_tokens'],generation_cap_hit=r['generation_cap_reached'],format_compliant=r['format_compliant']))
 assert all(r['evidence_tokens']<=r['budget'] and r['source_support']==1 for r in rows)
 lora=json.loads((qual/'real_qlora_smoke.json').read_text());assert lora['status']=='PASS' and lora['optimizer_steps']==1 and not lora['adapter_saved']
 inv=inventory(c);pd.DataFrame(inv).to_csv(qual/'baseline_inventory.csv',index=False)
 cohort=json.loads((out/'protocol/cohorts.json').read_text());atomic(qual/'notebook_validation.json',notebooks)
 source_paths=sorted(p for d in ['src/week12','scripts/week12','configs/week12','notebooks/week12','tests/week12','operations/week12'] for p in (ROOT/d).glob('*') if p.is_file() and p.suffix not in ('.pyc',))
 files='\n'.join('- `'+str(p.relative_to(ROOT))+'`' for p in source_paths)
 impl_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 text=f'''# Week 12 implementation and smoke report

**The framework is ready for manual development runs. No full experiment, full training, confirmation evaluation or QASPER TEST evaluation was started.** Week13 confirmation remains blocked pending provenance certification and a final method freeze.

## Branch, files and preservation

Branch: `gagan`. Implementation checkout when this report was built: `{impl_commit}`. Runtime code hash: `{code}`. Model/source/package identities are recorded in qualification receipts and immutable result stores.

{files}

Historical byte check: **{len(before):,} tracked W0–W11 files unchanged**, including the user's pre-existing notebook outputs/flags. Historical frozen gates still pass. New caches/results are under Week12 only; historical SFR embeddings are reused through a read-only SQLite connection. No dependency was installed into the historical environment and no historical model cache was deleted.

## QBGF definition and agreed corrections

N=10, tau_p=tau_s=5, budgets 256/1024, pinned SFR and pinned Qwen3-8B. Passage softmax weights multiply query-ranked sentence relevance mass. DROP/SENT1/2/4/8/FULL form exact multiple-choice knapsack options. DP optimizes this relevance surrogate, **not true F1 or causal support**. Utility ties at1e-12 prefer fewer tokens, then lexicographic option indices. Oversized FULL is merely infeasible. Exact source sentences are selected by relevance and exposed in original source order. Overlapping candidate source regions fail before optimization; equivalent k options are deduplicated.

Costs include source separators in additive Qwen token-ID blocks. The reader consumes the exact assembled IDs; it never retokenizes the joined evidence and silently changes the budget. Each QBGF result checks sum(option costs)=actual evidence-ID count<=B. Prospective matched controls share this serialization. RAW_STRICT separately reproduces historical whole-text/max-three-tokenizer strict-prefix allocation on the W12 pool; RAW_SKIP uses skip-to-fit. Old W11 numeric scores are never reused as matched W12 outcomes.

W12-C is **development/validation**, not confirmation. Method selection occurs on development only, before a future Week13 locked evaluation. No Week12 entrypoint evaluates reserved confirmation examples.

## Baseline inventory

{pd.DataFrame(inv).fillna('').to_markdown(index=False)}

Canonical SARA executes original BM25 top-10/chunk-256, k5 full text plus five SFR vectors, trained Mistral checkpoint, historical BF16 prompt/interface and 64-token greedy answer cap. The original full-text RAG reference uses its own r16 adapted checkpoint and all ten textual passages. They remain specialist references, separate from public Qwen; their reader/checkpoint/prompt/cap assumptions do not justify a bare “QBGF beats SARA” statement. Canonical source/input/checkpoint hashes and execution smoke receipts are saved.

SARA_SFR_K5 and SARA_SFR_K6_HISTORICAL are explicitly unqualified until historical pool/prompt parity is established. They cannot be silently relabelled as original SARA. Natural SFR/BM25 Qwen references and budgeted controls have distinct names.

## True ablations

- NO_QUERY: uniform passage and sentence weights, source-order sentence choices, **within the same query-retrieved pool**. Tests verify score mutation does not affect these options. It tests allocation conditioning, not removal of retrieval conditioning.
- SENTENCE_ONLY: no FULL choice; PASSAGE_ONLY: DROP/FULL only.
- NO_MASS: uniform min(k,m)/m coverage, retaining query-ranked sentence selection. It does not collapse every non-DROP option to the same utility.
- GREEDY: deterministic feasible marginal utility/token upgrades; toy tests distinguish it from exact DP.
- BM25: secondary retrieval-plus-relevance sensitivity, not an isolated granularity ablation.

## Split audit and remaining provenance work

Development: 50 questions each from the explicit historical W11 cohort. This is reused development data, never fresh confirmation.

Reserved candidate confirmation counts: { {d:len(v) for d,v in cohort['confirmation'].items()} }. They exclude known pilot question hashes, and SciFact reservation excludes connected cited-paper components touched by pilot claims. **These reservations are not certified fresh:** W0–W10 historical exposure still requires exhaustive review. No fresh QASPER confirmation is claimed, and QASPER TEST stays permanently closed. The complete lists/hashes are in `protocol/cohorts.json`, `split_audit.json` and `W12_CONFIRMATION_FREEZE.json`; the concise human audit is `results/week12/W12_SPLIT_AUDIT.md`.

TRAIN readiness: { {d:v['available'] for d,v in cohort['training'].items()} }. HotpotQA/SciFact TRAIN must be prepared explicitly. Full F creates the actual filtered membership manifest, excludes development/reserved question hashes, excludes historical QASPER DEV papers, and uses connected SciFact cited-paper groups. Train/validation leakage is checked before model loading. Historical TRAIN reuse is allowed; historical validation examples are not treated as unseen data. Token length caps fail rather than silently truncating training targets.

## LoRA readiness and decomposition

The default F smoke trains a random miniature Qwen adapter for two CPU steps, checking frozen base weights and changed adapter weights. Separately the pinned **real Qwen3-8B NF4** completed one optimizer step on a synthetic string: adapter changed, frozen base-parameter samples stayed unchanged, and no adapter was saved. Peak allocated VRAM was approximately {lora['peak_vram_bytes']/2**30:.2f} GiB. This qualifies pipeline compatibility, not model quality or full-training resource feasibility.

Manual full training uses dataset-balanced sampling, r8/alpha16/dropout0.05, q/v projections, one epoch and no hyperparameter search. Optimizer checkpoints resume. Corpus preparation resumes per query/mode/budget shard. BASE_NF4 and LORA_NF4 use identical source evidence; G verifies evidence hashes against C and verifies the adapter checksum. The FP16 main surface is not used as the causal QLoRA baseline.

The mandatory real 2×2 is RAW_SKIP/QBGF × BASE_NF4/LORA_NF4. Analysis reports selection, RAW adaptation, QBGF adaptation and combined effects, plus the paired interaction `(LoRA_QBGF−Base_QBGF)−(LoRA_RAW−Base_RAW)` with bootstrap CI. Full training is still disabled and unexecuted.

## LLMLingua-2

Status: **{llm['status']}**. Reason: `{llm['reason']}`. The official package is absent, the model revision is not pinned, and downloads are disabled by default. No token-compression result was fabricated, no model substituted, and no full LLMLingua experiment ran. Resource inventory records disk/HF cache/RAM/VRAM/package versions and estimated checkpoint footprint before any attempted installation/download. The adapter follows the official Microsoft API: [package](https://github.com/microsoft/LLMLingua), [checkpoint](https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank).

To add it later, install/qualify separately, pin the checkpoint revision before outcomes, and use a new configured output version if the scientific configuration changes. Its own tokenizer target is not accepted as a Qwen budget guarantee; the adapter validates actual Qwen tokens and fails without silent truncation if its predefined retries cannot fit. Token deletion is not called exact contiguous-span grounding or native query conditioning.

## Tests and smoke evidence

- **{count} unit/integration tests passed.** Includes DP versus brute force, greedy distinction, grounding, source order, no-query score independence, strict/skip behavior, macro-F1 resampling, Holm, split guards, train/val leakage refusal before model load, source identity invalidation and the synthetic 2×2 schema.
- **10/10 notebooks executed successfully with their original safe defaults**; copies stay in ignored smoke outputs.
- Real model smoke uses **three unique development queries total**, reused across qualification stages. No confirmation query or TEST record was used. Repeated smoke checks after fixes are not independent research trials.

{pd.DataFrame(smoke).to_markdown(index=False)}

All three selections passed exact source validation and restored source ordering; budget checks use actual input IDs. Smoke answer scores are not a quality result and did not tune the algorithm. One sampled QASPER question has an existing annotation-mapping warning, preserved as such.

Canonical SARA and canonical RAG each completed one QASPER execution smoke on that same QASPER query. Real QLoRA qualification used synthetic data only. Resume verification reran the real QBGF smoke and confirmed completed result shards were byte-identical; no completed answers were regenerated.

## Metrics, statistics and figures

QA F1/EM remain separate from SciFact macro-F1/accuracy. SciFact trims whitespace and accepts exact normalized-adapter labels only; invalid explanations/case changes remain invalid. Macro-F1 is recomputed from predictions/labels in every cluster bootstrap. QASPER clusters by paper; SciFact uses connected cited-document components; Hotpot uses normalized-question clusters. Classification HELP/HURT uses correctness transitions, not per-query macro-F1.

The 39-member primary family is fixed before outcomes across dataset/budget/contrast. Paired bootstrap: 10,000, seed 42. Raw two-sided cluster label-swap randomization p-values state the exchangeability assumption. Holm includes reserved optional hypotheses conservatively at p=1 internally; missing tests remain pending and never acquire invented observed p-values. LoRA precision groups remain separate in Pareto comparisons. No mixed raw dataset score is produced.

Saved analysis includes paired primary tests/CI, per-class SciFact metrics, evidence mapping metrics, answer-type breakdowns, HELP/HURT, allocation, unused-budget/compression tables, Pareto and LoRA decomposition. Plot scaffolding covers the twelve requested views and a paired-CI panel, with PNG/PDF and underlying CSV. Missing LLMLingua or adapted outcomes are not fabricated. End-to-end latency remains null unless measured; no latency superiority is claimed.

## Manual notebook order and workload

See `notebooks/week12/README.md`. Order: 0 → A → B → C → D → optional E/F/G → H → I.

- **Run All defaults remain smoke-only**. For C/D/G change `RUN_FULL=True`; the next cell derives effective_smoke=False. Leave RESUME=True, OVERWRITE=False.
- E requires qualification before `RUN_LLMLINGUA_FULL=True`; download stays opt-in.
- F requires explicit TRAIN preparation before `RUN_FULL_LORA_TRAIN=True`. It displays actual dataset/mode/length/group counts before training.
- H/I: `RUN_SMOKE=False` reads real saved development results only. I needs explicit FINAL_METHOD/FINAL_REASON; there is no automatic winner selection or Week13 evaluation.

Estimated logical development rows: C 2,400 + 100 separate canonical QASPER references; D 2,100 (300 overlap C); optional E 300; optional G 1,200. These are not physical-call counts or independent questions. Training cap is 500 source questions per dataset before filtering; the actual corpus must be inspected in F. Full execution time is not estimated from tiny smoke timings.

## Go/no-go and remaining risks

**GO for manual base-Qwen development C/D after reviewing notebook0/B.** Actual counts, source validation, pinned model loading, canonical smoke, default execution and resume checks pass.

**Conditional:** F/G await official TRAIN files and full user-run training; E is unqualified; SFR-SARA variants await faithful historical adapters. **NO-GO for Week13 confirmation now:** certify historical exposure/group disjointness, review the final development comparison and freeze the final method first. Available disk is about 20 GiB at qualification, so full training/checkpoint storage still needs a separate capacity check. A natural full-text reference that exceeds context fails loudly; it is never silently truncated. A 50-query development pilot may not support the conservative noninferiority gates; retaining fixed sentences is an allowed outcome.

No full runs were started. No claim that QBGF, LoRA or fusion improves quality follows from smoke tests.
'''
 (ROOT/'results/week12/W12_IMPLEMENTATION_REPORT.md').write_text(text)
 atomic(qual/'readiness.json',dict(framework_ready_for_manual_development=True,unit_tests=count,notebooks_passed=10,real_unique_queries=3,canonical_sara='SMOKE_PASSED_BM25_K5_N10',canonical_rag='SMOKE_PASSED',llmlingua=llm['status'],full_runs_started=False,full_lora_training_started=False,confirmation_ready=False,QASPER_TEST_used=False,historical_files_unchanged=len(before),identity=identity(c)))
 print('Implementation report and readiness artifacts written.')
if __name__=='__main__':main()
