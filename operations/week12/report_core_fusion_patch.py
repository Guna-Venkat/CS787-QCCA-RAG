"""Metadata-only final CORE D addendum. Never regenerates splits or runs models."""
from pathlib import Path
import sys,json,re
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.week12.protocol import load,root,identity,sha,atomic,immutable,utc
from src.week12.splits import read_freeze
from src.week12.execution import methods,cost_plan
from src.week12.statistics import registry,fusion_registry

def main():
 c=load();f=read_freeze(c);out=root(c)/'qualification/core_d_fusion_patch';out.mkdir(parents=True,exist_ok=True)
 before=json.loads(Path('/tmp/w12_core_fusion_protected.json').read_text());changed=[p for p,h in before.items() if sha(p)!=h];assert not changed,changed
 tests=Path('/tmp/w12_core_fusion_tests.log').read_text();assert tests.rstrip().endswith('OK');n=int(re.search(r'Ran (\d+) tests',tests).group(1));v=json.loads((root(c)/'smoke/notebooks'/identity(c)['code_hash'][:12]/'validation.json').read_text());assert len(v)==10 and all(r['status']=='PASS' for r in v)
 cost=cost_plan(c,{d:len(p['VAL']) for d,p in f['datasets'].items()});assert sum(r['additional_call_upper_bound'] for r in cost if r['CORE_FINAL'])==133392
 assert not list((root(c)/'validation').glob('*/status.json')) and not (root(c)/'lora/complete.json').exists() and not (ROOT/'results/week13').exists()
 supplement=root(c)/'protocol/core_d_fusion_addendum.json'
 if not supplement.exists():immutable(supplement,dict(created_at=utc(),reason='User-requested pre-full-run fusion necessity controls',CORE_D=methods('D'),budgets=c['budgets'],original_primary_contrasts=39,fusion_diagnostic_contrasts=fusion_registry(c),joint_holm_family_size=51,original_registry_sha256=sha(root(c)/'protocol/primary_comparisons.json'),split_hash=f['split_hash'],identity=identity(c),no_full_outcomes_observed=True))
 atomic(out/'C_D_cost_audit.json',cost);atomic(out/'verification.json',dict(tests_passed=n,notebooks_passed=10,synthetic_D_rows=24,all_four_methods_both_budgets=True,protected_files_verified=len(before),changed_historical_artifacts=changed,identity=identity(c),full_runs_started=False))
 text=f'''# Final CORE D fusion-control patch

Branch gagan. W12-QBGF-v3 and frozen split hash `{f['split_hash']}` remain unchanged. No split was regenerated. QBGF, reader, retriever, prompts, decoding, loss-weighted full-coverage training, and confirmation IDs were not modified.

CORE_FINAL=True now runs these D methods at **both256 and1024**:

- QBGF_FULL
- QBGF_NO_QUERY
- QBGF_SENTENCE_ONLY
- QBGF_PASSAGE_ONLY

NO_MASS, GREEDY and BM25 remain implemented secondary opt-ins. C is unchanged. D holds the retrieved pool and generator fixed, separating query-conditioned allocation from the value of offering both sentence and passage representations. NO_QUERY still has the query-dependent SFR retrieved pool; it does not remove all query information from the pipeline.

H/I display twelve paired full-vs-sentence-only / full-vs-passage-only comparisons (three datasets, two budgets). Original39 primary comparisons and their frozen registry remain intact. A pre-run addendum records the12 fusion diagnostics; analysis also reports conservative joint Holm correction across all51 contrasts, retaining pending optional arms internally at p=1 rather than fabricating results. No comparison was selected using full VAL or confirmation outcomes.

## Updated cost bounds

| Phase | Logical rows | Additional answer-call upper bound |
|---|---:|---:|
| C core |91,707|83,370|
| D core |66,696|50,022|
| Combined |158,403|133,392|

D's16,674 QBGF_FULL rows reuse compatible C answers. The additional two fusion controls add at most33,348 calls over the preceding core plan. Identical/empty evidence may further reduce physical calls. These are bounds, not measured runtime estimates.

## Verification and remaining flags

**{n} tests pass;10/10 notebooks pass with safe smoke-only defaults.** The synthetic D smoke has24 rows: three fixture queries × four methods × two budgets. Tests verify both fusion controls appear and secondary arms remain absent. All{len(before):,} protected historical/config/QBGF/split artifacts are byte-identical to the pre-patch snapshot.

Keep monitoring LoRA gradient norms, clipping, nonfinite values and per-dataset VAL metrics. Stop and review instability rather than silently changing the objective mid-run. The qualified LLMLingua baseline is the exact smaller official multilingual-BERT checkpoint recorded in its runtime lock, not a reproduction of published XLM-R-large results. Confirmation remains HotpotQA200 + SciFact50; SciFact intervals may be wide. QASPER has no fresh confirmation and supports no fresh-generalization claim.

## Proceed manually

You can proceed with C then D: leave CORE_FINAL=True, RESUME=True and OVERWRITE=False; enable RUN_FULL=True yourself in the relevant notebook. H/I show the new fusion comparison table. No full validation, LoRA training, confirmation or QASPER TEST was started by this patch. Do not change split IDs/retrieval or tune against confirmation outcomes. Full all-TRAIN storage/compute capacity remains a separate operational check.
'''
 (ROOT/'results/week12/W12_CORE_D_FUSION_PATCH.md').write_text(text)
 print('Final CORE D addendum verified; no full runs started.')
if __name__=='__main__':main()
