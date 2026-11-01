# Final CORE D fusion-control patch

Branch gagan. W12-QBGF-v3 and frozen split hash `78f5ace11f9080d05dda12716e634020711182f721ac562dc81d19bad6c1c117` remain unchanged. No split was regenerated. QBGF, reader, retriever, prompts, decoding, loss-weighted full-coverage training, and confirmation IDs were not modified.

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

**122 tests pass;10/10 notebooks pass with safe smoke-only defaults.** The synthetic D smoke has24 rows: three fixture queries × four methods × two budgets. Tests verify both fusion controls appear and secondary arms remain absent. All2,730 protected historical/config/QBGF/split artifacts are byte-identical to the pre-patch snapshot.

Keep monitoring LoRA gradient norms, clipping, nonfinite values and per-dataset VAL metrics. Stop and review instability rather than silently changing the objective mid-run. The qualified LLMLingua baseline is the exact smaller official multilingual-BERT checkpoint recorded in its runtime lock, not a reproduction of published XLM-R-large results. Confirmation remains HotpotQA200 + SciFact50; SciFact intervals may be wide. QASPER has no fresh confirmation and supports no fresh-generalization claim.

## Proceed manually

You can proceed with C then D: leave CORE_FINAL=True, RESUME=True and OVERWRITE=False; enable RUN_FULL=True yourself in the relevant notebook. H/I show the new fusion comparison table. No full validation, LoRA training, confirmation or QASPER TEST was started by this patch. Do not change split IDs/retrieval or tune against confirmation outcomes. Full all-TRAIN storage/compute capacity remains a separate operational check.
