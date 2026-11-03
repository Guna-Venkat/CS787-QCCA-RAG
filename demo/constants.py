from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('QASPER', 'HotpotQA', 'SciFact')
METHODS = ('RAW_SKIP', 'FIXED_SENTENCE', 'H3', 'QBGF_FULL', 'NONE')
BUDGETS = (256, 1024)
COLORS = {'RAW_SKIP':'#f59e0b','FIXED_SENTENCE':'#22c3ad','H3':'#818cf8','QBGF_FULL':'#38bdf8','NONE':'#94a3b8','BM25':'#f59e0b','MiniLM':'#818cf8','SFR':'#22c3ad','Mistral':'#f59e0b','Qwen':'#22c3ad','Llama':'#818cf8',
          'HELPFUL':'#34d399','NEUTRAL':'#94a3b8','HARMFUL':'#fb7185'}
LABELS = {'RAW_SKIP':'Passage · skip to fit','FIXED_SENTENCE':'Sentence · query selected','H3':'Hierarchy · fixed rule',
          'QBGF_FULL':'Adaptive · QBGF','NONE':'No evidence','BM25_PLAIN_RAG':'BM25 passage · RAW_SKIP','C2':'Top-eight sentence control'}
EXPLANATIONS = {
 'RAW_SKIP':'Whole passages in retrieval order. Include those that fit; skip oversized passages and continue. Budget matched, not unrestricted full-context RAG.',
 'FIXED_SENTENCE':'Rank exact source sentences by query similarity. Include fitting sentences, skip non-fitting ones, then restore source order. A fixed policy, not a fixed sentence count.',
 'H3':'Query-ranked sentences from the first three passages, followed by query-ranked deterministic fine source spans from the remaining passages. Uses the registered prefix allocator.',
 'QBGF_FULL':'Choose DROP, SENT_1, SENT_2, SENT_4, SENT_8 or FULL for each retrieved passage, optimizing question-conditioned relevance utility under one global token budget. It was not promoted as a universal winner.',
 'NONE':'Question and instructions without external evidence. Still one conceptual reader call.',
 'BM25_PLAIN_RAG':'Saved internal alias: budget-matched BM25 RAW_SKIP passage RAG. This uses lexical retrieval; do not call it the SFR matched control.',
 'C2':'Top-eight query-ranked exact sentences with the historical prefix allocator. This is a validation control, not the final primary policy.'}
SOURCES = {
 'retrieval_pilot':'results/week11/multigranular/v1/retrieval/retrieval_summary_df.csv',
 'retrieval_manifest':'results/week11/multigranular/v1/retrieval/manifest.json',
 'reader_pilot':'results/week11/multigranular/v1/reader_granularity/audited_summary.csv',
 'reader_manifest':'results/week11/multigranular/v1/reader_granularity/manifest.json',
 'reader_pairs':'results/week11/multigranular/v1/reader_granularity/named_reader_comparisons.csv',
 'reader_interactions':'results/week11/multigranular/v1/reader_granularity/named_reader_granularity_interactions.csv',
 'reader_compliance':'results/week11/multigranular/v1/reader_granularity/label_compliance_diagnostic_summary.csv',
 'reader_panel':'results/week10/root_cause/v1/full_reader_panel_summary.csv',
 'reader_panel_ci':'results/week10/root_cause/v1/full_reader_bootstrap.csv',
 'w10_protocol':'results/week10/root_cause/v1/protocol.json',
 'extension':'results/week12/qbgf/v3/c_extension/v1/analysis/merged_validation_summary.csv',
 'specialists':'results/week12/qbgf/v3/c_extension/v1/analysis/specialist_references.csv',
 'gold_overlap':'results/week12/attribution/v1/analysis/gold_overlap.csv',
 'retention':'results/week12/attribution/v1/analysis/finite_retention_summary.csv',
 'w10_runner':'src/week10/root_cause/runner.py',
 'w11_runner':'src/week11/runner.py',
 'w10':'results/week10/root_cause/v1/retrieval_summary.csv',
 'w11':'results/week11/multigranular/v1/granularity_budget/c_summary_df.csv',
 'validation':'results/week12/qbgf/v3/analysis/summary.csv',
 'primary':'results/week12/qbgf/v3/analysis/primary_comparisons.csv',
 'fusion':'results/week12/qbgf/v3/analysis/fusion_diagnostic_comparisons.csv',
 'confirmation':'results/week12/final_confirmation/v1/summary.csv',
 'contrasts':'results/week12/final_confirmation/v1/contrasts.csv',
 'consistency':'results/week12/final_confirmation/v1/validation_comparison.csv',
 'units':'results/week12/attribution/v1/analysis/unit_importance.csv',
 'profiles':'results/week12/attribution/v1/analysis/query_profiles.csv',
 'correlations':'results/week12/attribution/v1/analysis/relevance_utility.csv',
 'qualitative':'results/week12/attribution/v1/analysis/qualitative_candidates.json',
 'freeze':'results/week12/W13_FINAL_METHOD_FREEZE.json',
 'freeze_copy':'results/week12/qbgf/v3/W13_FINAL_METHOD_FREEZE.json',
 'split':'results/week12/qbgf/v3/protocol/W12_SPLIT_FREEZE.json',
 'code':'src/week12/algorithm.py',
}
FREEZE_SHA = 'd0188a3582d6aead0a3a31f9f15bad9b51a88a6b475b05ac79ae38e58f05366b'
