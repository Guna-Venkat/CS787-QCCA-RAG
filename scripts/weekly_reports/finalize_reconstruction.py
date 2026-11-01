"""Offline report/source verification and machine-readable project timeline; no narrative sources."""
from reportlib import *
import numpy as np

def primary_correction():
 # Explicit primary-only correction: human summary is unsupported by blank raw ratings.
 w=BASE/'week_03';f=obj('results/weekly_reports/week_03/ReportFreeze.json');raw=frame('results/week3/human_audit/human_scoring_template.csv');assert raw.score_0_1_2.notna().sum()==0
 sentence='The gate explicitly records PENDING_MANUAL_EVALUATION. A scoring template/package exists; completed human judgment evidence is not established.'
 corrected='SOURCE CONFLICT: human_audit_summary.csv reports 50 judgments per k, but all 150 score_0_1_2 entries in human_scoring_template.csv are blank and week3_gate.json says PENDING_MANUAL_EVALUATION. The summary cannot be certified as observed human evaluation; completed human judgment evidence is not established.'
 old=dict(f)
 changed=False
 for n in f['reports']:
  p=w/n;text=p.read_text()
  if sentence in text:p.write_text(text.replace(sentence,corrected));changed=True
 if changed:
  (w/'PrePrimaryCorrectionFreeze.json').write_text(json.dumps(old,indent=2)+'\n')
  f['reports']={n:sha(w/n) for n in f['reports']};f['primary_only_revision']=dict(reason='Blank raw human ratings conflict with numerical human summary; no legacy narrative informed this correction',raw_nonnull_ratings=0,raw_rows=150,legacy_markdown_influenced_revision=False)
  (w/'ReportFreeze.json').write_text(json.dumps(f,indent=2)+'\n')

def main():
 primary_correction();timeline=[];legacy=[];inventory=[];project=[];weeks=[];counts={}
 for w in range(14):
  out=BASE/f'week_{w:02d}';f=json.loads((out/'ReportFreeze.json').read_text());assert len(list(out.glob('*.md')))==2
  for n,h in f['reports'].items():
   assert sha(out/n)==h,'Report hash changed: '+str(out/n)
   text=(out/n).read_text();assert '# Existing Markdown Used as Evidence\n\nNONE' in text
   # Verify relative file/figure links; ignore prose/code arrows and fragments.
   for target in re.findall(r'\]\(([^)]+)\)',text):
    if target.startswith(('http:','https:','#')):continue
    assert (out/target).exists(),(out/n,target)
  for path,h in f['sources'].items():assert sha(ROOT/path)==h,'Primary artifact changed: '+path
  inv=frame(str((out/'SourceInventory.csv').relative_to(ROOT))) if (out/'SourceInventory.csv').stat().st_size>1 else pd.DataFrame();inventory+=inv.to_dict('records') if len(inv) else []
  ti=out/'TimelineRows.csv'
  if ti.exists():timeline+=pd.read_csv(ti).to_dict('records')
  else:
   # Original Week0 timeline was persisted before this multi-week continuation.
   first=pd.read_csv(BASE/'ExperimentTimeline.csv');timeline+=first[first.week==w].to_dict('records')
  le=out/'LegacyMarkdownAudit.csv'
  if le.exists():
   ld=pd.read_csv(le);legacy+=ld.to_dict('records')
  weeks.append(dict(week=w,parts=f['parts'],primary_artifacts=len(f['sources']),numerical_tables=f.get('numerical_tables',10),figures=f.get('figures',f.get('detailed_figures',0)),partwise_path=str((out/'WeeklyPartWiseAnalysis.md').relative_to(ROOT)),summary_path=str((out/'WeeklySummaryAnalysis.md').relative_to(ROOT)),status='FROZEN_WITH_GAPS' if w in [0,1,3,8,13] else 'FROZEN',legacy_markdown_used_as_evidence=False))
 # Project-wide conflicts and semantic consistency, sourced to explicit files/code rather than old narrative.
 def row(scope,dimension,status,issue,evidence,action):project.append(dict(scope=scope,dimension=dimension,status=status,issue=issue,authoritative_evidence=evidence,action=action))
 row('W0','simulation provenance','SOURCE CONFLICT','Generic layer/attention/ablation CSV values exactly match deterministic mock formulas; separate real pilot/final40 exist.','results/weekly_reports/week_00/PrimaryRecomputationChecks.json; results/week0/final40/execution_metadata.json','Quarantine mock paths; do not merge real and simulated cohorts.')
 row('W1','chronology','RESOLVED USER SCOPE','User identifies Baselines/ as Week1; SARA/RAG checkpoint states and executed notebook logs establish baseline training/qualification.','results/weekly_reports/week_01/SourceInventory.csv; Baselines/SARA-main/notebooks/sara_baseline_checkpoints.ipynb','Use corrected Week1 reports; no standalone quality scores are borrowed from Week2.')
 row('W1','training coverage and epoch identity','SOURCE CONFLICT','Both checkpoints record250steps; notebook RAG log says506 examples while localTRAIN has2088 and max_train_samples=2000. Current YAML epochs3 versus saved RAG2/SARA1.','results/weekly_reports/week_01/TrainingRecomputation.csv; results/weekly_reports/week_01/NotebookExecutionChecks.json; Baselines/SARA-main/checkpoints/finetune/rag_qasper_lora_r16_seed42/args.json','Preserve exact execution identities; matched optimizer counts do not establish matched data coverage or quality.')
 row('W2/W3','action space','RESOLVED DEFINITION DIFFERENCE','W2 SARA oracle includes k0; W3 primary oracle excludes k0 and k10. Different ceilings are expected.','configs/week2/sweep_config.yaml; configs/week3/week3_config.yaml; results/week3/processed/dev_quality_oracle.csv','Preserve action sets and retrospective-oracle terminology.')
 row('W3','human evaluation','SOURCE CONFLICT','Numerical summary claims ratings while all150 raw ratings are blank and gate says pending.','results/week3/human_audit/human_scoring_template.csv; results/week3/human_audit/human_audit_summary.csv; results/week3/processed/week3_gate.json','Human evaluation unestablished; manual provenance review required.')
 row('W4','bootstrap contrast','SOURCE CONFLICT','Stored V2 difference-vs-k5 point0.3038 equals policy score, not difference from0.3629.','results/week4/qcca_v2/bootstrap_policy_comparison.csv; results/week2/processed/dev_fixed_k_summary.csv','Do not accept this row as a valid paired contrast.')
 row('W2/W5','token costs','SOURCE CONFLICT','k8 reference1758.5 vs1750.5; source constants differ by8 tokens.','src/week5/evaluation.py; src/week5/supervision.py; results/week2/processed/dev_fixed_k_summary.csv','Retain artifact-specific accounting; no pooled cost reduction.')
 row('W5','oracle costs','SOURCE CONFLICT','Quality-oracle token cost846.1 vs1021.5 across content A/B/C tables despite equal printed F1/k.','results/week5/content_allocator/tables/main_results.csv; results/week5/content_allocator/w5b/tables/w5b_main_results.csv','Manual review of cost provenance; no silent harmonization.')
 row('W6/W7','baseline drift','SOURCE CONFLICT','Semantic k4 imported point0.391333 vs later semantic-surface0.390929.','results/week6/passage_routing/tables/w6_main_results.csv; results/week7/w7b_semantic_budget/tables/w7b_main_dev_results.csv','Retain per-part source identities; not six-decimal rounding.')
 row('W8','uncertainty and completion','EVIDENCE GAP','Historical B output contains degenerate token features; G/I no full outcomes found; H/J only smoke-labelled final artifacts.','notebooks/week8/week8b_draft_diagnostics.ipynb; results/week8/w8e_controller/controller_summary.json; results/week8/tables/w8j_bootstrap_dafir_vs_k4_smoke.csv','Do not upgrade smoke/scaffold to completed full evaluation; distinguish later gate backfills.')
 row('W9','TEST exposure','CONFIRMED','Historical QASPER TEST1309 was consumed in final evaluation.','results/week9/w9h_final/v2/report/summary_quality.csv','Permanently close QASPER TEST; no fresh confirmation claim.')
 row('W10','reader adaptation/interface','WHOLE-SYSTEM CONFOUND','C fine-tuned Mistral and G public Mistral have different adaptation/prompt interfaces.','results/week10/root_cause/v1/summary.csv; configs/week10/root_cause_v1.json','Do not attribute difference solely to reader architecture or latent representation.')
 row('W10/W11','gold mapping','METRIC-SCOPE CAVEAT','Later evidence coverage uses different exact-span/corpus mappings than early lexical paragraph overlap.','results/week0/dataset_analysis/retrieval_statistics.csv; results/week10/root_cause/v1/retrieval_summary.csv; results/week11/multigranular/v1/setup/dataset_summary_df.csv','Do not interpret cross-week absolute recall changes as matched causal comparisons.')
 row('W11','sample/corpus','CONFIRMED','50 queries per dataset; HotpotQA provided distractor contexts and SciFact full abstract corpus.','results/week11/multigranular/v1/setup/dataset_summary_df.csv','Pilot scope; no full-corpus HotpotQA claim.')
 row('W11','SciFact compliance','WHOLE-SYSTEM CONFOUND','Mistral/Llama low task scores are dominated by output-format violations.','results/week11/multigranular/v1/reader_granularity/label_compliance_diagnostic_summary.csv','Do not compare reader reasoning cleanly from these primary scores.')
 row('W11','allocator','DESIGN CONFOUND','Strict-prefix whole-passage allocation can fail at256; C2 top8 saturates with larger budgets.','src/week11/budgets.py; configs/week11/week11_compressors.yaml; results/week11/multigranular/v1/compressor_reader/audited_summary.csv','Compression-quality gains do not establish general superiority over fairer passage allocators.')
 row('W11/W12','LLMLingua','RESOLVED CHECKPOINT/STAGE DIFFERENCE','W11 XLM-R-large pending; W12 smaller official mBERT qualified but E benchmark skipped.','configs/week11/week11_compressors.yaml; results/week12/qbgf/v3/protocol/llmlingua_qualification_lock.json; results/week12/qbgf/v3/protocol/optional_phase_decisions.json','Qualification is not quality evaluation; never imply published XLM-R-large reproduction.')
 row('W12 C-EXT','aliases','RESOLVED TERMINOLOGY','BM25_PLAIN_RAG and SFR_PLAIN_RAG mean matched RAW_SKIP, not unrestricted RAG.','results/week12/qbgf/v3/c_extension/v1/analysis/merged_validation_summary.csv; src/week12_extension/baselines.py','Keep named BM25/SFR/checkpoint/interface controls separate.')
 row('W12 D','mechanism','SCOPE LIMIT','NO_QUERY keeps query-dependent SFR pool; intervention removes allocation conditioning only.','src/week12/algorithm.py; results/week12/qbgf/v3/analysis/summary.csv','Claim query-conditioned allocation benefit on VAL only, not necessity of all query information.')
 row('W12 E/F/G','execution','SKIPPED','No full LLMLingua evaluation or final LoRA training/validation; user decision recorded.','results/week12/qbgf/v3/protocol/optional_phase_decisions.json','Do not infer full TRAIN optimizer contributions or scores from scheduler/tiny-smoke code.')
 row('W12 C/D','raw replication','VERIFIED','All51 condition quality means reproduce from141729 unique rows;16674 duplicate rows agree.','results/weekly_reports/week_12/OfflineValidationRecomputation.json; results/weekly_reports/week_12/OfflineValidationRecomputation.csv','Maintain source files unchanged.')
 row('W12 L','retention','SOURCE CONFLICT / ADDITIVE CORRECTION','Original summary propagates undefined NaN ratios; finite-only table has6–34 eligible queries/curve.','results/week12/attribution/v1/analysis/curve_summary.csv; results/week12/attribution/v1/analysis/finite_retention_summary.csv','Use finite counts, preserve original artifacts; zero deletion effect does not imply ignored evidence.')
 row('W12 M','confirmation','VERIFIED / CONSUMED','250 queries,2500 rows,2213 distinct generations; HotpotQA200groups, SciFact50claims/43groups, QASPER0.','results/week12/final_confirmation/v1/complete.json; results/week12/W13_FINAL_METHOD_FREEZE.json; results/week12/final_confirmation/v1/summary.csv','This cohort is no longer a fresh holdout and must not be used for further tuning; no policy promotion after outcomes.')
 row('W13','naming','HISTORICAL ALIAS','Historical wrapper/freeze names do not establish a separate executed Week13 experiment.','scripts/week13/run_confirmation.py; results/week12/final_confirmation/v1/complete.json','Count actual final confirmation once, as Week12 M.')
 pd.DataFrame(timeline).to_csv(BASE/'ExperimentTimeline.csv',index=False);pd.DataFrame(legacy).to_csv(BASE/'LegacyMarkdownAudit.csv',index=False);pd.DataFrame(project).to_csv(BASE/'ProjectWideConsistencyAudit.csv',index=False);pd.DataFrame(weeks).to_csv(BASE/'ReconstructionStatus.csv',index=False);pd.DataFrame(inventory).to_csv(BASE/'PrimaryArtifactInventory.csv',index=False)
 # No additional prose Markdown: the top-level deliverables are machine-readable.
 stats=dict(weeks_documented=len(weeks),active_experiment_weeks=13,gap_week=None,historical_alias_week=13,parts=len(timeline),unique_primary_artifacts=len({r['artifact'] for r in inventory}),primary_artifact_entries=len(inventory),raw_week12_rows_inspected=158403,legacy_markdown_audited=len(legacy),legacy_corrected=sum(str(r.get('corrected')).lower()=='true' for r in legacy),legacy_manual_review=sum('manual review' in str(r.get('action','')).lower() for r in legacy),figures=sum(x['figures'] for x in weeks),project_audit_rows=len(project),all_report_source_hashes_verified=True,primary_evidence_modified=False)
 (BASE/'ReconstructionManifest.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
