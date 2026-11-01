"""Separate legacy phase. Frozen reconstructed Markdown is never edited here."""
from pathlib import Path
import hashlib,json,re,csv,os
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/weekly_reports/week_00'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 freeze=json.loads((OUT/'ReportFreeze.json').read_text());assert all(sha(OUT/n)==h for n,h in freeze['reports'].items())
 report=(OUT/'WeeklyPartWiseAnalysis.md').read_text()
 plan={
 'dataset_analysis/dataset_analysis_report.md':('INCORRECT','Calls the local 2,088/231 subsets official splits and an exact 90/10 question partition.','A','Reconstruct generated report from frozen Part A; preserve primary data.'),
 'layer_analysis/layer_analysis_report.md':('CONSISTENT','Correctly labels generic layer results synthetic/invalid; no empirical endorsement taken from it.',None,'Leave unchanged; methodological scope only.'),
 'evidence_analysis/evidence_analysis_report.md':('CONSISTENT','Correctly labels generic attention/ablation simulation invalid.',None,'Leave unchanged; methodological scope only.'),
 'gpu_pilot/pilot_report.md':('INCORRECT','Later paragraphs claim execution under/about 20 seconds, contradicting its own header and execution_metadata.json runtime 791.17 seconds.','GPU-PILOT','Reconstruct generated pilot report; eliminate contradictory runtime narrative.'),
 'gpu_pilot/intervention_validation/validation_report.md':('FABRICATED / UNSUPPORTED','Claims replacement signs prove positive semantic contribution and confirm genuine distraction; input perturbations do not identify internal reasoning.','INTERVENTION-VALIDATION','Reconstruct generated validation report with descriptive sensitivity and control limitations.'),
 'final40/final40_report.md':('INCORRECT','Claims any/all recall@10=84.8%/64.9% versus primary full-dev 87.0130%/80.0866%; observed effect gap is 0.597889, not bootstrap-mean 0.6029; says median only 1–2 positive passages versus features median 7; p<0.001 is not backed by a saved formal test.','FINAL40','Reconstruct generated final40 report from frozen primary-source section.'),
 'repository_audit.md':('UNKNOWN','Mixes repository/setup descriptions with claims beyond the frozen Week0 numerical audit; not certified wholesale.',None,'Leave unchanged; manual review required for setup/model/split identity claims.'),
 'scientific_audit.md':('PARTIALLY STALE','Mock quarantine is consistent, but pending-GPU recommendations predate separate real pilot/final40 execution; zero-impact wording includes ignored, which is not established.',None,'Preserve historical audit; manual review required; do not use as experimental evidence.'),
 'sara_motivation/week0_synthesis.md':('UNKNOWN','Architecture/theory and allocation-motivation narrative mix Markdown-derived figures and downstream plans; not independently certified by this phase.',None,'Leave unchanged; manual review required; do not use as experimental evidence.'),
 'week0_report.md':('PARTIALLY STALE','Uses official-split terminology for local reformatted cohort; generic simulation is quarantined, but later separate final40 execution is not an interchangeable update to generic pending status.',None,'Preserve historical report; manual review required.')}
 rows=[]
 for rel,(status,claim,part,action) in plan.items():
  p=ROOT/'results/week0'/rel;old=p.read_text();before=sha(p) # legacy content read only AFTER report freeze
  if part:
   marker='# Part '+part+' — ';start=report.index(marker);end=report.find('\n# Part ',start+1)
   if end<0:end=report.find('\n# Week 0 — Integrated Analysis',start)
   section=report[start:end]
   def rewrite(m):
    target=(OUT/m.group(2)).resolve();return m.group(1)+os.path.relpath(target,p.parent)+m.group(3)
   section=re.sub(r'(\]\()([^\)]+)(\))',rewrite,section)
   header='This file was reconstructed from primary experimental artifacts. Existing narrative was not used as evidence.\n\n'
   p.write_text(header+'Generated legacy report corrected after the two authoritative Week0 reports were frozen. Primary artifacts were not changed.\n\n'+section+'\n\nAuthoritative reports: '+os.path.relpath(OUT/'WeeklyPartWiseAnalysis.md',p.parent)+' and '+os.path.relpath(OUT/'WeeklySummaryAnalysis.md',p.parent)+'.\n')
  rows.append(dict(file=str(p.relative_to(ROOT)),status=status,incorrect_claim=claim,authoritative_evidence='results/weekly_reports/week_00/ReportFreeze.json; associated primary sources in SourceInventory.csv',action=action,sha_before=before,sha_after=sha(p),corrected=bool(part)))
 for path in [OUT/'LegacyMarkdownAudit.csv',ROOT/'results/weekly_reports/LegacyMarkdownAudit.csv']:
  with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 assert all(sha(OUT/n)==h for n,h in freeze['reports'].items())
 print(json.dumps(dict(files_audited=len(rows),corrected=sum(r['corrected'] for r in rows),manual_review=sum('manual review' in r['action'].lower() for r in rows),authoritative_reports_unchanged=True),indent=2))
if __name__=='__main__':main()
