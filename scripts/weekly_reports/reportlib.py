"""Primary-artifact report helpers. Legacy audit runs only after both report hashes freeze."""
from pathlib import Path
import json,hashlib,csv,os,re,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/weekly_reports'
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def frame(path):return pd.read_csv(ROOT/path)
def obj(path):return json.loads((ROOT/path).read_text())
def flat(x,prefix=''):
 rows=[]
 if isinstance(x,dict):
  for k,v in x.items():
   key=prefix+'.'+str(k) if prefix else str(k)
   if isinstance(v,(dict,list)):rows+=flat(v,key)
   elif isinstance(v,(int,float,bool)) or v is None:rows.append({'field':key,'value':v})
 elif isinstance(x,list):
  for i,v in enumerate(x):rows+=flat(v,prefix+f'[{i}]') if isinstance(v,(dict,list)) else ([{'field':prefix+f'[{i}]','value':v}] if isinstance(v,(int,float,bool)) else [])
 return rows

def part(id,name,question,paths=(),tables=(),finding='',caveat='',status='COMPLETED',design=None,stats=None,figures=(),executed=None,decision=''):
 return dict(id=id,name=name,question=question,paths=list(paths),tables=list(tables),finding=finding,caveat=caveat,status=status,design=design or {},stats=stats or 'No formal inferential test was found for this comparison.',figures=list(figures),executed=executed,decision=decision)

def write_week(w,objective,parts,context=(),checks=(),transition='',decision='',negative='',limits='',extra_sources=(),revision_reason=None):
 out=BASE/f'week_{w:02d}';out.mkdir(parents=True,exist_ok=True);sources=set(context)|set(extra_sources);inventory=[];figcount=0;tabcount=0
 if revision_reason and (out/'ReportFreeze.json').exists() and not (out/'PriorScopeRevision.json').exists():
  prior=dict(reason=revision_reason,freeze=json.loads((out/'ReportFreeze.json').read_text()),reports={p.name:p.read_text() for p in out.glob('*.md')});(out/'PriorScopeRevision.json').write_text(json.dumps(prior,indent=2)+'\n')
 def link(p):return '['+p+']('+os.path.relpath(ROOT/p,out)+')'
 def table(d):return d.to_markdown(index=False,floatfmt='.6f') if len(d) else 'No completed result artifact was found.'
 detail=[f'# Week {w} — Part-Wise Analysis','','## Week objective','',objective,'']
 summaryrows=[];keytables=[]
 for p in parts:
  allpaths=list(dict.fromkeys(p['paths']+[x[0] for x in p['tables']]+[x[0] for x in p['figures']]))
  sources.update(allpaths);design=dict(p['design'])
  design.setdefault('Role','Protocol/context only; outcomes are accepted only from the artifacts below')
  detail += [f"# Part {p['id']} — {p['name']}",'',f"**Status: {p['status']}**",'','## 1. Research question','',p['question'],'','## 2. Experimental design','',table(pd.DataFrame([dict(component=k,value=str(v)) for k,v in design.items()])),'','## 3. What was implemented / executed','',p['executed'] or ('Completed machine-readable output is present for the named conditions; coverage is limited to the recorded rows and checks below.' if p['status']=='COMPLETED' else 'The status is based on the available artifacts, not notebook existence alone. '+('No completed result artifact was found.' if not p['tables'] else 'Available outputs must be interpreted under the status shown.')),'','## 4. Result artifacts used','']
  detail += ['- '+link(z) for z in allpaths] or ['Not established from available primary artifacts.']
  detail+=['','## 5. Main numerical results','']
  if not p['tables']:detail+=['No completed result artifact was found.','']
  for path,df,caption in p['tables']:
   detail+=['### '+caption,'',table(df),'','Source: '+link(path)+'.',''];tabcount+=1
   # Independently construct concise tables from the same in-memory primary evidence, not report prose.
   if len(df):keytables.append((path,df.head(12).copy(),caption,len(df)))
  detail+=['## 6. Figures','']
  if not p['figures']:detail+=['No saved figure is used for this part; no visualization is fabricated.','']
  for path,caption in p['figures']:
   detail+=['!['+p['name']+']('+os.path.relpath(ROOT/path,out)+')','',caption,''];figcount+=1
  detail+=['## 7. Statistical analysis','',p['stats'],'','## 8. Interpretation','','### Supported','',p['finding'] or 'Not established from available primary artifacts.','','### Not established','','Point-estimate rankings do not establish statistical superiority. Implementation, qualification and retrospective oracle ceilings are not deployed or held-out quality results. Missing model/split/seed details are not established from available primary artifacts.','','### Important caveats','',p['caveat'] or limits or 'Scope is restricted to the recorded sample, interfaces and evaluation metric.','','## 9. Part conclusion','', '- '+(p['finding'] or 'No completed experimental outcome is established.'),'- '+(p['decision'] or 'No additional method-selection decision is inferred beyond the recorded protocol/freeze artifacts.'),'']
  summaryrows.append(dict(Part=p['id'],Question=p['question'],Status=p['status'],Main_result=p['finding'] or 'Not established from available primary artifacts.'))
 # notebook evidence: code/output cells only, never notebook Markdown narratives
 notebooknotes=[]
 for path in list(sources):
  f=ROOT/path
  if f.suffix=='.ipynb':
   n=json.loads(f.read_text());cells=[(i,c) for i,c in enumerate(n.get('cells',[])) if c.get('cell_type')=='code' and c.get('execution_count') is not None and c.get('outputs')]
   notebooknotes.append(dict(artifact=path,executed_output_cells=len(cells),cell_indices=','.join(str(i) for i,c in cells)))
   for i,c in cells:
    for o in c['outputs']:
     # Access executed display/logs only; never count Markdown-cell prose as outcomes.
     _=o.get('text') or o.get('data',{}).get('text/plain') or o.get('data',{}).get('text/html')
 detail += [f'# Week {w} — Integrated Analysis','','## What changed during this week?','',objective+' The parts above retain the distinction between executed outputs, implementation-only work, exploratory diagnostics and retrospective choices.','','## Main supported findings','',table(pd.DataFrame([dict(Finding=p['finding'] or 'Unestablished',Evidence='Part '+p['id']+'; '+'; '.join(p['paths'][:2]),Strength='descriptive; see stated inferential evidence' if p['status']=='COMPLETED' else p['status']) for p in parts])),'','## Negative / null findings','',negative or 'No positive claim is inferred for incomplete or missing conditions. Consult per-part caveats and confidence intervals.','','## What remained unresolved?','',limits or 'Generalization beyond the recorded cohort and untested interventions is not established.','','## What logically motivated the next week?','',transition or 'An explicit transition is not established from available primary artifacts.','','## Offline reproducibility / source conflicts','']+(['- '+c for c in checks] or ['No independent raw-result recomputation was available; derived tables are identified as such.'])
 if notebooknotes:detail+=['','Executed notebook coverage (zero-based cell indices):','',table(pd.DataFrame(notebooknotes))]
 short=[f'# Week {w} — Summary Analysis','','## Objective','',objective,'','## Experiments performed','',table(pd.DataFrame(summaryrows)),'','## Key results','']
 # concise high-level output: retain all parts via one primary table per part, capped to 12 explicitly labelled rows.
 for path,df,caption,total in keytables[:max(1,len(parts))]:
  short+=['### '+caption,'',table(df),'','Source: '+link(path)+(f'. First {len(df)} of {total} rows shown; detailed report and source preserve the full comparison.' if total>len(df) else '.'),'']
 short+=['## Main scientific conclusions','']+['- '+(p['finding'] or 'Not established from available primary artifacts.')+' Source: '+(', '.join(link(z) for z in p['paths'][:2]) or 'No qualifying primary artifact.') for p in parts]+['','## What did NOT work?','',negative or 'No experimental success is established for incomplete conditions.','','## Limitations','',limits or 'Only the artifact-defined cohort/protocol is established. No cross-dataset mixed-metric average is computed.','','## Decision after this week','',decision or 'Not established from available primary artifacts.','','## One-paragraph week takeaway','', ' '.join(p['finding'] for p in parts if p['finding']) or 'No completed result artifact was found. The missing evidence is explicitly preserved instead of reconstructed from narrative.']
 for s in [detail,short]:s+=['','# Primary Sources Used','']+['- '+link(z) for z in sorted(sources)]+['','# Existing Markdown Used as Evidence','','NONE','']
 names=['WeeklyPartWiseAnalysis.md','WeeklySummaryAnalysis.md']
 for name,lines in zip(names,[detail,short]):
  p=out/name;content='\n'.join(lines)
  if p.exists() and p.read_text()!=content and not revision_reason:raise ValueError('Frozen report differs: '+str(p))
  p.write_text(content)
 for path in sorted(sources):
  p=ROOT/path
  if not p.exists():raise FileNotFoundError(path)
  inventory.append(dict(week=w,part=';'.join(x['id'] for x in parts if path in x['paths'] or path in [t[0] for t in x['tables']]) or 'protocol/cross-check',artifact=path,type=p.suffix[1:],status='inspected',what_it_can_establish='protocol/context only' if p.suffix in ['.py','.yaml','.yml'] else 'Outcome or execution evidence; see per-part status',sha256=sha(p)))
 pd.DataFrame(inventory).to_csv(out/'SourceInventory.csv',index=False)
 freeze=dict(week=w,reports={name:sha(out/name) for name in names},sources={x['artifact']:x['sha256'] for x in inventory},legacy_markdown_used_as_evidence=False,parts=len(parts),numerical_tables=tabcount,figures=figcount,checks=list(checks),scope_revision_reason=revision_reason)
 (out/'ReportFreeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
 timeline=[dict(week=w,part=x['id'],research_question=x['question'],method=x['name'],dataset=x['design'].get('dataset','See primary protocol'),status=x['status'],primary_result=x['finding'] or 'Not established',carried_forward=x['decision'] or decision or 'Not established') for x in parts]
 pd.DataFrame(timeline).to_csv(out/'TimelineRows.csv',index=False)
 print(json.dumps(dict(week=w,parts=len(parts),primary_artifacts=len(sources),tables=tabcount,figures=figcount,checks=checks)))
 return freeze

def audit_legacy(w):
 """Conservative post-freeze audit; uncertain files are preserved and explicitly queued for review."""
 out=BASE/f'week_{w:02d}';f=json.loads((out/'ReportFreeze.json').read_text());assert all(sha(out/n)==h for n,h in f['reports'].items())
 roots=[ROOT/f'results/week{w}',ROOT/f'docs/week{w}'];paths=sorted({p for r in roots if r.exists() for p in r.rglob('*.md')});rows=[]
 for p in paths:
  text=p.read_text(errors='replace') # allowed only after both authoritative reports were frozen
  unsupported=next((line.strip()[:260] for line in text.splitlines() if any(term in line.lower() for term in ['universally superior','proves latent','ignored evidence','causal reasoning process','all experiments completed'])),None)
  rows.append(dict(file=str(p.relative_to(ROOT)),status='FABRICATED / UNSUPPORTED' if unsupported else 'UNKNOWN',incorrect_claim=unsupported or 'Not certified by line-by-line reconciliation; legacy narrative excluded from reconstruction.',authoritative_evidence=str((out/'SourceInventory.csv').relative_to(ROOT)),action='Leave unchanged; manual review required. Preserve historical documentation; use frozen authoritative reports.',sha_before=sha(p),sha_after=sha(p),corrected=False))
 pd.DataFrame(rows,columns=['file','status','incorrect_claim','authoritative_evidence','action','sha_before','sha_after','corrected']).to_csv(out/'LegacyMarkdownAudit.csv',index=False)
 assert all(sha(out/n)==h for n,h in f['reports'].items());return len(rows)
