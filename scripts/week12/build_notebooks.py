"""Build Week12 notebooks only. No existing notebook is modified."""
from pathlib import Path
import nbformat as nb
ROOT=Path(__file__).resolve().parents[2]
NAMES={'0':'protocol_and_split_audit','A':'qbgf_algorithm','B':'baseline_qualification','C':'validation_surface','D':'qbgf_ablations','E':'llmlingua2','F':'lora_training','G':'lora_validation','H':'validation_analysis','I':'week13_freeze'}
IMPORTS="""from pathlib import Path
import sys, json, subprocess
import pandas as pd
from IPython.display import display, Markdown, Image
ROOT=Path.cwd().resolve()
while not (ROOT/'src/week12').is_dir():
    if ROOT==ROOT.parent: raise FileNotFoundError('Open within CS787-RAG-Project')
    ROOT=ROOT.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.week12.protocol import load, root
from src.week12.runner import execute
cfg=load()
if OVERWRITE or not RESUME: raise ValueError('Immutable results: RESUME=True, OVERWRITE=False required')
effective_smoke=RUN_SMOKE and not RUN_FULL
print('MODE:', 'MANUAL FULL DEVELOPMENT' if RUN_FULL else 'SMOKE ONLY' if effective_smoke else 'READ SAVED ONLY')
"""
def main():
 out=ROOT/'notebooks/week12';out.mkdir(parents=True,exist_ok=True)
 for phase,title in NAMES.items():
  flags='CORE_FINAL = True\nRUN_SMOKE = True\nRUN_FULL = False\nRESUME = True\nOVERWRITE = False\nSMOKE_LIMIT = 3\nRUN_REAL_MODEL_SMOKE = False\nRUN_CANONICAL_SMOKE = False\nALLOW_LLMLINGUA_DOWNLOAD = False\nRUN_LLMLINGUA_SMOKE = True\nRUN_LLMLINGUA_FULL = False\nRUN_TINY_LORA_SMOKE = True\nRUN_FULL_LORA_TRAIN = False\nFINAL_METHOD = None\nFINAL_REASON = None\nRETAIN_LORA = False\n'
  intro=f'# Week12 {phase} — {title.replace("_"," ")}\n\nRun All is safe by default: synthetic smoke or read-only metadata. Full work requires changing the FLAGS cell yourself. The agent has not enabled full runs. Completed checksummed rows resume automatically; a changed scientific identity fails instead of overwriting. QASPER TEST and all Week13 confirmation outcomes are blocked.\n\nW12-C uses the full frozen VALIDATION split. The historical W11 pilot is hypothesis generation only. W12-I freezes the reviewed method; Week13 is the future locked evaluation. No outcome from reserved confirmation may inform selection.\n'
  if phase=='D':intro+='\nCORE D includes QBGF_FULL, QBGF_NO_QUERY, QBGF_SENTENCE_ONLY and QBGF_PASSAGE_ONLY at both 256 and 1024. NO_MASS, GREEDY and BM25 stay secondary. NO_QUERY removes allocation conditioning within the same query-dependent SFR pool; it does not remove query information from retrieval.\n'
  cells=[nb.v4.new_markdown_cell(intro),nb.v4.new_code_cell(flags),nb.v4.new_code_cell(IMPORTS)]
  if phase=='0':body="""from src.week12.data import audit
from src.week12.splits import read_freeze
freeze=audit(cfg)
summary=json.loads((root(cfg)/'protocol/split_summary.json').read_text())
display(pd.DataFrame(summary['counts']))
display(pd.DataFrame(summary['official_counts']))
display(summary['overlap_matrix'])
print('QASPER_FRESH_CONFIRMATION_AVAILABLE =',summary['QASPER_FRESH_CONFIRMATION_AVAILABLE'])
print('W11 cohort: HISTORICAL_DEVELOPMENT / HYPOTHESIS_GENERATION')
display(Markdown((ROOT/'results/week12/W12_DATA_EXPOSURE_AUDIT.md').read_text()))
"""
  elif phase=='E':body="""from src.week12.qualification import qualify_llm
qualification=qualify_llm(cfg,allow_download=ALLOW_LLMLINGUA_DOWNLOAD,smoke=RUN_LLMLINGUA_SMOKE)
display(qualification)
if RUN_LLMLINGUA_FULL:
    if qualification['status']!='QUALIFIED': raise RuntimeError('LLMLingua not qualified; no fake result or substitution')
    result=execute('E',cfg,full=True,smoke=False,explicit=True)
    display(pd.DataFrame(result))
else: print('No full LLMLingua experiment requested.')
"""
  elif phase=='F':body="""from src.week12.lora import tiny_smoke
from src.week12.protocol import atomic
from src.week12.splits import read_freeze
frozen=read_freeze(cfg)
counts=[{'dataset':d,'TRAIN':len(p['TRAIN']),'VAL':len(p['VAL']),'CONFIRM':len(p['CONFIRM']),'TRAIN_groups':len({r['group'] for r in p['TRAIN']}),'loss_weight':sum(len(x['TRAIN']) for x in frozen['datasets'].values())/(len(cfg['datasets'])*len(p['TRAIN']))} for d,p in frozen['datasets'].items()]
display(pd.DataFrame(counts))
print('ALL eligible TRAIN queries; one deterministic evidence condition/query. VAL is never used for gradients.')
if RUN_TINY_LORA_SMOKE: display(tiny_smoke(cfg))
if RUN_FULL_LORA_TRAIN:
    command=[str(ROOT/'sara_env/bin/python'),str(ROOT/'scripts/week12/run.py'),'--phase','F','--full','--explicit']
    subprocess.run(command+['--prepare-corpus'],cwd=ROOT,check=True)
    corpus=json.loads((root(cfg)/'lora/corpus/rows.json').read_text())
    panel=pd.DataFrame([{'dataset':r['dataset'],'partition':r['partition'],'mode':r['mode'],'budget':r['budget'],'tokens':len(r['input_ids']),'target':r['target'],'cluster_id':r['cluster_id']} for r in corpus])
    display(panel.groupby(['dataset','partition','mode','budget']).tokens.agg(['count','mean','max']))
    display(panel[['dataset','partition','target']].value_counts().head(30))
    display(panel.groupby(['dataset','partition']).cluster_id.nunique())
    display(json.loads((root(cfg)/'lora/corpus/training_plan.json').read_text()))
    print('Gradients: TRAIN only. Checkpoint selection: frozen VAL only. Starting manually opted-in training.')
    subprocess.run(command+['--train'],cwd=ROOT,check=True)
else: print('Full LoRA training disabled. Tiny smoke uses random miniature Qwen, not the real reader.')
"""
  elif phase in ('H','I'):body=f"""from src.week12.analysis import analyze, freeze_final
result=analyze(cfg,synthetic=effective_smoke)
display(result['summary'])
display(result['comparisons'])
display(result['fusion_comparisons'])
display(result['specialist_references'])
for f in result['figures']:
    display(Image(filename=f['png']))
"""+("display(freeze_final(cfg,FINAL_METHOD,FINAL_REASON,RETAIN_LORA))\n" if phase=='I' else '')
  else:body=f"""result=execute('{phase}',cfg,full=RUN_FULL,smoke=effective_smoke,explicit=RUN_FULL,limit=SMOKE_LIMIT,real_smoke=RUN_REAL_MODEL_SMOKE,core_final=CORE_FINAL)
display(pd.DataFrame(result))
print('SYNTHETIC rows are pipeline checks, never findings. Completed real rows are reused by identity.')
"""
  if phase=='A':
   body+="""from src.week12.runner import fixture_examples
from src.week12.laboratory import Lab
from src.week12.algorithm import options
from src.week11.budgets import FixtureTokenizer
example=fixture_examples()[0]
lab=Lab(cfg,FixtureTokenizer(),synthetic=True)
try:
    pool,sent,props,ps,ss,lineage=lab.prepare(example)
    display(pd.DataFrame([{'passage_id':p['unit_id'],'text':p['source_text'],'score':score} for p,score in zip(pool,ps)]))
    display(pd.DataFrame([{'sentence_id':s['unit_id'],'passage_id':s['passage_id'],'start':s['start_char'],'end':s['end_char'],'text':s['source_text'],'score':ss[s['unit_id']]} for s in sent]))
    table=pd.DataFrame([{k:v for k,v in o.items() if k not in ('units','ids')} for group in options(pool,sent,ps,ss,FixtureTokenizer()) for o in group])
    display(table)
    display(table.groupby('passage_id')[['utility','cost']].max().plot.bar(title='SYNTHETIC: available utility and token cost').figure)
finally: lab.close()
"""
  if phase=='B':body+="""if RUN_CANONICAL_SMOKE:
    for system in ['SARA_CANONICAL_BM25_K5_N10','CANONICAL_RAG_BM25']:
        subprocess.run([str(ROOT/'sara_env/bin/python'),str(ROOT/'scripts/week12/run.py'),'--phase','B','--canonical',system],cwd=ROOT,check=True)
else: print('Canonical Mistral references: source inventory shown; execution smoke explicitly opt-in.')
"""
  if phase=='C':body+="""if RUN_FULL and not CORE_FINAL:
    for system in ['SARA_CANONICAL_BM25_K5_N10','CANONICAL_RAG_BM25']:
        subprocess.run([str(ROOT/'sara_env/bin/python'),str(ROOT/'scripts/week12/run.py'),'--phase','C','--canonical',system,'--full','--explicit'],cwd=ROOT,check=True)
"""
  cells+=[nb.v4.new_markdown_cell('## Execute / resume\n\nAll necessary setup is derived above. CORE_FINAL=True restricts C/D to preregistered primary methods and budgets; secondary sensitivities require CORE_FINAL=False. No follow-on cell silently enables a full run.'),nb.v4.new_code_cell(body),nb.v4.new_markdown_cell('## Interpretation and next step\n\nUse the method registry and primary comparison table. No automatic winner promotion. Compare QBGF with RAW_SKIP and fixed sentences; relevance mass is a surrogate, not predicted F1. Macro-F1 is recomputed per resample. QASPER/Mistral specialists are separate from Qwen comparisons. All confirmation data remain locked.')]
  path=out/f'week12_{phase}_{title}.ipynb';notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'sara_env','language':'python','name':'sara_env'}});nb.write(notebook,path)
 print('Created 10 smoke-only notebooks; no experiment executed.')
if __name__=='__main__':main()
