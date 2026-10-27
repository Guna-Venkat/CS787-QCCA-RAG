"""Build safe notebook entry points. This script never executes experiments."""
from pathlib import Path
import sys
import nbformat as nb
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.week11.catalog import NOTEBOOKS,TABLES,PLOTS
from src.week11.protocol import DIRECTORIES

IMPORTS="""from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Markdown, Image
ROOT = Path.cwd().resolve()
while not (ROOT/'src/week11').is_dir():
    if ROOT == ROOT.parent: raise FileNotFoundError('Open inside CS787-RAG-Project')
    ROOT = ROOT.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.week11.protocol import load, review
from src.week11.runner import execute, notebook_smoke
from src.week11.dashboard import status, resources
cfg = load()
pd.set_option('display.max_columns', 24)
pd.set_option('display.max_colwidth', 70)
configuration_df = pd.DataFrame([{'setting': k, 'value': str(v)} for k,v in cfg.items()])
display(configuration_df)
run_status_df = status(cfg)
display(run_status_df)
display(resources(cfg))
"""


def write(path,cells):
    notebook=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='sara_env',language='python',name='sara_env')))
    nb.write(notebook,path)


def main():
    out=ROOT/'notebooks/week11';out.mkdir(parents=True,exist_ok=True)
    for phase,name in NOTEBOOKS.items():
        analysis=phase in ('H','I')
        flags='RUN_SMOKE = '+('False' if analysis else 'True')+'\nRUN_FULL = False\nRESUME = True\nOVERWRITE = False\n'
        flags+=f'RUN_{"SETUP" if phase=="0" else phase}_SMOKE = RUN_SMOKE\nRUN_{phase}_FULL = RUN_FULL\n'
        if phase=='0':flags+='BUILD_FULL_CACHES = RUN_FULL\n'
        flags+='SMOKE_LIMIT = 3\nREVIEW_PHASE = None\nREVIEWED_CONFIG = None\nREVIEW_REASON = None\n'
        intro=f'# Week 11 — {phase}: {DIRECTORIES[phase].replace("_"," ")}\n\nSAFE BY DEFAULT. Run All loads saved real results if present; otherwise it uses fabricated smoke fixtures (H/I read saved results only). No datasets/models are downloaded automatically. QASPER TEST is blocked.\n\n'
        intro+='For a user full run: set `RUN_FULL=True` and `RUN_SMOKE=False` in the next cell. Every later setup is derived automatically. Existing validated rows are reused. `OVERWRITE=True` is deliberately unsupported.\n\n'
        if phase=='0':intro+='Before full setup, prepare HotpotQA/SciFact once with `sara_env/bin/python scripts/week11/prepare_datasets.py --download`. Decide the fixed sample size (default 50 questions per dataset) and LLMLingua qualification before freezing this setup.\n'
        if phase=='F':intro+='Full F refuses to execute until `finalist_run_ids` and their manually reviewed configurations are supplied in `configs/week11/week11_runs.yaml`. No automatic finalist selection.\n'
        cells=[nb.v4.new_markdown_cell(intro),nb.v4.new_code_cell(flags),nb.v4.new_code_cell(IMPORTS)]
        cells.append(nb.v4.new_markdown_cell('## Execute or resume\n\nSynthetic output is stored only under `smoke/`. Pending token-compressor conditions are reported explicitly, never filled with invented scores. Full dependent phases require the previous manual freeze.'))
        cells.append(nb.v4.new_code_cell(f"""if OVERWRITE: raise ValueError('Immutable results: use a new output version instead of overwrite')
if not RESUME: raise ValueError('Resume stays enabled; new experiments require a new configured version')
effective_smoke = notebook_smoke(cfg, '{phase}', RUN_SMOKE, RUN_FULL)
frames, figures = execute('{phase}', cfg, full=RUN_FULL, explicit=RUN_FULL, smoke=effective_smoke, limit=SMOKE_LIMIT)
print('SYNTHETIC SMOKE ONLY' if effective_smoke else 'Saved/full result tables; inspect status before review')
display(frames.get('condition_status_df', pd.DataFrame()))
"""))
        for table in TABLES[phase]:
            cells.append(nb.v4.new_markdown_cell('### '+table))
            cells.append(nb.v4.new_code_cell(f"{table} = frames.get('{table}', pd.DataFrame())\ndisplay({table})"))
        cells.append(nb.v4.new_markdown_cell('## Figures and underlying tables\n\nEach PNG/PDF has a source CSV. Different datasets are plotted separately; cross-dataset displays use ranks/counts.'))
        cells.append(nb.v4.new_code_cell("""for figure in figures:
    print(figure['name'], figure['dataset'])
    display(pd.read_csv(figure['data']))
    display(Image(filename=figure['png']))
if not figures: print('No completed data for these plots yet; no fabricated values were substituted.')
"""))
        cells.append(nb.v4.new_markdown_cell('## Manual review / freeze\n\nAfter inspecting real completed results, edit the review fields in the first FLAGS cell. Set `REVIEW_PHASE` to this phase, `REVIEWED_CONFIG` to a dictionary, and give a rationale. No winning setting is chosen automatically. Smoke cannot be frozen. If optional conditions remain pending, list them explicitly in `REVIEWED_CONFIG["waive_pending"]`, or qualify them before running.'))
        cells.append(nb.v4.new_code_cell(f"""if REVIEW_PHASE is not None:
    if RUN_SMOKE: raise ValueError('Cannot freeze fabricated smoke output')
    if REVIEW_PHASE != '{phase}': raise ValueError('Review this notebook phase only')
    display(review(cfg, REVIEW_PHASE, REVIEWED_CONFIG, REVIEW_REASON))
else:
    print('Review not requested. No downstream configuration was selected automatically.')
display(status(cfg))
"""))
        write(out/name,cells)
    cells=[nb.v4.new_markdown_cell('# Week 11 MASTER — read-only dashboard\n\nThis notebook never starts generation or dataset/model downloads.'),
           nb.v4.new_code_cell('RUN_SMOKE = False\nRUN_FULL = False\nRESUME = True\nOVERWRITE = False'),nb.v4.new_code_cell(IMPORTS),
           nb.v4.new_markdown_cell('## Exact manual execution order'),
           nb.v4.new_code_cell("""from src.week11.catalog import NOTEBOOKS
for phase, name in NOTEBOOKS.items():
    display(Markdown(f"{phase}. [{name}]({ROOT/'notebooks/week11'/name})"))
"""),nb.v4.new_markdown_cell('## Dataset, cache and result availability'),
           nb.v4.new_code_cell("""from src.week11.runner import load_existing
setup_frames, _ = load_existing(cfg, '0')
for name, table in setup_frames.items():
    print(name)
    display(table)
for phase in NOTEBOOKS:
    _, figures = load_existing(cfg, phase)
    if figures:
        print('Completed figures:', phase)
        for figure in figures[:2]: display(Image(filename=figure['png']))
""")]
    write(out/'week11_MASTER.ipynb',cells)
    print('Created 11 safe notebooks. No experimental phase was executed.')


if __name__=='__main__':main()
