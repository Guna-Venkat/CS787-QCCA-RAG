"""Execute an authorized full notebook copy; retain per-cell progress and outputs.

This operational launcher does not change frozen experimental source/configuration.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import nbformat
from nbclient import NotebookClient
from src.week11.cache import atomic, file_hash, lock
from src.week11.catalog import NOTEBOOKS
from src.week11.protocol import load, gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=list(NOTEBOOKS))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--cached-b', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        raise ValueError('Explicit --execute required for user-authorized full runs')
    cfg = load()
    gate(cfg, args.phase)
    source = ROOT / 'notebooks/week11' / NOTEBOOKS[args.phase]
    destination = ROOT / 'results/week11/executed_notebooks' / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    state_path = ROOT / 'results/week11/monitoring' / ('cell_state_' + args.phase + '.json')
    notebook = nbformat.read(source, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    flags = next(cell for cell in notebook.cells if cell.cell_type == 'code')
    overrides = dict(RUN_FULL=True, RUN_SMOKE=False, RESUME=True, OVERWRITE=False,
                     REVIEW_PHASE=None, REVIEWED_CONFIG=None, REVIEW_REASON=None)
    lines = flags.source.splitlines()
    for i, line in enumerate(lines):
        name = line.split('=', 1)[0].strip()
        if name in overrides:
            lines[i] = name + ' = ' + repr(overrides[name])
    flags.source = '\n'.join(lines)
    if args.cached_b:
        if args.phase != 'B':
            raise ValueError('--cached-b is restricted to Phase B')
        imports = next(cell for cell in notebook.cells if cell.cell_type == 'code' and 'cfg = load()' in cell.source)
        imports.source += '''
# Recorded execution optimization; frozen scientific source/config remain unchanged.
from operations.week11.fast_b import execute_b
execute_reference = execute
def execute(phase, cfg=None, **kwargs):
    if phase == 'B' and kwargs.get('full'):
        return execute_b(cfg, **kwargs)
    return execute_reference(phase, cfg, **kwargs)
'''
    state = dict(phase=args.phase, pid=os.getpid(), notebook=str(destination),
                 source_notebook_sha256=file_hash(source),
                 launcher_sha256=file_hash(Path(__file__)),
                 start_utc=datetime.now(timezone.utc).isoformat(), status='RUNNING',
                 cached_b=args.cached_b,
                 authorization='User delegated sequential full runs, interpretation, fixes and reviews on 2026-10-28')

    def save(**changes):
        state.update(changes, updated_utc=datetime.now(timezone.utc).isoformat())
        atomic(state_path, state)
        nbformat.write(notebook, destination)

    def start_cell(cell, cell_index, **kwargs):
        save(current_cell=cell_index, cell_source_preview=cell.source[:160])
        print('Phase', args.phase, 'cell', cell_index, 'started', flush=True)

    def end_cell(cell, cell_index, **kwargs):
        save(last_executed_cell=cell_index)

    with lock(state_path.with_suffix('.lock')):
        save()
        try:
            client = NotebookClient(notebook, timeout=None, kernel_name='sara_env',
                                    resources={'metadata': {'path': str(ROOT)}},
                                    allow_errors=False, iopub_timeout=600)
            client.on_cell_start = start_cell
            client.on_cell_executed = end_cell
            client.execute()
            save(status='COMPLETE', current_cell=None)
            print('Phase', args.phase, 'notebook COMPLETE', flush=True)
        except BaseException as exc:
            save(status='ERROR', error_type=type(exc).__name__, error=str(exc)[-8000:])
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
