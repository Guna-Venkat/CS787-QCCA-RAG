"""Execute the user-authorized available-panel notebook and save cell outputs."""
import argparse
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:raise RuntimeError('Explicit execution authorization required')
    import nbformat
    from nbclient import NotebookClient
    from src.week10.root_cause.storage import lock,atomic
    from src.week10.root_cause.protocol import load
    cfg=load(ROOT/'configs/week10/root_cause_v1.json')
    root=ROOT/cfg['output_root'];path=ROOT/'notebooks/week10/week10_root_cause.ipynb'
    notebook=nbformat.read(path,as_version=4)
    def save():
        temp=path.with_suffix('.running.ipynb');nbformat.write(notebook,temp);temp.replace(path)
    started=time.time()
    def on_start(**kwargs):print(f"CELL {kwargs.get('cell_index')} started; elapsed={time.time()-started:.0f}s",flush=True)
    def on_done(**kwargs):
        save();print(f"CELL {kwargs.get('cell_index')} saved; elapsed={time.time()-started:.0f}s",flush=True)
    with lock(root/'notebook_orchestration/available_execution.lock'):
        client=NotebookClient(notebook,timeout=None,kernel_name='sara_env',resources={'metadata':{'path':str(ROOT)}},
            on_cell_start=on_start,on_cell_executed=on_done)
        try:
            client.execute();save()
            atomic(root/'notebook_orchestration/available_execution.json',dict(status='complete',seconds=time.time()-started,
                active_readers=['Mistral','Qwen'],pending_readers=['Llama'],full_panel_complete=False))
            print('AVAILABLE NOTEBOOK COMPLETE; Llama pending; full panel incomplete.',flush=True)
        except BaseException as exc:
            save();atomic(root/'notebook_orchestration/available_execution.json',dict(status='interrupted_or_failed',
                error_type=type(exc).__name__,seconds=time.time()-started,active_readers=['Mistral','Qwen'],pending_readers=['Llama']))
            raise

if __name__=='__main__':main()
