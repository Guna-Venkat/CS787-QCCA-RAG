"""Resume measured gate backfill, construct frozen controller, validate notebook."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-for-pid',type=int,help='Wait for this already-running FIT worker before continuing')
    a=p.parse_args()
    folder=ROOT/'results/week8/w8e_controller';folder.mkdir(parents=True,exist_ok=True)
    status=folder/'pipeline_status.json'
    def save(stage,**extra):status.write_text(json.dumps({'stage':stage,**extra},indent=2))
    if a.wait_for_pid:
        save('waiting_for_authorized_fit_backfill',pid=a.wait_for_pid)
        while True:
            try:os.kill(a.wait_for_pid,0)
            except ProcessLookupError:break
            # Existing process may be a zombie until its shell is reaped.
            stat=Path(f'/proc/{a.wait_for_pid}/stat')
            if stat.exists() and stat.read_text().split()[2]=='Z':break
            time.sleep(10)
    environment={**os.environ,'HF_HUB_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1'}
    commands=[]
    repair=ROOT/'results/week8/archive_w8d_question_normalization/repair_completed.json'
    if repair.exists():
        commands.append(('verify_repaired_base_inputs',[sys.executable,str(ROOT/'scripts/week8/verify_w8d_question_repair.py')]))
    if not (ROOT/'results/week8/w8d_closeout/closeout_summary.json').is_file():
        commands.append(('cached_w8d_closeout',[sys.executable,str(ROOT/'scripts/week8/run_w8d_closeout.py')]))
    commands.append(('report_w8d_closeout',[sys.executable,str(ROOT/'scripts/week8/report_w8d_closeout.py')]))
    commands += [('backfill_measured_gate_inputs',[sys.executable,str(ROOT/'scripts/week8/backfill_w8e_gate_inputs.py'),'--split','all']),
        ('validate_measured_inputs',[sys.executable,str(ROOT/'scripts/week8/run_w8e_controller_tuning.py'),'--check']),
        ('fit_freeze_and_confirm_controller',[sys.executable,str(ROOT/'scripts/week8/run_w8e_controller_tuning.py'),'--full'])]
    for stage,command in commands:
        save(stage)
        print(stage,flush=True)
        with (folder/f'{stage}.log').open('a') as log:
            result=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            save('failed',failed_stage=stage,exit_code=result.returncode,log=str(folder/f'{stage}.log'))
            raise RuntimeError(f'{stage} failed; inspect its saved log')
    save('validate_notebook')
    import nbformat
    from nbclient import NotebookClient
    for name in ['week8e_controller.ipynb','week8d_generator_interaction.ipynb','week8d_cached_closeout.ipynb']:
        path=ROOT/'notebooks/week8'/name
        n=nbformat.read(path,as_version=4)
        NotebookClient(n,timeout=300,kernel_name='sara_env',resources={'metadata':{'path':str(ROOT)}}).execute()
        nbformat.write(n,path)
    save('complete',notebook=str(ROOT/'notebooks/week8/week8e_controller.ipynb'),summary=str(folder/'controller_summary.json'))
    print('Complete: frozen controller and executed notebook saved',flush=True)

if __name__=='__main__':main()
