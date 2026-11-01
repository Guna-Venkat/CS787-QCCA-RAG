"""Week12 M wrapper. Defaults are metadata-only; no real-cohort smoke exists."""
import argparse,json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight','synthetic','full','prepare','generate','analyze'],default='preflight')
    for flag in ['full','explicit','allow-confirmation-label-access','run-confirmation-analysis']:p.add_argument('--'+flag,action='store_true')
    p.add_argument('--freeze-sha');a=p.parse_args()
    from src.week12_confirmation.freeze_guard import preflight,authorize
    if a.stage=='preflight':print(json.dumps(preflight(),indent=2));return
    if a.stage=='synthetic':
        from src.week12_confirmation.synthetic import run
        from src.week12_confirmation.confirmation_analysis import analyze_synthetic
        out,ident=run();analyze_synthetic(out,ident);print('SYNTHETIC ONLY:',out);return
    permit=authorize(full=a.full and a.explicit,allow_labels=a.allow_confirmation_label_access,analysis=a.run_confirmation_analysis,ack_sha=a.freeze_sha)
    from src.week12.protocol import load
    from src.week12_confirmation.confirmation_runner import prepare,generate,ensure_k_idle
    ensure_k_idle();cfg=load();ready=preflight(cfg);print(json.dumps(ready,indent=2))
    if ready['free_disk_GiB']<2:raise RuntimeError('Less than 2 GiB free; preserve artifacts and free space before full confirmation')
    if a.stage=='full':
        # Separate processes release SFR before loading Qwen FP16.
        from src.week12.protocol import lock
        from src.week12_confirmation.freeze_guard import folder
        with lock(folder()/'run.lock'):
          for stage in ['prepare','generate','analyze']:
            subprocess.run([sys.executable,__file__,'--stage',stage,'--full','--explicit','--allow-confirmation-label-access','--run-confirmation-analysis','--freeze-sha',a.freeze_sha],cwd=ROOT,check=True)
    elif a.stage=='prepare':prepare(cfg,permit)
    elif a.stage=='generate':generate(cfg,permit)
    else:
        from src.week12_confirmation.confirmation_analysis import analyze
        analyze(cfg,permit)
if __name__=='__main__':main()
