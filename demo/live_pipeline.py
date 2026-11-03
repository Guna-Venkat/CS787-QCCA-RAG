"""Conservative optional live gate. Never load models or download on app startup.
Full model execution is deliberately unavailable without a verified original pool;
cache cleanup means reproducing a new pool would not be a historical replay.
"""
import shutil, subprocess
from pathlib import Path
from demo.constants import ROOT

def readiness():
    hub=Path.home()/'.cache/huggingface/hub'
    models={}
    for label,folder in [('SFR','models--Salesforce--SFR-Embedding-Mistral'),('Qwen','models--Qwen--Qwen3-8B')]:
        p=hub/folder
        models[label]=p.exists() and any(p.glob('snapshots/*/*.safetensors'))
    try:
        x=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
        free=int(x.stdout.splitlines()[0]) if x.returncode==0 else None
    except (FileNotFoundError,ValueError,IndexError,subprocess.TimeoutExpired):free=None
    prepared=ROOT/'results/week12/qbgf/v3/validation/C/prepared'
    pool=next((p for p in prepared.glob('*.json') if p.name!='identity.json'),None) if prepared.exists() else None
    return dict(models=models,gpu_free_mib=free,original_prepared_pool=pool is not None,enabled=False,
                reason='Live inference unavailable; using cached experimental results. Original top-10 replay shards are missing. No models are loaded and no automatic downloads are permitted.')
