"""Manual notebook reader setup. Never changes frozen Week-10 method code.

--check is read-only/offline. --download and --qualify are explicit actions used
only when the user runs the notebook. Downloads and per-reader smoke receipts
resume independently. No benchmark quality scoring or model selection occurs.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.week10.root_cause.storage import atomic,digest,file_hash,lock
from src.week10.root_cause.protocol import load,source_hashes

WEIGHT_BYTES={'Mistral':14_500_000_000,'Qwen':16_400_000_000,'Llama':16_100_000_000}
RESERVE_BYTES=3*1024**3


def snapshot_complete(directory):
    if directory is None:return False
    directory=Path(directory)
    if not all((directory/n).is_file() for n in ('config.json','tokenizer_config.json')):return False
    if not any((directory/n).is_file() for n in ('tokenizer.json','tokenizer.model')):return False
    for name in ('model.safetensors.index.json','pytorch_model.bin.index.json'):
        index=directory/name
        if index.is_file():
            shards=set(json.loads(index.read_text()).get('weight_map',{}).values())
            return bool(shards) and all((directory/n).is_file() and (directory/n).stat().st_size>0 for n in shards)
    return any((directory/n).is_file() and (directory/n).stat().st_size>0 for n in ('model.safetensors','pytorch_model.bin'))


def cached_snapshot(spec):
    from huggingface_hub import snapshot_download
    try:return Path(snapshot_download(spec['id'],revision=spec['revision'],local_files_only=True))
    except (OSError,ValueError):return None


def cache_root():
    from huggingface_hub.constants import HF_HUB_CACHE
    p=Path(HF_HUB_CACHE).expanduser()
    while not p.exists():p=p.parent
    return p


def inspect_panel(cfg):
    rows=[]
    for key in cfg['reader_panel']:
        spec=cfg['models'][key]
        if not spec.get('revision') or spec['revision']=='main':raise ValueError('Immutable reader revision required')
        directory=cached_snapshot(spec);complete=snapshot_complete(directory)
        rows.append(dict(reader=key,model=spec['id'],revision=spec['revision'],complete=complete,
                         directory=str(directory) if directory else None,
                         estimated_missing_bytes=0 if complete else WEIGHT_BYTES[key]))
    return rows


def check_space(rows,free_bytes):
    needed=sum(r['estimated_missing_bytes'] for r in rows)
    if needed and free_bytes<needed+RESERVE_BYTES:
        raise RuntimeError(f'Reader downloads paused: {free_bytes/1e9:.1f} GB free; need approximately '
            f'{(needed+RESERVE_BYTES)/1e9:.1f} GB free for the missing pinned weights and margin. '
            'Free space without deleting frozen results, then rerun Run All. No download was started.')
    return needed


def ensure_panel(cfg,download=False):
    rows=inspect_panel(cfg)
    for row in rows:print(f"{row['reader']}: {'complete cache' if row['complete'] else 'missing weights'} | {row['model']} @ {row['revision']}",flush=True)
    if not download:return rows
    check_space(rows,shutil.disk_usage(cache_root()).free)
    from huggingface_hub import snapshot_download
    for row in rows:
        if row['complete']:continue
        print('Downloading/resuming pinned reader '+row['reader']+'...',flush=True)
        try:
            directory=Path(snapshot_download(row['model'],revision=row['revision']))
        except Exception as exc:
            raise RuntimeError('Cannot download '+row['model']+' ('+type(exc).__name__+'). '
                'Check network and Hugging Face access. For Llama, obtain model access and authenticate '
                'with your own account, then rerun. No model substitution or result deletion is required.') from None
        if not snapshot_complete(directory):raise RuntimeError('Incomplete model snapshot: '+row['model'])
    return inspect_panel(cfg)


def smoke_inputs(root):
    path=root/'runs/C-Mistral'
    identity=json.loads((path/'identity.json').read_text())
    from src.week10.root_cause.storage import Store
    store=Store(path,identity);rows=store.rows()
    if not (path/'complete.json').exists():raise RuntimeError('Complete C before reader qualification')
    # Select by length only, not by answers, gold, or observed quality.
    full=max((r for r in rows if r['action']=='full'),key=lambda r:(r['evidence_tokens_canonical'],r['question_id']))
    question_id=full['question_id']
    group={r['action']:r for r in rows if r['question_id']==question_id}
    return question_id,{v:group[v]['evidence_text'] for v in ('no_evidence','full','qcs256')}


def weight_signature(directory):
    directory=Path(directory)
    return {str(p.relative_to(directory)):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(directory.rglob('*')) if p.is_file() and p.suffix in ('.safetensors','.bin','.json','.model')}


def qualify_one(cfg,key):
    from huggingface_hub import snapshot_download
    spec=cfg['models'][key]
    directory=Path(snapshot_download(spec['id'],revision=spec['revision'],local_files_only=True))
    if not snapshot_complete(directory):raise RuntimeError('Reader cache incomplete: '+key)
    root=ROOT/cfg['output_root'];out=root/'reader_setup';out.mkdir(exist_ok=True)
    question_id,evidence=smoke_inputs(root)
    question=None
    with Path(cfg['dataset']).open() as stream:
        for line in stream:
            row=json.loads(line)
            if str(row['id'])==question_id:question=row['question'];break
    if question is None:raise ValueError('DEV question identity missing')
    identity=dict(reader=spec,checkpoint_signature=weight_signature(directory),evidence_hash=digest(evidence),
        question_hash=digest(question),runtime_sha256=file_hash(ROOT/'src/week10/root_cause/runtime.py'),
        setup_sha256=file_hash(Path(__file__)),precision='fp16',attention='efficient_sdpa',max_new_tokens=128)
    receipt=out/(key+'.json')
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved.get('identity')==identity and saved.get('passed') is True:
            print('SKIP reader qualification '+key+': cached smoke passed; no model load.',flush=True)
            return
        raise ValueError('Reader qualification identity changed; inspect before requalifying '+key)
    import torch
    from src.week10.root_cause.runtime import TextReader
    if not torch.cuda.is_available():raise RuntimeError('Reader qualification requires the study GPU')
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    reader=None;measurements=[]
    try:
        reader=TextReader(spec)
        for variant,text in evidence.items():
            result=reader.generate(question,text)
            if not result['content'].strip():raise RuntimeError('Empty reader qualification output: '+key+'/'+variant)
            measurements.append(dict(variant=variant,prompt_tokens=result['prompt_tokens'],
                generated_tokens=result['generated_tokens'],seconds=result['latency']))
        # D uses sampling as a diagnostic; qualify this path without scoring it.
        if key=='Mistral':
            result=reader.generate(question,evidence['full'],seed=42)
            if not result['content'].strip():raise RuntimeError('Empty sampled diagnostic answer')
            measurements.append(dict(variant='D-sampling',prompt_tokens=result['prompt_tokens'],generated_tokens=result['generated_tokens']))
        peak=torch.cuda.max_memory_allocated()
        if peak>22*1024**3:raise MemoryError('Reader qualification exceeds the 22 GiB working-memory ceiling')
        hashes={name:file_hash(directory/name) for name in identity['checkpoint_signature']}
        atomic(receipt,dict(passed=True,identity=identity,weight_sha256=hashes,peak_allocated_bytes=peak,
            gpu=torch.cuda.get_device_name(0),question_id=question_id,measurements=measurements,
            quality_scored=False,generator_panel_changed=False))
        print('Reader qualification passed: '+key,flush=True)
    finally:
        if reader is not None:reader.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--check',action='store_true');parser.add_argument('--download',action='store_true')
    parser.add_argument('--qualify',action='store_true');parser.add_argument('--one',choices=WEIGHT_BYTES)
    parser.add_argument('--readers',nargs='+',choices=WEIGHT_BYTES,help='Explicit subset; frozen model definitions remain unchanged')
    args=parser.parse_args();cfg=load(args.config)
    frozen=json.loads((ROOT/cfg['output_root']/'execution_freeze.json').read_text())
    if frozen['config']!=cfg or frozen['sources']!=source_hashes():raise ValueError('Frozen study identity changed')
    if args.one:
        if not args.qualify:raise ValueError('--one requires explicit --qualify')
        return qualify_one(cfg,args.one)
    if not args.check and not args.download and not args.qualify:raise ValueError('Select check/download/qualify explicitly')
    with lock(ROOT/cfg['output_root']/'reader_setup/setup.lock'):
        selected=args.readers or cfg['reader_panel']
        if len(set(selected))!=len(selected) or any(k not in cfg['reader_panel'] for k in selected):
            raise ValueError('Reader subset must be unique and belong to the frozen panel')
        rows=ensure_panel(dict(cfg,reader_panel=selected),args.download)
        if args.qualify:
            if not all(r['complete'] for r in rows):raise RuntimeError('Pinned readers missing; run with --download first')
            for key in selected:
                # Separate processes guarantee one heavy reader at a time.
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--config',args.config,
                    '--qualify','--one',key],cwd=ROOT,check=True)
        elif args.check:
            free=shutil.disk_usage(cache_root()).free
            print(f'Free cache disk: {free/1e9:.1f} GB; missing weights estimate: {sum(r["estimated_missing_bytes"] for r in rows)/1e9:.1f} GB')

if __name__=='__main__':main()
