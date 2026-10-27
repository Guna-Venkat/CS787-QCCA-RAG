"""Read-only probe by default; explicit fabricated qualification only."""
import argparse
import json
from pathlib import Path
import sys
import yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.week11.protocol import load
from src.week11.compressors import LLMLingua2,availability
from src.week11.readers import tokenizers
from src.week11.cache import atomic,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--qualify',action='store_true');p.add_argument('--revision')
    p.add_argument('--download',action='store_true');p.add_argument('--write-config',action='store_true');a=p.parse_args();cfg=load()
    print(json.dumps(availability(cfg),indent=2))
    if not a.qualify:return
    if not a.revision or a.revision=='main':raise ValueError('Immutable compressor revision required')
    from huggingface_hub import snapshot_download
    snapshot_download(cfg['token_compressor']['id'],revision=a.revision,local_files_only=not a.download,ignore_patterns=['original/*'])
    spec=dict(cfg['token_compressor'],revision=a.revision,qualified=True)
    text='The fabricated trial supports treatment alpha. Replication is required. '*30
    result=LLMLingua2(spec).compress(text,256,'Which treatment is discussed?',tokenizers(cfg))
    atomic(ROOT/cfg['output_root']/'smoke/token_compressor_qualification.json',dict(passed=True,spec=spec,
        fabricated=True,input_hash=digest(text),output=result,no_answers_generated=True))
    if a.write_config:
        path=ROOT/'configs/week11/week11_compressors.yaml';value=yaml.safe_load(path.read_text())
        value['token_compressor']=dict(spec,status='QUALIFIED_FABRICATED_INPUT_ONLY');path.write_text(yaml.safe_dump(value,sort_keys=False))
    print('Fabricated token-compressor qualification passed. Do not infer QA performance.')


if __name__=='__main__':main()
