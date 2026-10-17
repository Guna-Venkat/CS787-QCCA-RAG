from pathlib import Path
import sys,argparse,torch,fcntl,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.week8.gate_diagnostics import backfill_gate_inputs

def main():
    p=argparse.ArgumentParser(description='Measured canonical token uncertainty for B0-GW')
    p.add_argument('--split',choices=['fit','internal_val','all'],required=True)
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/week8/w8e_gate_inputs')
    p.add_argument('--device',default='cuda');p.add_argument('--limit',type=int)
    a=p.parse_args();torch.set_num_threads(4)
    if a.limit is not None and a.limit<1:p.error('--limit must be positive')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for split in (['fit','internal_val'] if a.split=='all' else [a.split]):
        with (a.output_dir/f'{split}.lock').open('w') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'A {split} backfill already owns this output directory') from exc
            r=backfill_gate_inputs(split,a.output_dir,a.device,a.limit)
            raw_path=a.output_dir/f'canonical_diagnostics_{split}.jsonl'
            raw=[json.loads(line) for line in raw_path.read_text().splitlines()]
            if not all(row.get('greedy_verified') is True for row in raw):
                raise ValueError('Cached token/logit alignment has not been verified')
            measured=pd.DataFrame([{'question_id':row['question_id'],'paper_id':row['paper_id'],
                                    **row['features']} for row in raw])
            pairs=r.merge(measured,on=['question_id','paper_id'],validate='one_to_one',suffixes=('_csv','_raw'))
            columns=[c for c in r if c.startswith('u')]
            if len(pairs)!=len(r) or not np.allclose(pairs[[c+'_csv' for c in columns]],pairs[[c+'_raw' for c in columns]],rtol=0,atol=1e-10):
                raise ValueError('Gate feature rows differ from their measured token diagnostics')
            provenance_path=a.output_dir/f'provenance_{split}.json'
            provenance=json.loads(provenance_path.read_text())
            provenance['feature_csv_sha256']=hashlib.sha256((a.output_dir/f'gate_features_{split}.csv').read_bytes()).hexdigest()
            provenance['raw_diagnostics_sha256']=hashlib.sha256(raw_path.read_bytes()).hexdigest()
            provenance_path.write_text(json.dumps(provenance,indent=2))
            print(f'Validated measured gate inputs: {len(r)} {split} queries',flush=True)
if __name__=='__main__':main()
