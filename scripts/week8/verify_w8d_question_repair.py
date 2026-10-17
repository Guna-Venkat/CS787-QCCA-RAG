"""Prove the targeted LLM-prompt repair leaves original selector inputs unchanged."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from src.week8.closeout import cached_split,BASE_COLUMNS,KEYS
from src.week8.generator_features import source_fingerprint


def main():
    archive=ROOT/'results/week8/archive_w8d_question_normalization';results={}
    for split in ['fit','internal_val']:
        old=pd.read_parquet(archive/f'{split}_unchanged_base_features.parquet').sort_values(KEYS).reset_index(drop=True)
        new=cached_split(split).sort_values(KEYS).reset_index(drop=True)
        if old[KEYS].to_dict('records')!=new[KEYS].to_dict('records'):raise ValueError('Selector candidate identities changed')
        difference=float(np.max(abs(old[BASE_COLUMNS].to_numpy(float)-new[BASE_COLUMNS].to_numpy(float))))
        if difference>1e-12:raise ValueError(f'{split}: base feature inputs changed by {difference}')
        results[split]={'candidate_rows':len(new),'base_input_max_abs_difference':difference,
            'source_fingerprint':source_fingerprint(split)[0]}
    (archive/'base_inputs_verified.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
