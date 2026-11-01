from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.week9.provenance import branch_guard
from src.week9.data import load_records
from src.week9.config import load_config
from src.week9.baseline_registry import build_registry

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['fit','internal_val','dev'],default='internal_val');p.add_argument('--config',default='configs/week9/w9h.yaml');a=p.parse_args()
    branch_guard();cfg=load_config(a.config);records,_=load_records(cfg,a.split)
    entries=build_registry(cfg,a.split,records)
    for e in entries:print(e['family'],e['display_name'],e['split'],e['status'],e['num_queries'])
if __name__=='__main__':main()
