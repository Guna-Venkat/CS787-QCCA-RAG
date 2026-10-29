"""Read-only TRAIN consumption audit. No model, corpus preparation or training."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.week12.protocol import load,root
from src.week12.splits import read_freeze
from src.week12.scheduling import full_coverage,CoverageLedger
c=load();f=read_freeze(c);rows=[dict(dataset=d,query_id=r['query_id']) for d,p in f['datasets'].items() for r in p['TRAIN']];_,plan=full_coverage(rows,c['lora']['gradient_accumulation_steps'],c['seed']);path=root(c)/'lora/coverage_events.jsonl'
if path.exists():
 ledger=CoverageLedger(path,[(r['dataset'],r['query_id']) for r in rows],plan['effective_loss_weights']);actual=ledger.summary()
else:actual=dict(status='NOT_STARTED',physical_unique_examples_seen={d:0 for d in f['datasets']},physical_repeats={d:0 for d in f['datasets']},applied_optimizer_steps=0)
print(json.dumps(dict(planned=plan,actual=actual),indent=2))
