"""Explicitly authorized K2 diagnostic; never substitute for primary B winner."""
import json
from pathlib import Path
from datetime import datetime,timezone
from src.week9.config import load_frozen,load_config
from src.week9.runner import dependencies
from src.week9.provenance import ROOT,branch_guard,file_hash,digest
from src.week9.caching import atomic_json
branch_guard()
primary=ROOT/'results/week9/frozen/w9b_frozen_config.json'
b=load_frozen(primary,'b')
assert b['settings']==dict(policy='K_E=0',evidence_k=0)
config=ROOT/'configs/week9/w9c_k2_diagnostic.yaml'
cfg=load_config(config)
root=ROOT/cfg['output_root']
(root/'frozen').mkdir(parents=True,exist_ok=True)
# Share completed read-only inputs; do not copy multi-GB A caches or regenerate B.
for folder in ('retrieval','ke_surface'):
 link=root/folder
 target=ROOT/'results/week9'/folder
 if link.is_symlink():assert link.resolve()==target.resolve()
 else:
  assert not link.exists()
  link.symlink_to(Path('../..')/folder,target_is_directory=True)
a=ROOT/'results/week9/frozen/w9a_frozen_config.json'
a_copy=root/'frozen'/a.name
if a_copy.exists():assert a_copy.read_bytes()==a.read_bytes()
else:a_copy.write_bytes(a.read_bytes())
value=dict(b)
value.update(settings=dict(policy='K_E=2',evidence_k=2),analysis_role='fit_diagnostic_fixed_k',diagnostic_only=True,
 diagnostic_label=cfg['diagnostic_label'],primary_frozen_artifact=str(primary),primary_settings=b['settings'],
 decision_basis='User explicitly authorized a separate positive-budget selector diagnostic. K=2 is the best positive FIT B budget. Primary cost-aware decision remains K=0.',
 timestamp=datetime.now(timezone.utc).isoformat())
value['settings_hash']=digest(value['settings'])
value['input_hashes']={**b['input_hashes'],str(primary):file_hash(primary),str(config):file_hash(config)}
path=root/'frozen'/primary.name
if path.exists():
 old=json.loads(path.read_text());value['timestamp']=old['timestamp'];assert value==old
else:atomic_json(path,value)
(root/'README.md').write_text('# W9-C K=2 selector diagnostic\n\nDiagnostic only, explicitly authorized after B selected K=0. Primary B is preserved in results/week9/frozen. The local B artifact pins K=2 for this experiment and is labelled fit_diagnostic_fixed_k; it is not the cost-aware primary winner.\n\nRetrieval and ke_surface are read-only upstream links, not duplicated data. C outputs belong only under this diagnostic root. Do not use this root to run A/B or automatically promote this selector into primary D-H. Keep FREEZE_FIT=False. Only FIT is prepared.\n')
frozen=dependencies(cfg,'c','fit')
assert frozen['b']['diagnostic_only'] and frozen['b']['settings']['evidence_k']==2
assert load_frozen(primary,'b')['settings']['evidence_k']==0
print('Primary K=0 preserved; separate K=2 diagnostic dependencies validated.')
