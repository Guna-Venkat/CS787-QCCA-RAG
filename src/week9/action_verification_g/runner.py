"""Saved-output W9-G preflight; future training requires separate explicit opt-in."""
import argparse
import json
from pathlib import Path
from ..provenance import branch_guard,digest,file_hash
from ..caching import atomic_json
from .protocol import config,inputs,identity,resolve,pin,verify_hashes
from .preflight import analyze


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=('preflight','train'),default='preflight')
    parser.add_argument('--enable-training',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);branch_guard();cfg=config(args.config)
    if args.stage=='train' and not args.enable_training:
        raise PermissionError('Controller training is OFF; evaluate preflight, then explicitly pass --enable-training')
    print('Validating saved FIT F/E3 actions and hashes; no generator/encoder is loaded.',flush=True)
    bundle=inputs(cfg);ident=identity(cfg,bundle);root=resolve(cfg['output_root']);directory=root/'preflight'
    if args.stage=='train':
        # Lazy import guarantees preflight cannot instantiate or fit an estimator.
        from .controller import train_manually
        return train_manually(cfg,bundle,ident,args.enable_training,args.resume)
    pin(directory/'protocol.json',ident)
    decision=analyze(bundle['groups'],cfg,directory)
    verify_hashes(bundle['input_hashes'])
    artifacts={str(p.resolve()):file_hash(p)
               for p in directory.iterdir() if p.is_file() and p.name!='preflight_manifest.json'}
    atomic_json(directory/'preflight_manifest.json',dict(schema=cfg['experiment'],split='fit',complete=True,
        queries=len(bundle['pairs']),papers=len({p for _,p in bundle['pairs']}),protocol_hash=digest(ident),
        old_artifacts_unchanged=True,lm_calls=0,embedding_model_loaded=False,controller_training_started=False,
        recommended_route=decision['recommended_route'],artifact_hashes=artifacts))
    print(json.dumps({k:v for k,v in decision.items() if k!='artifact_hashes'},indent=2),flush=True)
    print('W9-G PREFLIGHT COMPLETED\nCONTROLLER TRAINING WAS NOT STARTED',flush=True)


if __name__=='__main__':main()
