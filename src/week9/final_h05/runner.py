"""H0.5/v2 gate: no implicit TEST execution and no fallbacks."""
import argparse
import subprocess
from pathlib import Path
from .readiness import publish,validate_v2,V2
from ..final_h import protocol as p


def confirm(root,allow=False,resume=False):
    p.require(allow,'Final confirmation disabled; explicit flag required')
    root=Path(root);freeze=validate_v2(root)
    registry=p.read(root/'producer_registry.json')
    p.require(set(registry['systems'])==set(p.REQUIRED),'All 12 producers required')
    spec=p.read(root/'test_materialization_spec.json')
    p.require(set(spec['systems'])==set(registry['order']) and len(registry['order'])==12,'Frozen producer set differs')
    p.require(not (root/'h2_seal.json').exists(),'One-shot confirmation already sealed; use H3 only')
    if not resume:
        for entry in registry['systems'].values():
            p.require(not p.resolve(entry['result_path']).exists() and not p.resolve(entry['receipt_path']).exists(),'Existing output requires exact explicit resume')
        p.require(not (root/'shared').exists() and not (root/'producer_cache').exists(),'Partial materialization requires explicit resume')
    claim=root/'h2_claim.json'
    if claim.exists():
        p.require(resume and p.read(claim)['freeze_digest']==freeze['manifest_digest'],'Existing claim requires identical explicit resume')
    else:
        p.require(not resume,'Cannot resume without an existing H2 claim')
        p.write_once(claim,dict(started_utc=p.timestamp(),freeze_digest=freeze['manifest_digest'],split='test',tuning_allowed=False))
    # Explicit order is independent of JSON object serialization order.
    for name in registry['order']:
        source=spec['systems'][name]
        p.verify_hashes(source['producer_source_hashes'])
        p.require(source['command']==registry['systems'][name]['command'],'Producer command differs')
        subprocess.run(source['command'],cwd=p.ROOT,check=True)
    from .materialize import lines
    from .storage import membership
    from ..final_h.analysis import evaluate,verify_post_generation_costs
    from ..final_h.runner import validate_w9_rows
    expected=lines(p.resolve(spec['membership_path']));keys=membership(expected)
    raw_labels=lines(p.resolve(spec['labels_path']));labels={str(r['question_id']):r for r in raw_labels}
    p.require(len(labels)==len(raw_labels) and sorted((q,str(r['paper_id'])) for q,r in labels.items())==keys,'Labels/common membership differ')
    groups={};hashes={}
    for name in registry['order']:
        source=spec['systems'][name];path=p.resolve(source['predictions_path']);receipt_path=p.resolve(source['receipt_path'])
        receipt=p.read(receipt_path)
        p.require(receipt['complete'] and receipt['system']==name and receipt['split']=='test','Invalid producer receipt')
        p.require(receipt['freeze_digest']==freeze['manifest_digest'] and receipt['method_config']==source['method_config'],'Producer method/freeze changed')
        p.require(receipt['producer_source_hashes']==source['producer_source_hashes'] and receipt['predictions_sha256']==p.file_hash(path),'Producer source/output checksum differs')
        raw=lines(path);p.require(all(r['system']==name for r in raw),'Mislabelled producer output')
        groups[name]=evaluate(raw,labels,'test')
        hashes[str(path)]=p.file_hash(path);hashes[str(receipt_path)]=p.file_hash(receipt_path)
    validate_w9_rows(groups,freeze);verify_post_generation_costs(groups[p.PRIMARY],groups[p.VERIFY])
    base={r['question_id']:r for r in groups['D0']}
    for group in groups.values():
        p.require(all(r['initial_draft']==base[r['question_id']]['prediction'] for r in group),'Canonical D0 differs across systems')
    rows=[dict(r,evaluation=labels[r['question_id']]['evaluation']) for group in groups.values() for r in group]
    p.write_once(root/'sealed_predictions.json',rows)
    for path in list((root/'shared').iterdir())+[root/'sealed_predictions.json']:
        if path.is_file():hashes[str(path)]=p.file_hash(path)
    validate_v2(root)
    p.write_once(root/'h2_seal.json',dict(complete=True,completed_utc=p.timestamp(),freeze_digest=freeze['manifest_digest'],
        split='test',systems=registry['order'],queries=len(keys),artifact_hashes=hashes,one_shot=True))
    return root/'h2_seal.json'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/week9/w9h_final_v2.json')
    parser.add_argument('--stage',choices=('h0','h1','h2','h3'),default='h0')
    parser.add_argument('--run-final-confirmation',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);cfg=p.load_config(p.resolve(args.config));root=p.resolve(cfg['output_root'])
    if args.stage=='h0':
        freeze=publish(p.resolve(args.config));print('W9-H H0 v2 PASS; 12 producers ready. Final confirmation NOT STARTED.')
    elif args.stage=='h1':
        validate_v2(root)
        from ..final_h.runner import materialize
        materialize(root)
    elif args.stage=='h2':confirm(root,args.run_final_confirmation,args.resume)
    else:
        validate_v2(root)
        from ..final_h.analysis import report
        report(root)


if __name__=='__main__':main()
