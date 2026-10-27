"""W9-H stages. Default H0 cannot run producers, read TEST or compute outcomes."""
import argparse
import json
import subprocess
from pathlib import Path
from . import protocol as p
from .audit import run as audit


def materialize(root):
    """H1 is a metadata-only plan; producer execution is part of explicit H2."""
    m = p.validate_freeze(root, need_ready=False)
    protocol = p.read(Path(root)/'protocol.json')
    plan = dict(schema='W9-H-H1-plan-v1', freeze_digest=m['manifest_digest'],
        ready=m['h0_passed'], blockers=m['blockers'], systems=p.REQUIRED,
        materialization_spec=protocol['materialization_spec'],
        test_data_opened=False, generation_started=False,
        note='Execute only pre-frozen producer commands inside explicit one-shot H2. No shell command strings.')
    p.write_once(Path(root)/'h1_materialization_plan.json', plan)
    return plan


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_w9_rows(groups, freeze):
    """Check exported primary identity, exact-source evidence and controlled ablations."""
    from ..senttopk_e3.sentences import evidence, validate
    maps = {s: {r['question_id']: r for r in groups[s]} for s in (p.PRIMARY, p.SAFE, p.VERIFY, 'D1')}
    for q, row in maps[p.PRIMARY].items():
        p.require(row['sentence_k'] == 3 and row['coarse_evidence_k'] == 2, 'W9 evidence budget changed')
        p.require(row['model'] == freeze['model'] and row['decoding'] == dict(do_sample=False, max_new_tokens=128), 'W9 generator/decoding changed')
        selected = row['selected_sentences']
        raw = maps['D1'][q]
        p.require(row['selected_chunk_ids'] == raw['selected_chunk_ids'], 'Primary/D1 coarse source IDs differ')
        expected_hashes = {c['chunk_id']: p.digest(c['text']) for c in raw['source_chunks']}
        p.require(row['source_chunk_hashes'] == expected_hashes, 'Primary/D1 source content hashes differ')
        validate(selected, raw['source_chunks'], row['paper_id'])
        p.require(len(selected) <= 3 and row['evidence_text'] == evidence(selected), 'Altered E3 evidence')
        p.require(row['prompt_template_hash'] == freeze['method']['prompts']['F0']['prompt_hash'], 'F0 prompt changed')
        order = sorted(selected, key=lambda s: (s['document_char_start'], s['original_chunk_rank'], s['sentence_order'], s['sentence_id']))
        p.require(selected == order and len({s['sentence_id'] for s in selected}) == len(selected), 'Source presentation order or uniqueness changed')
        p.require(all(p.digest(s['text']) == s['text_hash'] for s in selected), 'Sentence hash changed')
        p.require(row.get('source_support_validation_rate') == 1.0 if selected else row.get('fail_closed') and row['prediction'] == row['initial_draft'], 'Unvalidated source evidence')
        safe = maps[p.SAFE][q]
        p.require(safe['prompt_template_hash'] == freeze['method']['prompts']['F2-A']['prompt_hash'], 'F2-A prompt changed')
        for other in (safe, maps[p.VERIFY][q]):
            for key in ('selected_chunk_ids', 'source_chunk_hashes', 'selected_sentence_ids', 'evidence_text'):
                p.require(other[key] == row[key], 'Controlled ablations used different evidence: ' + key)


def confirm(root, *, allow_final=False, resume=False):
    # Authorization and complete H0 checked BEFORE any TEST artifact is opened.
    p.require(allow_final, 'Final confirmation disabled; explicit --run-final-confirmation required')
    root = Path(root); freeze = p.validate_freeze(root)
    protocol = p.read(root/'protocol.json'); spec = protocol['materialization_spec']
    p.require(spec is not None, 'Missing frozen TEST materialization spec')
    p.require(not (root/'h2_seal.json').exists(), 'One-shot confirmation already sealed; use H3 only')
    claim = root/'h2_claim.json'
    if claim.exists():
        p.require(resume, 'Confirmation already claimed; preserve the run. Resume only the identical frozen computation explicitly.')
        p.require(p.read(claim)['freeze_digest'] == freeze['manifest_digest'], 'Cannot resume a different protocol')
    else:
        p.write_once(claim, dict(started_utc=p.timestamp(), freeze_digest=freeze['manifest_digest'], split='test', tuning_allowed=False))
    # Commands must be frozen argv lists invoking hashed local producer scripts.
    for name, source in spec['systems'].items():
        p.verify_hashes(source['producer_source_hashes'])
        command = source['command']
        p.require(isinstance(command, list) and all(isinstance(x, str) for x in command), 'Producer command must be an argv list')
        scripts = {str(p.resolve(x)) for x in command if x.endswith('.py')}
        hashed = {str(p.resolve(x)) for x in source['producer_source_hashes']}
        p.require(bool(scripts & hashed), 'Producer command must invoke a frozen local script')
        subprocess.run(command, cwd=p.ROOT, check=True)
    membership = read_jsonl(p.resolve(spec['membership_path']))
    keys = [(str(r['question_id']), str(r['paper_id'])) for r in membership]
    p.require(len(keys) == len(set(keys)) and len({q for q, _ in keys}) == len(keys) and keys, 'Invalid final membership')
    labels_raw = read_jsonl(p.resolve(spec['labels_path']))
    labels = {str(r['question_id']): r for r in labels_raw}
    p.require(len(labels) == len(labels_raw) and {(q, str(r['paper_id'])) for q, r in labels.items()} == set(keys), 'Labels/membership mismatch')
    from .analysis import evaluate, verify_post_generation_costs
    groups = {}; receipts = {}
    for name, source in spec['systems'].items():
        receipt_path = p.resolve(source['receipt_path']); receipt = p.read(receipt_path)
        predictions_path = p.resolve(source['predictions_path'])
        p.require(receipt['split'] == 'test' and receipt['system'] == name and receipt['complete'], 'Invalid producer receipt')
        p.require(receipt['freeze_digest'] == freeze['manifest_digest'], 'Producer did not use pretest freeze')
        p.require(receipt['method_config'] == source['method_config'], 'Producer changed its frozen method')
        p.require(receipt['producer_source_hashes'] == source['producer_source_hashes'], 'Producer source lineage differs')
        p.require(receipt['predictions_sha256'] == p.file_hash(predictions_path), 'Unsealed producer predictions')
        raw = read_jsonl(predictions_path)
        p.require(all(r.get('system') == name for r in raw), 'Mislabelled confirmation system')
        groups[name] = evaluate(raw, labels, 'test')
        receipts[str(receipt_path)] = p.file_hash(receipt_path)
        receipts[str(predictions_path)] = p.file_hash(predictions_path)
    p.require(set(p.REQUIRED) <= groups.keys(), 'Required baseline cannot be dropped')
    validate_w9_rows(groups, freeze)
    d0 = {r['question_id']: r for r in groups['D0']}
    for group in groups.values():
        p.require(all(r['initial_draft'] == d0[r['question_id']]['prediction'] for r in group), 'Different canonical D0 references')
    verify_post_generation_costs(groups[p.PRIMARY], groups[p.VERIFY])
    # Keep common gold evaluation only in sealed output, never in inference payloads.
    rows = [dict(r, evaluation=labels[r['question_id']]['evaluation']) for g in groups.values() for r in g]
    p.write_once(root/'sealed_predictions.json', rows)
    receipts[str(p.resolve(spec['labels_path']))] = p.file_hash(p.resolve(spec['labels_path']))
    receipts[str(p.resolve(spec['membership_path']))] = p.file_hash(p.resolve(spec['membership_path']))
    receipts[str(root/'sealed_predictions.json')] = p.file_hash(root/'sealed_predictions.json')
    p.validate_freeze(root)
    p.write_once(root/'h2_seal.json', dict(complete=True, completed_utc=p.timestamp(),
        freeze_digest=freeze['manifest_digest'], split='test', systems=sorted(groups),
        queries=len(keys), artifact_hashes=receipts, one_shot=True))
    return root/'h2_seal.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/week9/w9h_final.json')
    parser.add_argument('--stage', choices=('h0', 'h1', 'h2', 'h3'), default='h0')
    parser.add_argument('--run-final-confirmation', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    c = p.load_config(p.resolve(args.config)); root = p.resolve(c['output_root'])
    if args.stage == 'h0':
        audit(p.resolve(args.config))
    elif args.stage == 'h1':
        print(json.dumps(materialize(root), indent=2))
    elif args.stage == 'h2':
        confirm(root, allow_final=args.run_final_confirmation, resume=args.resume)
    else:
        from .analysis import report
        print(json.dumps(report(root), indent=2))


if __name__ == '__main__':
    main()
