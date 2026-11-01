"""TEST I/O is reachable only after an explicit frozen H2 claim, never in H0.5."""
import gc
import json
import os
import time
from pathlib import Path
from .storage import atomic_bytes, atomic_json, Shards, membership
from ..final_h import protocol as p

TEST_FORMATTED = p.ROOT/'Baselines/SARA-main/data/reformatted/QASPER_test.jsonl'
TEST_RAW = Path('/home/gunavenkat/.cache/huggingface/datasets/allenai___qasper/qasper/0.3.0/fdc9d8214fbab5dd782958601db4d678e6934a54/qasper-test.arrow')


def lines(path):
    with Path(path).open() as f:return [json.loads(x) for x in f if x.strip()]


def dump_lines(path, rows):
    atomic_bytes(path, ''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows).encode())


def claim_required(root, freeze):
    claim=Path(root)/'h2_claim.json'
    p.require(claim.is_file() and p.read(claim)['freeze_digest']==freeze['manifest_digest'], 'Real TEST access requires an explicit matching H2 claim')


def ensure_shared(root, freeze):
    claim_required(root,freeze)
    root=Path(root); shared=root/'shared';shared.mkdir(exist_ok=True)
    # Hash TEST input bytes only inside H2. These identities are immutable on resume.
    hashes={str(path):p.file_hash(path) for path in (TEST_FORMATTED,TEST_RAW)}
    p.write_once(shared/'input_identity.json',dict(freeze_digest=freeze['manifest_digest'],hashes=hashes))
    retrieval=shared/'retrieval.jsonl'
    if not retrieval.exists():
        from src.week2.precompute_dev_embeddings import retrieve_dev_passages
        original_cwd=Path.cwd()
        try:
            os.chdir(shared)  # Original retriever's incidental outputs/cache stays under v2.
            records=retrieve_dev_passages(TEST_FORMATTED,chunk_size=256,num_passages=10)
        finally:os.chdir(original_cwd)
        originals={str(r['id']):r for r in lines(TEST_FORMATTED)}
        for r in records:r['source_context']=originals[str(r['question_id'])]['context']
        dump_lines(retrieval,records)
        p.write_once(shared/'retrieval_identity.json',dict(sha256=p.file_hash(retrieval),input_hashes=hashes))
    identity=shared/'retrieval_identity.json'
    # Crash after publication but before seal is recoverable from the exact atomic file.
    if not identity.exists():p.write_once(identity,dict(sha256=p.file_hash(retrieval),input_hashes=hashes))
    p.require(p.read(identity)['sha256']==p.file_hash(retrieval),'Shared retrieval changed on resume')
    records=lines(retrieval)
    ids=[dict(question_id=str(r['question_id']),paper_id=str(r['paper_id']),split='test') for r in records]
    membership(ids)
    for name, rows in [('membership.jsonl',ids),('labels.jsonl',[dict(question_id=str(r['question_id']),paper_id=str(r['paper_id']),
            evaluation=dict(answers=r['gold_answers'],answers_reformatted=r['gold_answers_reformatted'])) for r in records])]:
        path=shared/name
        expected=''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows).encode()
        if path.exists():p.require(path.read_bytes()==expected,'Shared labels/membership changed on resume')
        else:atomic_bytes(path,expected)
    embeddings_path=shared/'sfr.pt'
    if not embeddings_path.exists():build_sfr(records,embeddings_path,freeze)
    p.write_once(shared/'sfr_identity.json',dict(sha256=p.file_hash(embeddings_path),freeze_digest=freeze['manifest_digest']))
    p.require(p.read(shared/'sfr_identity.json')['sha256']==p.file_hash(embeddings_path),'SFR cache changed')
    return records,embeddings_path


def build_sfr(records,path,freeze):
    """Original Week-5 BF16 SFR encoding, batch16; atomic text shards enable resume."""
    import torch
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download
    snapshot=snapshot_download('Salesforce/SFR-Embedding-Mistral',local_files_only=True)
    p.require(Path(snapshot).name==freeze['sfr']['revision'],'SFR revision changed')
    cache=Shards(path.parent/'sfr_vectors',dict(freeze_digest=freeze['manifest_digest'],model='Salesforce/SFR-Embedding-Mistral',
        revision=freeze['sfr']['revision'],dtype='bfloat16',normalization=False,batch_size=16))
    texts={p.digest(x['text']):x['text'] for r in records for x in r['passages']}
    pending=[h for h in texts if cache.get(h) is None]
    if pending:
        model=SentenceTransformer(snapshot,device='cuda',local_files_only=True,model_kwargs=dict(torch_dtype=torch.bfloat16))
        try:
            for start in range(0,len(pending),16):
                keys=pending[start:start+16];begin=time.perf_counter()
                values=model.encode([texts[h] for h in keys],batch_size=len(keys),convert_to_tensor=True,show_progress_bar=False,device='cuda').detach().to('cpu',dtype=torch.bfloat16)
                elapsed=time.perf_counter()-begin
                for key,vector in zip(keys,values):cache.put(key,dict(vector=vector.float().tolist(),origin_seconds=elapsed/len(keys)))
                print(f'H2 SFR encoded {min(start+16,len(pending))}/{len(pending)} pending unique passages',flush=True)
        finally:
            del model;gc.collect();torch.cuda.empty_cache()
    matrices={str(r['question_id']):torch.tensor([cache.get(p.digest(x['text']))['vector'] for x in r['passages']],dtype=torch.bfloat16) for r in records}
    temp=path.with_suffix('.tmp');torch.save(dict(embeddings=matrices,metadata=dict(model='Salesforce/SFR-Embedding-Mistral',dtype='bfloat16',unique_texts=len(texts),
        unique_origin_seconds=sum(cache.get(h)['origin_seconds'] for h in texts),cost_scope='Reusable offline passage encoding, excluded from query-time LM prompt totals')),temp)
    temp.replace(path)


def record_view(record, d0):
    c=d0['canonical']; clean=dict(question_id=str(record['question_id']),paper_id=str(record['paper_id']),
        question=str(record['question']).strip(),initial_draft=d0['prediction'],
        passages=[{k:x[k] for k in ('passage_id','rank','score','text')} for x in record['passages'][:10]],
        selected_full_indices=c['selected_full_indices'],compressed_indices=c['compressed_indices'])
    return dict(inference=clean,canonical=c,original_passages=record['source_context'],
        evaluation=dict(answers=record['gold_answers'],answers_reformatted=record['gold_answers_reformatted']))
