"""Explicit manual downloads only. No model or experimental run is started."""
import argparse
import io
import json
from pathlib import Path
import sys
import tarfile
import urllib.request
import urllib.error

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.week11.protocol import load,guard
from src.week11.cache import atomic,file_hash

SOURCES={
 'HotpotQA':'https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json',
 'SciFact':'https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz'}
HOTPOT_REVISION='1908d6afbbead072334abe2965f91bd2709910ab'
HOTPOT_FALLBACK=('https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/'+HOTPOT_REVISION+
                 '/distractor/validation-00000-of-00001.parquet?download=true')


def download(url,target):
    if target.exists():return
    print('Downloading '+target.name+'...',flush=True)
    target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix(target.suffix+'.partial')
    # Restart only this small data download after interruption; never touch results.
    with urllib.request.urlopen(url,timeout=120) as source,temporary.open('wb') as output:
        while True:
            block=source.read(1024*1024)
            if not block:break
            output.write(block)
    temporary.replace(target)


def original_hotpot_rows(rows):
    """Lossless field-layout conversion; preserve all source strings and order."""
    output=[]
    for row in rows:
        context=row['context'];support=row['supporting_facts']
        value={k:row[k] for k in ['question','answer','type','level']}
        value.update(_id=row['id'],context=[list(p) for p in zip(context['title'],context['sentences'],strict=True)],
                     supporting_facts=[list(p) for p in zip(support['title'],support['sent_id'],strict=True)])
        output.append(value)
    return output


def prepare_hotpot(target):
    receipt=target.parent/'hotpot_source.json'
    if target.exists():return
    try:
        download(SOURCES['HotpotQA'],target)
        atomic(receipt,dict(source_url=SOURCES['HotpotQA'],conversion=None))
    except (urllib.error.URLError,TimeoutError) as exc:
        print('Original HotpotQA host unavailable; using pinned hotpotqa/hotpot_qa '
              'distractor/validation. This fallback is recorded in source_lock.json.',flush=True)
        parquet=target.parent/'hotpot_dev_distractor_hf.parquet'
        download(HOTPOT_FALLBACK,parquet)
        import pyarrow.parquet as pq
        rows=original_hotpot_rows(pq.read_table(parquet).to_pylist())
        if len(rows)!=7405 or len({r['_id'] for r in rows})!=7405:
            raise ValueError('Pinned HotpotQA distractor validation must contain 7405 unique questions')
        temporary=target.with_suffix(target.suffix+'.partial')
        temporary.write_text(json.dumps(rows,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
        temporary.replace(target)
        atomic(receipt,dict(source_url=HOTPOT_FALLBACK,repository='hotpotqa/hotpot_qa',revision=HOTPOT_REVISION,
            subset='distractor',split='validation',download_sha256=file_hash(parquet),
            original_url=SOURCES['HotpotQA'],fallback_reason=str(exc),
            conversion='id to _id; nested column lists to native context/supporting_facts pairs; strings and row order unchanged',
            converted_sha256=file_hash(target),n_queries=len(rows)))


def validate_files(cfg):
    """Check input structure only; never run inference or score answers."""
    hotpot=json.loads((ROOT/cfg['datasets']['HotpotQA']['path']).read_text())
    if len(hotpot)!=7405 or len({r['_id'] for r in hotpot})!=7405:raise ValueError('Incomplete HotpotQA dev distractor')
    for row in hotpot:
        if not all(k in row for k in ['question','answer','context','supporting_facts']):raise ValueError('HotpotQA schema mismatch')
    counts={'HotpotQA_dev_distractor':len(hotpot)}
    for key in ['path','corpus_path']:
        path=ROOT/cfg['datasets']['SciFact'][key]
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        required=['id','claim','evidence'] if key=='path' else ['doc_id','title','abstract']
        if not rows or any(not all(k in r for k in required) for r in rows):raise ValueError('SciFact schema mismatch: '+str(path))
        counts['SciFact_dev_claims' if key=='path' else 'SciFact_corpus_abstracts']=len(rows)
    for key in ['path','raw_path']:
        if not (ROOT/cfg['datasets']['QASPER'][key]).is_file():raise FileNotFoundError('Existing QASPER DEV input missing')
    counts['QASPER_project_dev']=sum(bool(line.strip()) for line in (ROOT/cfg['datasets']['QASPER']['path']).read_text().splitlines())
    return counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');a=p.parse_args();guard();cfg=load()
    lock_path=ROOT/'data/week11/source_lock.json'
    if not a.download:raise ValueError('Explicit --download required; notebooks never download datasets automatically')
    if lock_path.exists():
        saved=json.loads(lock_path.read_text())
        for path,sha in saved['files'].items():
            if file_hash(ROOT/path)!=sha:raise ValueError('Locked dataset bytes changed')
        print('Locked development datasets already available; no download.',validate_files(cfg));return
    prepare_hotpot(ROOT/cfg['datasets']['HotpotQA']['path'])
    archive=ROOT/'data/week11/scifact_release.tar.gz';download(SOURCES['SciFact'],archive)
    with tarfile.open(archive) as tar:
        for name,key in [('corpus.jsonl','corpus_path'),('claims_dev.jsonl','path')]:
            target=ROOT/cfg['datasets']['SciFact'][key]
            if target.exists():continue
            member=next(m for m in tar.getmembers() if Path(m.name).name==name and m.isfile())
            target.parent.mkdir(parents=True,exist_ok=True)
            stream=tar.extractfile(member)
            temporary=target.with_suffix(target.suffix+'.partial')
            with temporary.open('wb') as output:
                while True:
                    block=stream.read(1024*1024)
                    if not block:break
                    output.write(block)
            temporary.replace(target)
    counts=validate_files(cfg)
    files={str((ROOT/spec[k]).relative_to(ROOT)):file_hash(ROOT/spec[k])
           for name,spec in cfg['datasets'].items() if name!='QASPER' for k in ['path','corpus_path'] if k in spec}
    hotpot_receipt=ROOT/'data/week11/hotpot_source.json'
    provenance=json.loads(hotpot_receipt.read_text()) if hotpot_receipt.exists() else dict(source='existing local JSON; origin not independently certified')
    atomic(lock_path,dict(sources=SOURCES,hotpot_provenance=provenance,counts=counts,files=files,archive_sha256=file_hash(archive),
        note='First downloaded bytes frozen before outcomes; mutable release URL is not an immutable dataset revision'))
    print('Development datasets saved and SHA256-locked. No QASPER TEST or experiments accessed.')
    print(counts)


if __name__=='__main__':main()
