"""Explicit dataset preparation only; never trains or evaluates a model."""
from pathlib import Path
import sys,argparse,json,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.week12.protocol import guard,sha,immutable

def main():
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');p.add_argument('--hotpot-path');p.add_argument('--scifact-path');args=p.parse_args();guard();out=ROOT/'data/week12';out.mkdir(parents=True,exist_ok=True);sources={}
    for name,provided,target in [('HotpotQA',args.hotpot_path,out/'hotpot_train_v1.1.json'),('SciFact',args.scifact_path,out/'scifact/claims_train.jsonl')]:
        if target.exists():sources[name]=dict(path=str(target),sha256=sha(target));continue
        target.parent.mkdir(parents=True,exist_ok=True)
        if provided:shutil.copyfile(provided,target)
        elif args.download:
            if name=='HotpotQA':
                from datasets import load_dataset
                ds=load_dataset('hotpotqa/hotpot_qa','distractor',split='train');rows=[]
                for r in ds:
                    context=r['context'];facts=r['supporting_facts'];rows.append(dict(_id=r['id'],question=r['question'],answer=r['answer'],context=list(zip(context['title'],context['sentences'])),supporting_facts=list(zip(facts['title'],facts['sent_id'])),type=r['type']))
                target.write_text(json.dumps(rows))
            else:
                import urllib.request,tarfile,io
                raw=urllib.request.urlopen('https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz',timeout=120).read()
                with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
                    members=[m for m in tar.getmembers() if m.name.endswith('/claims_train.jsonl')];assert len(members)==1;target.write_bytes(tar.extractfile(members[0]).read())
        else:print(name,'missing: provide --hotpot-path / --scifact-path or explicit --download');continue
        sources[name]=dict(path=str(target),sha256=sha(target))

    if len(sources)==2:immutable(out/'source_lock.json',sources)
    print(sources)
if __name__=='__main__':main()
