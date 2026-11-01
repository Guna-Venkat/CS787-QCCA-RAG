"""Version-2 TRAIN/VAL/CONFIRM membership. Audit reads metadata, never QASPER TEST."""
import csv,json,subprocess
from pathlib import Path
from collections import defaultdict
from .protocol import ROOT,root,digest,sha,immutable,atomic,utc,guard
from .data import qhash,components
DATASETS=('QASPER','HotpotQA','SciFact')

def source_paths(cfg):
    raw=Path(cfg['datasets']['QASPER']['raw_path'])
    return {'QASPER_train':raw,'QASPER_val':raw.with_name('qasper-validation.arrow'),
        'QASPER_info':raw.with_name('dataset_info.json'),
        'HotpotQA_train':ROOT/'data/week12/hotpot_train_v1.1.json','HotpotQA_val':ROOT/cfg['datasets']['HotpotQA']['path'],
        'SciFact_train':ROOT/'data/week12/scifact/claims_train.jsonl','SciFact_val':ROOT/cfg['datasets']['SciFact']['path'],
        'SciFact_corpus':ROOT/cfg['datasets']['SciFact']['corpus_path']}

def records(cfg):
    """Membership metadata only; official TEST payloads are deliberately absent."""
    from datasets import Dataset
    paths=source_paths(cfg);out={d:dict(TRAIN=[],VAL=[]) for d in DATASETS}
    for role in ('train','val'):
        for index,p in enumerate(Dataset.from_file(str(paths['QASPER_'+role]))):
            for j,question in enumerate(p['qas']['question']):
                out['QASPER'][role.upper()].append(dict(query_id=str(p['qas']['question_id'][j]),paper_id=str(p['id']),group=str(p['id']),question_hash=qhash(question),source_index=index,question_index=j))
        for r in json.loads(paths['HotpotQA_'+role].read_text()):
            h=qhash(r['question']);out['HotpotQA'][role.upper()].append(dict(query_id=str(r['_id']),paper_id=str(r['_id']),group='question:'+h,question_hash=h,documents=[digest([title,sents]) for title,sents in r['context']]))
        native=[json.loads(l) for l in paths['SciFact_'+role].read_text().splitlines() if l.strip()]
        for r in native:
            out['SciFact'][role.upper()].append(dict(query_id=str(r['id']),paper_id=str(r['id']),group='',question_hash=qhash(r['claim']),documents=list(map(str,r.get('cited_doc_ids',[])))))
    all_claims=sum(out['SciFact'].values(),[]);groups=components([dict(query_id=r['query_id'],split_documents=r['documents']) for r in all_claims])
    for r in all_claims:r['group']='cited-paper-component:'+groups[r['query_id']]
    return out

def historical(cfg):
    """Enumerate identity-bearing local artifacts. Unknown large artifacts stay explicit."""
    used={d:set() for d in DATASETS};byweek={};files=[];unread=[]
    def collect(v,seen):
        if isinstance(v,dict):
            row=v.get('row',v)
            if not isinstance(row,dict):row=v
            d=row.get('dataset','QASPER' if week<11 else None);qid=row.get('query_id',row.get('question_id'))
            if isinstance(d,str) and d in used and qid is not None:seen[d].add(str(qid))
            key=v.get('key')
            if isinstance(key,list) and len(key)>1 and isinstance(key[0],str) and key[0] in used:seen[key[0]].add(str(key[1]))
            for k in ('selected','queries','members','query_ids','cohort'):
                x=v.get(k)
                if isinstance(x,list):
                    for z in x:collect(z,seen)
                elif isinstance(x,dict):
                    for d,values in x.items():
                        if d in used and isinstance(values,list):
                            for z in values:
                                if isinstance(z,dict):collect(dict(z,dataset=d),seen)
                                else:seen[d].add(str(z))
        elif isinstance(v,list):
            for z in v:collect(z,seen)
    for week in range(12):
        seen={d:set() for d in DATASETS};base=ROOT/f'results/week{week}'
        if not base.exists():continue
        for p in sorted(base.rglob('*')):
            if not p.is_file() or p.suffix not in ('.json','.csv','.jsonl'):continue
            # W0-W10 are QASPER; exact TEST membership has a dedicated read below.
            # Skip bulky content surfaces, retain their source coverage exclusion.
            if p.stat().st_size>8*2**20:
                unread.append(str(p.relative_to(ROOT)));continue
            try:
                if p.suffix=='.json':collect(json.loads(p.read_text()),seen)
                elif p.suffix=='.jsonl':
                    for line in p.open():
                        if line.strip():collect(json.loads(line),seen)
                else:
                    for row in csv.DictReader(p.open()):collect(row,seen)
                files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
            except (ValueError,UnicodeError,csv.Error) as e:unread.append(str(p.relative_to(ROOT))+':'+type(e).__name__)
        byweek[str(week)]={d:sorted(v) for d,v in seen.items()}
        for d in used:used[d].update(seen[d])
    pilot=json.loads((ROOT/'results/week11/multigranular/v1/granularity_budget/cohort.json').read_text())['selected']
    pilotids={d:sorted({str(r['query_id']) for r in pilot if r['dataset']==d}) for d in DATASETS}
    for d in used:used[d].update(pilotids[d])
    testpath=ROOT/'results/week9/w9h_final/v2/shared/membership.jsonl'
    consumed=[json.loads(l) for l in testpath.read_text().splitlines()] if testpath.exists() else []
    # No official TEST file, labels, answers, or prediction payload is opened here.
    return dict(used={d:sorted(v) for d,v in used.items()},by_week=byweek,pilot_ids=pilotids,scanned_artifacts=files,
        large_or_unread_artifacts=unread,QASPER_historical_test_ids=sorted({str(r['question_id']) for r in consumed}),
        coverage='Conservative local artifact audit; large QASPER surfaces excluded through historical source-paper union. Unknown/off-machine runs cannot establish freshness.')

def assign_groups(rows,seed,confirm_count):
    groups=sorted({r['group'] for r in rows},key=lambda g:digest([seed,g]));chosen=set();n=0
    for g in groups:
        if n>=confirm_count:break
        chosen.add(g);n+=sum(r['group']==g for r in rows)
    return [r for r in rows if r['group'] not in chosen],[r for r in rows if r['group'] in chosen]

def check_disjoint(dataset,parts):
    for a,b in [('TRAIN','VAL'),('TRAIN','CONFIRM'),('VAL','CONFIRM')]:
        for field in ['query_id','question_hash','group']:
            if {r[field] for r in parts[a]} & {r[field] for r in parts[b]}:raise ValueError(f'{dataset}: {field} overlaps {a}/{b}')
    return True

def freeze_path(cfg):return root(cfg)/'protocol/W12_SPLIT_FREEZE.json'
def read_freeze(cfg):
    p=freeze_path(cfg)
    if not p.exists() and p.with_suffix(p.suffix+'.gz').exists():
        import gzip
        p.write_bytes(gzip.decompress(p.with_suffix(p.suffix+'.gz').read_bytes()))
    if not p.exists():raise RuntimeError('Run Week12 notebook 0: audited split freeze required before any real training/validation')
    obj=json.loads(p.read_text());payload={k:v for k,v in obj.items() if k!='split_hash'}
    if obj['split_hash']!=digest(payload) or obj['config_hash']!=digest(cfg):raise ValueError('Split freeze checksum/config changed; new output version required')
    for d,parts in obj['datasets'].items():check_disjoint(d,parts)
    for p,h in obj['source_hashes'].items():
        if sha(p)!=h:raise ValueError('Frozen dataset source changed')
    return obj

def audit(cfg):
    guard()
    if freeze_path(cfg).exists() or freeze_path(cfg).with_suffix('.json.gz').exists():return read_freeze(cfg)
    missing=[str(p) for p in source_paths(cfg).values() if not p.exists()]
    if missing:raise FileNotFoundError('Prepare official TRAIN data first: '+str(missing))
    rec=records(cfg);hist=historical(cfg);out=root(cfg)/'protocol';out.mkdir(parents=True,exist_ok=True)
    # Translate historical project integer paper indices into canonical raw TRAIN IDs.
    from datasets import Dataset
    raw=Dataset.from_file(str(source_paths(cfg)['QASPER_train']));oldpapers=set();oldvalpapers=set();project_ids=set()
    for name in ('QASPER_train.jsonl','QASPER_train_split.jsonl','QASPER_dev.jsonl'):
        p=ROOT/'Baselines/SARA-main/data/reformatted'/name
        for line in p.open():
            r=json.loads(line);pid=str(raw[int(r['example_id'])]['id']);oldpapers.add(pid);project_ids.add(str(r['id']))
            if name=='QASPER_dev.jsonl':oldvalpapers.add(pid)
    datasets={};notes={}
    for d,official in rec.items():
        allrows=official['TRAIN']+official['VAL'];used=set(hist['used'][d]);bad={r['question_hash'] for r in allrows if r['query_id'] in used};badgroups={r['group'] for r in allrows if r['query_id'] in used}
        if d=='QASPER':
            unused=[r for r in official['TRAIN'] if r['group'] not in oldpapers]
            # No hand-waving about untouched data: require enough papers for >=20 independent confirmation clusters.
            if len({r['group'] for r in unused})>=cfg['minimum_qasper_unused_papers']:
                groups=sorted({r['group'] for r in unused},key=lambda g:digest([cfg['seed'],g]));n=max(1,len(groups)//10);cg=set(groups[:n]);vg=set(groups[n:2*n]);parts={role:[r for r in unused if (r['group'] in cg if role=='CONFIRM' else r['group'] in vg if role=='VAL' else r['group'] not in cg|vg)] for role in ('TRAIN','VAL','CONFIRM')};fresh=True
            else:
                # Official validation is legitimate validation, never renamed confirmation.
                parts=dict(TRAIN=[r for r in official['TRAIN'] if r['group'] not in oldvalpapers],VAL=[r for r in official['VAL'] if r['group'] not in oldpapers and r['question_hash'] not in bad],CONFIRM=[]);fresh=False
            notes[d]=dict(fresh_confirmation_available=fresh,untouched_train_papers=len({r['group'] for r in unused}),historical_source_excluded_papers=sorted(oldpapers),historical_validation_papers_excluded_from_training=sorted(oldvalpapers),limitation='QASPER TEST consumed and permanently closed; no fabricated confirmation. Historical TRAIN reuse allowed for gradients only.')
        else:
            eligible=[r for r in official['VAL'] if r['query_id'] not in used and r['question_hash'] not in bad and (d!='SciFact' or r['group'] not in badgroups)]
            val,confirm=assign_groups(eligible,cfg['seed'],cfg['confirmation_counts'][d]);holdhash={r['question_hash'] for r in val+confirm};holdgroups={r['group'] for r in val+confirm}
            train=[r for r in official['TRAIN'] if r['question_hash'] not in holdhash and r['group'] not in holdgroups and (d!='SciFact' or r['group'] not in badgroups)]
            parts=dict(TRAIN=train,VAL=val,CONFIRM=confirm);notes[d]=dict(fresh_confirmation_available=bool(confirm),excluded_historical_ids=sorted(used),excluded_historical_groups=sorted(badgroups),official_test_labels_locally_available=False,confirmation_source='Official DEV group-disjoint held-out subset, fixed without outcomes',document_policy='SciFact cited-document components cross TRAIN/DEV; Hotpot normalized-question groups, provided corpus documents may repeat and are not label exposure')
        # All original queries participate once in TRAIN; cross-role duplicate text blocked.
        blocked={r['question_hash'] for r in parts['VAL']+parts['CONFIRM']};parts['TRAIN']=[r for r in parts['TRAIN'] if r['question_hash'] not in blocked];check_disjoint(d,parts);datasets[d]=parts
    obj=dict(schema='W12-TRAIN-VAL-CONFIRM-v2',config_hash=digest(cfg),seed=cfg['seed'],created_at=utc(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hashes={str(p):sha(p) for p in source_paths(cfg).values()},datasets=datasets,notes=notes,historical_pilot_role='HISTORICAL_DEVELOPMENT / HYPOTHESIS_GENERATION',historical_pilot_ids=hist['pilot_ids'],confirmation_evaluated=False,QASPER_TEST_permanently_closed=True,all_eligible_training=True,grouping=dict(QASPER='paper_id',HotpotQA='normalized question hash',SciFact='transitive cited-document component'),historical_manifest_hash=digest(hist))
    obj['split_hash']=digest(obj);immutable(freeze_path(cfg),obj);immutable(out/'exposure_manifest.json',hist)
    counts=[dict(dataset=d,TRAIN=len(p['TRAIN']),VAL=len(p['VAL']),CONFIRM=len(p['CONFIRM']),TRAIN_groups=len({r['group'] for r in p['TRAIN']}),VAL_groups=len({r['group'] for r in p['VAL']}),CONFIRM_groups=len({r['group'] for r in p['CONFIRM']})) for d,p in datasets.items()]
    summary=dict(counts=counts,official_counts={d:{role:len(rows) for role,rows in v.items()} for d,v in rec.items()},official_QASPER_split_papers=json.loads(source_paths(cfg)['QASPER_info'].read_text())['splits'],overlap_matrix={d:{a+'_'+b:0 for a,b in [('TRAIN','VAL'),('TRAIN','CONFIRM'),('VAL','CONFIRM')]} for d in datasets},QASPER_FRESH_CONFIRMATION_AVAILABLE=notes['QASPER']['fresh_confirmation_available'],historical_test_consumed=len(hist['QASPER_historical_test_ids']),unread_artifact_count=len(hist['large_or_unread_artifacts']),notes=notes)
    atomic(out/'split_summary.json',summary)
    import pandas as pd
    text='# Week12 data exposure audit — corrected v2 protocol\n\n'+pd.DataFrame(counts).to_markdown(index=False)+'\n\nW11 pilot = HISTORICAL_DEVELOPMENT / HYPOTHESIS_GENERATION, never final method selection. No full run executed.\n\nOfficial source counts and group overlaps:\n```json\n'+json.dumps(summary,indent=2)+'\n```\n\nHistorical per-week IDs and exact source hashes are in `qbgf/v2/protocol/exposure_manifest.json`. Large older QASPER content surfaces are not reparsed: the historical source-paper union is excluded conservatively. This is not proof about unavailable/off-machine history. QASPER TEST membership is audited from saved membership metadata only; official TEST payloads and labels were never opened. HotpotQA/SciFact official TEST labels are not locally available, so confirmation uses eligible DEV groups. Shared indexed corpus documents are distinguished from exposed labeled query groups. Raw eligible TRAIN is uncapped. Frozen confirmation is never loaded by Week12 inference/training.\n'
    (ROOT/'results/week12/W12_DATA_EXPOSURE_AUDIT.md').write_text(text);immutable(ROOT/'results/week12/W12_SPLIT_FREEZE.json',obj)
    from .statistics import registry
    immutable(out/'primary_comparisons.json',registry(cfg,llm=True,lora=True));immutable(out/'protocol_config.json',cfg)
    return obj

def membership(cfg,role):
    if role not in ('TRAIN','VAL'):raise ValueError('Week12 never loads confirmation/test examples')
    return read_freeze(cfg)['datasets'],role

def load_examples(cfg,role,dataset=None,per_dataset_limit=None):
    """No confirmation label loader exists in Week12. Match IDs BEFORE normalizing."""
    parts,role=membership(cfg,role);result=[]
    from src.week11.datasets import hotpot,scifact,read_jsonl
    from datasets import Dataset
    from .data import native_qasper
    paths=source_paths(cfg)
    for d in ([dataset] if dataset else DATASETS):
        selected=parts[d][role] if per_dataset_limit is None else parts[d][role][:per_dataset_limit]
        wanted={r['query_id']:r for r in selected}
        if d=='QASPER':
            # Native loader is restricted at construction, not filtered after parsing all labels.
            source_roles={('train' if r['group'] not in {x['paper_id'] for x in records_for_validation(cfg)} else 'val') for r in wanted.values()} if role=='VAL' else {'train'}
            xs=[]
            for source in sorted(source_roles):xs.extend(native_qasper(paths['QASPER_'+source],allowed_ids=set(wanted)))
        elif d=='HotpotQA':
            source=paths[d+('_train' if role=='TRAIN' else '_val')];rows=[r for r in json.loads(source.read_text()) if str(r['_id']) in wanted];xs=hotpot(rows,'fit' if role=='TRAIN' else 'dev')
        else:
            source=paths[d+('_train' if role=='TRAIN' else '_val')];rows=[r for r in read_jsonl(source) if str(r['id']) in wanted];xs=scifact(rows,read_jsonl(paths['SciFact_corpus']),'fit' if role=='TRAIN' else 'dev')
        if {e['query_id'] for e in xs}!=set(wanted):raise ValueError('Frozen membership missing from source: '+d)
        for e in xs:e['cluster_id']=wanted[e['query_id']]['group'];e['partition']=role;e['split']='fit' if role=='TRAIN' else 'dev';e['inference']['split']=e['split']
        result.extend(xs)
    return result

def records_for_validation(cfg):
    # Identity metadata, not labels; cached source role is deterministic.
    from datasets import Dataset
    return [dict(paper_id=p['id']) for p in Dataset.from_file(str(source_paths(cfg)['QASPER_val']))]
