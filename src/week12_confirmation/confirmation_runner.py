"""Original scientific helpers; new role-isolated immutable storage and strict gates."""
import json,time,os
from pathlib import Path
from src.week12.protocol import ROOT,Store,digest,sha,atomic,immutable,utc
from src.week12.algorithm import choose,validate_spans,serialize
from .freeze_guard import METHODS,BUDGETS,folder,protocol,require,verify,VERSION

def ensure_k_idle():
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(Path(a).name=='run_attribution.py' for a in args if a) and '--full' in args and 'generate' in args:raise RuntimeError('Part K is still executing. Wait; do not load confirmation data or share its GPU.')

def expected_keys(xs):return {(e['dataset'],e['query_id'],m,b,'BASE_FP16') for e in xs for m in METHODS for b in BUDGETS}

def _prepare(cfg,xs,tok,lab,out,ident):
    st=Store(out/'prepared',ident)
    try:
        for i,e in enumerate(xs):
            if all(st.get(list(k)) is not None for k in expected_keys([e])):continue
            pool,sent,props,ps,ss,lineage=lab.prepare(e,'SFR')
            for m in METHODS:
                for b in BUDGETS:
                    key=[e['dataset'],e['query_id'],m,b,'BASE_FP16']
                    if st.get(key) is not None:continue
                    sel=choose(pool,sent,props,ps,ss,tok,b,m,{'Qwen':tok});validate_spans(sel['units'],e['inference']['documents']);sel.update(lineage=lineage,raw_reference_tokens=serialize(pool,tok)["evidence_tokens"])
                    st.put(key,dict(dataset=e['dataset'],query_id=e['query_id'],confirmation_group=e['cluster_id'],selection=sel))
            getattr(lab,'units_cache',{}).clear()
            print(f'W12-M prepared {i+1}/{len(xs)} queries; cached selections reused',flush=True)
    finally:lab.close()
    rows=st.rows()
    if {tuple([r['dataset'],r['query_id'],r['selection']['method'],r['selection']['budget'],'BASE_FP16']) for r in rows}!=expected_keys(xs):raise ValueError('Prepared grid incomplete or contains unexpected identities')
    atomic(out/'prepared_complete.json',dict(complete=True,logical_rows=len(rows),identity=digest(ident)))

def prepare(cfg,permit):
    require(permit);verify(cfg);ensure_k_idle()
    from .cohort_loader import load_cohort
    from src.week12.laboratory import Lab
    from transformers import AutoTokenizer
    ident=protocol(cfg);out=folder();immutable(out/'protocol.json',ident)
    immutable(out/'FULL_EXECUTION_STARTED.json',dict(time=utc(),freeze_sha=ident['freeze_sha'],explicit_label_access=True,pre_full_outcomes_observed_by_this_workflow=False,meaning='Written immediately before first authorized confirmation example access. No claim about unavailable/off-machine history.')) if not (out/'FULL_EXECUTION_STARTED.json').exists() else None
    xs=load_cohort(cfg,permit);spec=cfg['models']['Qwen'];tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True)
    _prepare(cfg,xs,tok,Lab(cfg,tok),out,ident)

def _generate(cfg,xs,tok,reader_factory,out,ident,synthetic=False):
    from src.week12.runner import evaluate_result
    from src.week11.evaluation import evaluate
    from src.week12.reader import prompt_ids,parse_label
    prep=Store(out/'prepared',ident);rows=Store(out/'rows',ident);gen=Store(out/'generation',ident);reader=None;lookup={(e['dataset'],e['query_id']):e for e in xs}
    try:
        # NONE first guarantees one no-evidence generation/query, shared across its two budget labels.
        ordered=sorted(prep.rows(),key=lambda r:(r['dataset'],r['query_id'],METHODS.index(r['selection']['method']),r['selection']['budget']))
        if len(ordered)!=len(expected_keys(xs)):raise ValueError('Run complete preparation before generation')
        for i,r in enumerate(ordered):
            e=lookup[r['dataset'],r['query_id']];s=r['selection'];key=[e['dataset'],e['query_id'],s['method'],s['budget'],'BASE_FP16']
            if rows.get(key) is not None:continue
            validate_spans(s['units'],e['inference']['documents'])
            def answer(selection):
                nonlocal reader
                k=[e['dataset'],e['query_id'],'BASE_FP16',digest(selection['evidence_ids'])];value=gen.get(k)
                if value is not None:return value,False
                if reader is None:reader=reader_factory()
                value=reader.generate(e,selection);gen.put(k,value);return value,True
            base,base_physical=answer(dict(s,text='',evidence_ids=[]));ans,physical=answer(s);base_score=evaluate(base['prediction'],e)['task_score'];result=evaluate_result(cfg,e,s,ans,dict(score=base_score,prediction=base['prediction']))
            result.update(split='SYNTHETIC' if synthetic else 'CONFIRM',synthetic=synthetic,confirmation_group=e['cluster_id'],artifact_version=VERSION,final_freeze_hash=ident['freeze_sha'],evidence_text=s['text'],evidence_token_ids=s['evidence_ids'],source_ids=s['selected_unit_ids'],source_spans=[{k:u[k] for k in ['unit_id','document_id','passage_id','start_char','end_char']} for u in s['units']],source_content_hashes=[digest(u['source_text']) for u in s['units']],source_provenance_validated=True,reader_identity=dict(cfg['models']['Qwen'],precision='BASE_FP16'),tokenizer_identity=cfg['models']['Qwen'],generation_config=dict(cfg['generation'],seed=cfg['seed']),prompt_parser_source_hash=sha(ROOT/'src/week12/reader.py'),physical_lm_calls=(int(physical)+int(base_physical)) if not synthetic else 0,conceptual_lm_calls=1)
            rows.put(key,result)
            if (i+1)%10==0:print(f'W12-M saved {i+1}/{len(ordered)} logical rows; exact-evidence answers reused',flush=True)
    except Exception as exc:
        atomic(out/'last_technical_failure.json',dict(error_type=type(exc).__name__,time=utc(),action='Resume exact same frozen configuration; no replacement or outcome-based exclusion.'))
        raise
    finally:
        if reader:reader.close()
    _complete(cfg,xs,tok,prep,rows,out,ident,synthetic)

def _complete(cfg,xs,tok,prep,rows,out,ident,synthetic=False):
    from src.week12.reader import prompt_ids,parse_label
    from src.week11.evaluation import evaluate
    expected=expected_keys(xs);saved=rows.rows();actual={(r['dataset'],r['query_id'],r['method'],r['budget'],r['reader']) for r in saved};lookup={(e['dataset'],e['query_id']):e for e in xs}
    if len(saved)!=len(expected) or actual!=expected:raise ValueError('Result grid incomplete/duplicate/unexpected')
    for r in saved:
        e=lookup[r['dataset'],r['query_id']];key=[r['dataset'],r['query_id'],r['method'],r['budget'],'BASE_FP16'];s=prep.get(key)['selection']
        if r['confirmation_group']!=e['cluster_id'] or r['reader_identity']!=dict(cfg['models']['Qwen'],precision='BASE_FP16') or r['tokenizer_identity']!=cfg['models']['Qwen']:raise ValueError('Wrong result group/model/tokenizer')
        if r['model_hash']!=digest([cfg['models']['Qwen'],'BASE_FP16',None]):raise ValueError('Actual reader model hash differs')
        if r['generation_config']!=dict(cfg['generation'],seed=cfg['seed']) or r['prompt_parser_source_hash']!=sha(ROOT/'src/week12/reader.py') or r['artifact_version']!=VERSION:raise ValueError('Generation/prompt/parser/version identity differs')
        if r['evidence_tokens']>r['budget'] or r['evidence_tokens']!=len(r['evidence_token_ids']) or r['evidence_token_ids']!=s['evidence_ids'] or r['evidence_text']!=s['text']:raise ValueError('Result evidence/budget changed')
        if r['evidence_hash']!=digest([s['evidence_ids'],s['text']]) or r['source_hash']!=digest(e['inference']):raise ValueError('Result provenance hash changed')
        validate_spans(s['units'],e['inference']['documents'])
        if r['prompt_hash']!=digest(prompt_ids(tok,e['question'],e['task_type'],s['evidence_ids'],cfg['labels'])):raise ValueError('Prompt hash differs from frozen reader')
        if e['dataset']=='SciFact' and r['prediction']!=parse_label(r['raw_prediction'],cfg['labels']):raise ValueError('Parser/compliance mismatch')
        if r['task_score']!=evaluate(r['prediction'],e)['task_score']:raise ValueError('Task scorer mismatch')
    checksums={p.name:sha(p) for p in (out/'rows').glob('*.json') if p.name!='identity.json'}
    immutable(out/'result_checksums.json',checksums)
    atomic(out/'complete.json',dict(complete=True,logical_rows=len(saved),expected=len(expected),all_methods=METHODS,budgets=BUDGETS,identity=digest(ident),freeze_sha=ident['freeze_sha'],source_and_budget_checks=True,parser_checks=True,distinct_cached_generations=len(Store(out/'generation',ident).rows()),conceptual_lm_calls=len(saved),synthetic=synthetic,real_confirmation=not synthetic))

def generate(cfg,permit):
    require(permit);verify(cfg);ensure_k_idle()
    from .cohort_loader import load_cohort
    from src.week12.reader import Reader
    from transformers import AutoTokenizer
    xs=load_cohort(cfg,permit);spec=cfg['models']['Qwen'];tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True)
    _generate(cfg,xs,tok,lambda:Reader(cfg,'BASE_FP16'),folder(),protocol(cfg))
