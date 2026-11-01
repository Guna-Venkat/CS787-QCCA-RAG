"""Explicit full opt-in, fabricated smoke defaults, atomic per-query resumption."""
import argparse
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .protocol import load,guard,authorization,gate,output,ROOT,DIRECTORIES,review,code_identity,scientific_config
from .cache import Cache,Store,atomic,digest,file_hash,lock,selection_key
from .provenance import identity,manifest,model_identity
from .datasets import load_dataset,source_identity
from .hierarchy import build,validate
from .readers import tokenizers,Reader,FixtureReader
from .embeddings import Embeddings,SyntheticEncoder
from .retrievers import retrieve,ranked,passage_scores,BM25Index
from .selectors import rank_units
from .budgets import allocate,evidence,token_counts,source_order
from .planning import plan,shared_action,pending_reason,action
from .compressors import LLMLingua2,availability,sara_guard
from .hierarchical import candidate_order
from .evaluation import evaluate,retrieval_metrics
from .generation import row
from .reporting import report,saved_rows


def sampled(examples,cfg):
    result=[]
    for name in cfg['datasets']:
        group=[e for e in examples if e['dataset']==name]
        group.sort(key=lambda e:digest([name,e['query_id'],cfg['seed']]))
        result.extend(group[:cfg['characterization_queries_per_dataset']])
    return result


class Laboratory:
    def __init__(self,cfg,examples,toks,cache_root,shared_identity,synthetic):
        self.cfg=cfg;self.examples=examples;self.toks=toks;self.root=cache_root;self.synthetic=synthetic
        self.identity=shared_identity;self.encoders={};self.hierarchies={};self.doc_units={};self.indices={};self.compressor=None
        self._selections=None
        self.core_identity={k:v for k,v in shared_identity.items() if k not in ('embedding_weight_hash','compressor_weight_hash')}
        self.docs=Cache(cache_root,'hierarchy',self.core_identity)
    @property
    def selections(self):
        if self._selections is None:
            self._selections=Cache(self.root,'selections/'+digest([self.identity.get('embedding_weight_hash'),self.identity.get('compressor_weight_hash')])[:16],self.identity)
        return self._selections
    def encoder(self,model):
        if model not in self.encoders:
            self.encoders[model]=SyntheticEncoder() if self.synthetic else Embeddings(self.cfg,model,'shared_units')
        return self.encoders[model]
    def close(self):
        for e in self.encoders.values():e.close()
        self.encoders={}
        if self.compressor is not None:
            self.compressor.model=None;self.compressor=None
            import gc
            gc.collect()
    def token_compressor(self):
        if self.compressor is None:self.compressor=LLMLingua2(self.cfg['token_compressor'])
        return self.compressor
    def units(self,example):
        guard(example['dataset'],example['split'])
        documents=example['inference']['documents']
        key=digest([(d['dataset'],d['document_id'],digest(d['text'])) for d in documents])
        if key in self.hierarchies:return self.hierarchies[key]
        all_units=[]
        for index,doc in enumerate(documents):
            dk=digest([doc,self.cfg['hierarchy'],self.cfg['models']['Qwen']])
            if dk not in self.doc_units:
                value=self.docs.get(dk)
                if value is None:
                    value=build([doc],self.toks['Qwen'],self.cfg['hierarchy']);self.docs.put(dk,value)
                self.doc_units[dk]=value
            all_units.extend(dict(u,document_order=[index]+u['document_order'][1:]) for u in self.doc_units[dk])
        validate(all_units,documents);self.hierarchies[key]=all_units
        return all_units
    def ranking(self,example,units,model):
        query=example['inference']['question'];corpus=digest([(u['unit_id'],u['hash']) for u in units])
        namespace='index/'+model.lower();index_identity={k:v for k,v in self.identity.items() if k!='compressor_weight_hash'}
        cache=Cache(self.root,namespace,index_identity)
        key=[example['dataset'],example['query_id'],digest(query),corpus,model]
        old=cache.get(key)
        if old is not None:
            by={u['unit_id']:u for u in units}
            return [dict(by[x['unit_id']],score=x['score'],rank=i+1) for i,x in enumerate(old['ranking'])],old['cost']
        if model=='BM25':
            idx_key=[corpus,'BM25_PROJECT_K1_1.5_B_.75'];value=cache.get(idx_key)
            if corpus not in self.indices:
                if value is None:
                    idx=BM25Index([u['source_text'] for u in units]);cache.put(idx_key,idx.serialize())
                else:idx=BM25Index.restore(value)
                self.indices[corpus]=idx
            started=time.perf_counter();r=ranked(units,self.indices[corpus].score(query));cost=dict(retrieval_latency=time.perf_counter()-started)
        else:r,cost=retrieve(query,units,model,self.encoder(model))
        if model!='BM25' and units and not self.synthetic:
            encoder=self.encoder(model)
            for kind in {u['unit_type'] for u in units}:
                metadata=Cache(self.root,'embeddings/'+kind.lower()+'/'+model,index_identity)
                mk=[corpus,model]
                if metadata.get(mk) is None:metadata.put(mk,dict(model=self.cfg['models'][model],
                    vector_database=str(encoder.directory/'embeddings.sqlite3'),normalization=False,pooling='explicit-window-mean-v1',
                    text_keys={u['unit_id']:encoder.key(u['source_text'],'document') for u in units if u['unit_type']==kind}))
        cache.put(key,dict(ranking=[dict(unit_id=u['unit_id'],score=u['score']) for u in r],cost=cost))
        return r,cost
    def selection(self,example,a,reuse_only=False):
        guard(example['dataset'],example['split'])
        key=selection_key(example,shared_action(a),self.cfg,{k:self.cfg['models'][k] for k in self.toks})
        old=self.selections.get(key)
        if old is not None:return old
        if reuse_only:raise RuntimeError('D must reuse the reviewed C selection; missing shared evidence, do not recompute')
        started=time.perf_counter();units=self.units(example);cost={};compress_seconds=None
        if a['granularity']=='NONE':selected=allocate([],0,self.toks)
        else:
            passages=[u for u in units if u['unit_type']=='PASSAGE']
            pool,cost=self.ranking(example,passages,a['retriever']);pool=pool[:a['pool_width']]
            cap=allocate(pool,self.cfg['candidate_context_token_ceiling'],self.toks)
            accepted=set(cap['selected_unit_ids']);pool=[p for p in pool if p['unit_id'] in accepted]
            descendants=[u for u in units if u['passage_id'] in {p['unit_id'] for p in pool}]
            if a['compressor']=='C3':
                sara_guard(a['reader'],example['dataset'])
                selected=allocate(pool[:self.cfg['sara_k']],'FULL',self.toks)
                vectors,_=self.encoder('SFR').encode([u['source_text'] for u in pool])
                selected.update(latent_vectors=vectors.tolist(),latent_pool=pool,source_support=None,source_metrics_valid=False,
                    notes='MISTRAL_ONLY_LATENT_BASELINE: top4 text plus remaining top10 latents; natural point, not a 256-token textual control')
            elif a['granularity']=='TOKEN':
                raw=evidence(pool);comp=self.token_compressor();t=time.perf_counter()
                result=comp.compress(raw,a['budget'],example['inference']['question'],self.toks);compress_seconds=time.perf_counter()-t
                selected=allocate([],'FULL',self.toks)
                selected.update(text=result['text'],token_counts=token_counts(result['text'],self.toks),selected_unit_ids=[u['unit_id'] for u in pool],
                    input_unit_ids=[u['unit_id'] for u in pool],requested_budget=a['budget'],source_support=result['source_support'],
                    source_metrics_valid=False,notes=result['notes'])
            else:
                if a['strategy']:
                    ordered=candidate_order(example['question'],pool,descendants,a['strategy'],self.encoder('SFR'),self.cfg)
                    if a['strategy']=='H4':
                        # Full head, sentence middle and token-compressed tail, all disjoint.
                        middle=self.cfg['hierarchical']['sentence_passages'];head=self.cfg['hierarchical']['full_head_passages']
                        head_units=pool[:head]
                        middle_ids={u['unit_id'] for u in pool[head:middle]}
                        middle_units=[u for u in descendants if u['unit_type']=='SENTENCE' and u['passage_id'] in middle_ids]
                        middle_units=rank_units(example['question'],middle_units,'semantic',self.encoder('SFR'))[0]
                        sub_budget='FULL' if a['budget']=='FULL' else int(a['budget']*(1-self.cfg['hierarchical']['token_tail_fraction']))
                        selected=allocate(head_units+middle_units,sub_budget,self.toks)
                        components=[dict(text=u['source_text'],order=u['document_order']) for u in selected['units']]
                        fragments=[];t=time.perf_counter()
                        for passage in pool[middle:]:
                            current='\n\n'.join(c['text'] for c in sorted(components,key=lambda c:c['order']))
                            remainder='FULL' if a['budget']=='FULL' else max(0,a['budget']-max(token_counts(current,self.toks).values()))
                            if remainder==0:break
                            tail=self.token_compressor().compress(passage['source_text'],remainder,example['question'],self.toks)
                            if not tail['text']:continue
                            components.append(dict(text=tail['text'],order=passage['document_order']))
                            fragments.append(dict(input_passage_id=passage['unit_id'],source_hash=passage['hash'],
                                output_hash=digest(tail['text']),text=tail['text'],document_order=passage['document_order'],exact_contiguous_span=False))
                        compress_seconds=time.perf_counter()-t
                        combined='\n\n'.join(c['text'] for c in sorted(components,key=lambda c:c['order']))
                        counts=token_counts(combined,self.toks)
                        if a['budget']!='FULL' and max(counts.values())>a['budget']:raise ValueError('Hierarchical token tail exceeds shared budget')
                        selected.update(text=combined,token_counts=counts,source_support=None,source_metrics_valid=False,
                            token_tail_fragments=fragments,input_unit_ids=selected['selected_unit_ids']+[f['input_passage_id'] for f in fragments],
                            selected_unit_ids=selected['selected_unit_ids']+[f['input_passage_id'] for f in fragments],
                            notes='H4: full head + selected middle sentences + per-passage LLMLingua-2 tail; all components restored to source order')
                    else:selected=allocate(ordered,a['budget'],self.toks)
                else:
                    choices=pool if a['granularity']=='PASSAGE' else [u for u in descendants if u['unit_type']==a['granularity']]
                    ordered,_=rank_units(example['question'],choices,a['selector'],self.encoder('SFR'),self.cfg['density_gamma'])
                    if a['compressor']=='C2':ordered=ordered[:self.cfg['exact_text_sentence_k']]
                    selected=allocate(ordered,a['budget'],self.toks)
            selected['raw_candidate_tokens']=token_counts(evidence(pool),self.toks)
        selected.update(retrieval_latency=cost.get('retrieval_latency'),selection_latency=time.perf_counter()-started,
                        compression_latency=compress_seconds)
        selected['candidate_pool_ids']=[p['unit_id'] for p in pool] if a['granularity']!='NONE' else []
        selected.update(requested_budget=a['budget'],actual_evidence_tokens=selected['token_counts']['Qwen'],
                        unused_budget=None if a['budget']=='FULL' else a['budget']-max(selected['token_counts'].values()))
        self.selections.put(key,selected)
        return selected


def setup_tables(cfg,examples,lab):
    datasets=[];answers=[];lengths=[];units=[];seen=set()
    for name in sorted({e['dataset'] for e in examples}):
        group=[e for e in examples if e['dataset']==name]
        datasets.append(dict(dataset=name,split=group[0]['split'],n_queries=len(group),
            candidate_corpus=cfg['datasets'][name]['corpus'],synthetic=lab.synthetic))
        for kind,count in Counter(e['answer_type'] for e in group).items():answers.append(dict(dataset=name,answer_type=kind,count=count))
        for example in group:
            current=lab.units(example)
            for doc in example['inference']['documents']:
                key=(name,doc['document_id'])
                if key in seen:continue
                seen.add(key);lengths.append(dict(dataset=name,document_id=doc['document_id'],tokens=len(lab.toks['Qwen'].encode(doc['text'],add_special_tokens=False))))
                local=[u for u in current if u['document_id']==doc['document_id']]
                counts=Counter(u['unit_type'] for u in local)
                units.extend(dict(dataset=name,document_id=doc['document_id'],unit_type=u['unit_type'],token_count=u['token_count'],units_per_document=counts[u['unit_type']]) for u in local)
    models=[]
    for name,spec in cfg['models'].items():
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(spec['id'],revision=spec['revision'],local_files_only=True);status='CACHED_SNAPSHOT'
        except (OSError,ValueError):status='MISSING_LOCAL_SNAPSHOT'
        models.append(dict(model=name,**spec,status=status,smoke_model_loaded=False,tokenizer=spec['id'],
            dtype=cfg['generation']['dtype'] if name in cfg['readers'] else 'float16' if name=='SFR' else 'float32',
            quantization=None,max_context=cfg['generation']['max_context'] if name in cfg['readers'] else None,
            max_new_tokens=cfg['generation']['max_new_tokens'] if name in cfg['readers'] else None,
            decoding='greedy' if name in cfg['readers'] else None,seed=cfg['seed'],peak_vram_measured=None))
    models.append(dict(model='LLMLingua-2',status=availability(cfg)['C1'],id=cfg['token_compressor']['id'],revision=cfg['token_compressor']['revision']))
    cache_rows=[dict(namespace=n,path=str(lab.root/n),files=sum(p.is_file() for p in (lab.root/n).rglob('*')) if (lab.root/n).exists() else 0)
        for n in ['datasets','hierarchy','embeddings','index/bm25','index/sfr','index/minilm','selections']]
    return {k:pd.DataFrame(v) for k,v in dict(dataset_summary_df=datasets,answer_type_df=answers,document_length_df=lengths,
        hierarchy_stats_df=units,model_registry_df=models,cache_status_df=cache_rows).items()}


def cache_datasets(examples,root,cache_identity):
    docs=Cache(root,'datasets/documents',cache_identity);queries=Cache(root,'datasets/queries',cache_identity);seen=set()
    for example in examples:
        for d in example['inference']['documents']:
            key=[d['dataset'],d['document_id']]
            if tuple(key) not in seen:
                if docs.get(key) is None:docs.put(key,d)
                seen.add(tuple(key))
        row=copy.deepcopy({k:v for k,v in example.items() if k not in ['inference','document_text']})
        row['inference']={k:v for k,v in example['inference'].items() if k!='documents'}
        row['document_ids']=[d['document_id'] for d in example['inference']['documents']]
        if queries.get([example['dataset'],example['query_id']]) is None:queries.put([example['dataset'],example['query_id']],row)


def load_existing(cfg,phase):
    root=output(cfg,phase);frames={}
    if (root/'figures.json').exists():figures=json.loads((root/'figures.json').read_text())
    else:figures=[]
    from .catalog import TABLES
    for name in TABLES[phase]:
        path=root/(name+'.csv')
        try:frames[name]=pd.read_csv(path) if path.exists() else pd.DataFrame()
        except pd.errors.EmptyDataError:frames[name]=pd.DataFrame()
    return frames,figures


def notebook_smoke(cfg,phase,run_smoke,run_full):
    """Default notebook runs prefer completed real results over fabricated smoke."""
    return bool(run_smoke and (run_full or not (output(cfg,phase)/'status.json').exists()))


def execute(phase,cfg=None,*,full=False,smoke=False,explicit=False,limit=3,plots=True):
    cfg=copy.deepcopy(cfg or load());guard();authorization(full,explicit,smoke,limit)
    target=(ROOT/cfg['output_root']).resolve()
    if not target.is_relative_to(ROOT/'results/week11') and not (smoke and target.is_relative_to(Path('/tmp'))):
        raise ValueError('Historical output directories are forbidden')
    if phase not in DIRECTORIES:raise ValueError('Unknown Week 11 phase')
    if not full and not smoke and phase not in ('H','I'):return load_existing(cfg,phase)
    if phase=='F' and full and not cfg['finalist_run_ids']:raise ValueError('Supply reviewed finalist_run_ids in config before full F')
    if full:gate(cfg,phase)
    if phase=='F' and full:
        existing=saved_rows(ROOT/cfg['output_root']);known={r.get('action_id') for r in existing if 'task_score' in r}
        if any(k not in known for k in cfg['finalist_run_ids']):raise ValueError('Finalist IDs must identify actually saved, manually reviewed configurations')
    if phase in ('H','I') and not smoke:
        root=ROOT/cfg['output_root'];rows=[r for r in saved_rows(root) if 'task_score' in r]
        retrieval=[dict(**{k:v for k,v in r.items() if k not in ['metrics','ranking','cost']},**metric)
                   for r in saved_rows(root) if 'metrics' in r for metric in r['metrics']]
        if not rows:return load_existing(cfg,phase)
        cfg['_retrieval_saved_rows']=retrieval
        frames,figures=report(phase,rows,cfg,output(cfg,phase))
        atomic(output(cfg,phase)/'status.json',dict(synthetic=False,executable_complete=bool(rows),completed=len(rows),expected=len(rows),
            pending=[],source_result_hash=digest(rows),generation_started=False))
        return frames,figures
    if smoke:
        from .fixtures import examples as fixtures
        examples=fixtures();available_counts=Counter(e['dataset'] for e in examples)
        examples=[e for name in sorted({e['dataset'] for e in examples}) for e in [x for x in examples if x['dataset']==name][:limit]]
        sources=dict(fabricated_fixture=digest(examples))
        if phase=='F':
            cfg['finalist_run_ids']=['SYNTHETIC_FINALIST'];cfg['finalist_configurations']={'SYNTHETIC_FINALIST':dict(granularity='SENTENCE',selector='semantic',budget=256)}
    else:
        sources=source_identity(cfg);all_examples=[e for name in cfg['datasets'] for e in load_dataset(name,cfg)]
        available_counts=Counter(e['dataset'] for e in all_examples);examples=sampled(all_examples,cfg)
    toks=tokenizers(cfg,smoke)
    models={} if smoke else {name:model_identity(cfg['models'][name],weights=True)
        for name in (['MiniLM','SFR'] if phase in ('A','B') else ['SFR']+cfg['readers'] if phase in ('C','D','E','F','G') else [])}
    if not smoke and phase in ('C','D','E','F','G'):models['MiniLM']=model_identity(cfg['models']['MiniLM'])
    if not smoke and phase=='E':
        models['SARA_checkpoint']={p.name:file_hash(p) for p in (ROOT/cfg['sara_checkpoint']).iterdir()
                                   if p.suffix in ('.json','.safetensors','.bin','.model')}
    if not smoke and phase in ('C','D','E','F','G') and cfg['token_compressor'].get('qualified'):
        receipt=ROOT/cfg['output_root']/'smoke/token_compressor_qualification.json'
        qualification=json.loads(receipt.read_text())
        if not qualification['passed'] or any(qualification['spec'][k]!=cfg['token_compressor'][k] for k in ('id','revision')):
            raise ValueError('Token compressor lacks a matching feasibility receipt')
        models['LLMLingua-2']=model_identity(cfg['token_compressor'])
    run_identity=identity(cfg,phase,sources,models,smoke)
    root=(ROOT/cfg['output_root']/'smoke'/phase/digest(run_identity)[:12]) if smoke else output(cfg,phase)
    cache_root=(root/'cache') if smoke else ROOT/cfg['cache_root']
    shared=dict(config_hash=digest(scientific_config(cfg)),code_hash=digest(code_identity()),dataset_hash=digest(sources),
                selection_models={k:cfg['models'][k] for k in ['SFR','MiniLM']},synthetic=smoke,
                tokenizer_hash='SYNTHETIC' if smoke else digest({k:model_identity(cfg['models'][k],False) for k in cfg['readers']}))
    if phase!='0':shared['embedding_weight_hash']=digest({k:models.get(k,'SYNTHETIC') for k in ['SFR','MiniLM']})
    if 'LLMLingua-2' in models:shared['compressor_weight_hash']=digest(models['LLMLingua-2'])
    with lock(root/'run.lock'):
        manifest(root/'manifest.json',cfg,run_identity,phase,models)
        cohort=dict(available_queries=dict(available_counts),sampling=cfg['query_sampling'] if not smoke else 'fabricated fixture limit',
            selected=[dict(dataset=e['dataset'],query_id=e['query_id'],document_id=e['document_id'],cluster_id=e['cluster_id']) for e in examples])
        if (root/'cohort.json').exists() and json.loads((root/'cohort.json').read_text())!=cohort:raise ValueError('Cohort identity changed')
        atomic(root/'cohort.json',cohort)
        cache_datasets(examples,cache_root,{k:v for k,v in shared.items() if k not in ('embedding_weight_hash','compressor_weight_hash')})
        lab=Laboratory(cfg,examples,toks,cache_root,shared,smoke)
        store=Store(root/'rows',run_identity)
        pending=[];selection_hashes=[]
        try:
            if phase=='0':
                setup=setup_tables(cfg,examples,lab)
                setup['dataset_summary_df']['available_queries']=setup['dataset_summary_df'].dataset.map(available_counts)
                frames,figures=report(phase,[],cfg,root,setup,smoke)
                atomic(root/'status.json',dict(synthetic=smoke,executable_complete=True,completed=len(examples),expected=len(examples),
                    pending=[],configuration_hash=digest(cfg),hierarchy_hash=digest(lab.hierarchies)))
                return frames,figures
            if phase in ('A','B'):
                for example in tqdm(examples,desc=f'W11-{phase} retrieval'):
                    units=lab.units(example)
                    for a in plan(cfg,phase,example['dataset']):
                        key=[example['dataset'],example['query_id'],a['action_id']]
                        if store.get(key) is not None:continue
                        choices=[u for u in units if u['unit_type']==a['granularity']]
                        if phase=='B':
                            enc=lab.encoder('SFR');q,_=enc.encode([example['question']],'query');pv,_=enc.encode([u['source_text'] for u in choices])
                            sv=[enc.encode([u['source_text'] for u in units if u['unit_type']=='SENTENCE' and u['passage_id']==p['unit_id']])[0] for p in choices]
                            scores=passage_scores(q[0],pv,sv,a['representation'],cfg['passage_pooling_beta']);rank=ranked(choices,scores);cost={}
                        else:rank,cost=lab.ranking(example,choices,a['retriever'])
                        metrics=retrieval_metrics(rank,example)
                        store.put(key,dict(dataset=example['dataset'],query_id=example['query_id'],cluster_id=example['cluster_id'],
                            action_id=a['action_id'],retriever=a['retriever'],evidence_granularity=a['granularity'],
                            passage_representation=a['representation'],metrics=metrics,cost=cost,
                            ranking=[u['unit_id'] for u in rank],mean_unit_tokens=float(np.mean([u['token_count'] for u in choices])) if choices else None,
                            mean_passage_tokens=float(np.mean([u['token_count'] for u in choices])) if choices and a['granularity']=='PASSAGE' else None,
                            synthetic=smoke))
                flattened=[]
                for r in store.rows():
                    flattened.extend(dict(**{k:v for k,v in r.items() if k not in ['metrics','ranking','cost']},
                        **metric,top_ids=r['ranking'][:metric['k']],retrieval_latency=r['cost'].get('retrieval_latency'),
                        document_preprocessing_original_seconds=r['cost'].get('document_preprocessing',{}).get('embedding_original_compute_seconds'),
                        document_embedding_lookup_seconds=r['cost'].get('document_preprocessing',{}).get('embedding_lookup_compute_seconds'),
                        query_embedding_original_seconds=r['cost'].get('query_embedding',{}).get('embedding_original_compute_seconds'),
                        query_embedding_lookup_seconds=r['cost'].get('query_embedding',{}).get('embedding_lookup_compute_seconds')) for metric in r['metrics'])
                expected=sum(len(plan(cfg,phase,e['dataset'])) for e in examples)
                atomic(root/'status.json',dict(synthetic=smoke,executable_complete=len(store.rows())==expected,
                    completed=len(store.rows()),expected=expected,pending=[]))
                return report(phase,flattened,cfg,root,synthetic=smoke)
            if phase in ('H','I') and smoke:
                # Read saved fabricated result shards only; NEVER trigger a generation stage.
                rows=[]
                for source_phase in ['C','D','E','F','G','A']:
                    candidates=sorted((ROOT/cfg['output_root']/'smoke'/source_phase).glob('*/rows/identity.json'),key=lambda p:p.stat().st_mtime,reverse=True)
                    for path in candidates:
                        saved=json.loads(path.read_text());source_manifest=json.loads((path.parent.parent/'manifest.json').read_text())
                        if saved['code_hash']!=run_identity['code_hash'] or scientific_config(source_manifest['configuration'])!=scientific_config(cfg):continue
                        current=Store(path.parent,saved).rows()
                        if source_phase=='A':
                            cfg['_retrieval_saved_rows']=[dict(**{k:v for k,v in r.items() if k not in ['metrics','ranking','cost']},**m) for r in current for m in r['metrics']]
                        else:rows.extend(current)
                        break
                return report(phase,rows,cfg,root,synthetic=True)
            actions=[]
            condition_status=[]
            for example in examples:
                for a in plan(cfg,phase,example['dataset']):
                    reason=pending_reason(a,cfg)
                    condition_status.append(dict(dataset=example['dataset'],action_id=a['action_id'],reader=a['reader'],
                        granularity=a['granularity'],compressor=a['compressor'],status='PENDING' if reason else 'EXECUTABLE',reason=reason))
                    if reason:
                        pending.append(reason);continue
                    actions.append((example,a))
            pd.DataFrame(condition_status).drop_duplicates().to_csv(root/'condition_status_df.csv',index=False)
            if pending and not cfg['allow_pending_conditions']:raise RuntimeError('Pending components must be qualified before this run')
            # Baselines are always explicit, same-reader no-evidence controls.
            for example in examples:
                for reader in sorted({a['reader'] for e,a in actions if e['dataset']==example['dataset']}):
                    baseline=action(cfg,reader=reader,granularity='NONE',selector='none',budget=0,pool_width=10 if phase=='E' else cfg['retrieval_width'])
                    if not any(e['query_id']==example['query_id'] and e['dataset']==example['dataset'] and a['action_id']==baseline['action_id'] for e,a in actions):actions.append((example,baseline))
            prepared={}
            for example,a in tqdm(actions,desc=f'W11-{phase} prepare shared evidence'):
                key=[example['dataset'],example['query_id'],a['action_id']]
                selection=lab.selection(example,a,reuse_only=phase=='D' and not smoke)
                prepared[tuple(key)]=selection;selection_hashes.append([key,digest(selection)])
            lab.close()  # Release ALL retrieval/compressor models before loading readers.
            receipt=root/'selection_identity.json';sel=dict(hash=digest(selection_hashes))
            if receipt.exists() and json.loads(receipt.read_text())!=sel:raise ValueError('Evidence selection hash changed; refuse resume')
            if not receipt.exists():atomic(receipt,sel)
            generation_cache=Cache(cache_root,'generation',{k:v for k,v in shared.items() if k not in ('embedding_weight_hash','compressor_weight_hash')})
            completed_total=len(store.rows())
            for reader in cfg['readers']:
                work=[(e,a) for e,a in actions if a['reader']==reader]
                work.sort(key=lambda ea:(ea[1]['compressor']=='C3',ea[1]['granularity']!='NONE',ea[0]['dataset'],ea[0]['query_id'],ea[1]['action_id']))
                model=None;sara=None;baselines={};started=time.perf_counter();times=[]
                try:
                    bar=tqdm(work,desc=f'W11-{phase} {reader}',mininterval=2.)
                    for example,a in bar:
                        key=[example['dataset'],example['query_id'],a['action_id']]
                        variant='MISTRAL_ONLY_LATENT_BASELINE' if a['compressor']=='C3' else 'PUBLIC_TEXT_READER'
                        basekey=(example['dataset'],example['query_id'],variant)
                        old=store.get(key)
                        if old is not None:
                            if a['granularity']=='NONE':baselines[basekey]=old
                            continue
                        selection=prepared[tuple(key)]
                        generation_key=dict(dataset=example['dataset'],query_id=example['query_id'],question=example['question'],
                            text_hash=digest(selection['text']),reader=cfg['models'][reader],task=example['task_type'],
                            generation=cfg['generation'],latent=digest(selection.get('latent_vectors')),
                            model_hash=digest(models.get(reader,'SYNTHETIC')),
                            sara_checkpoint_hash=digest(models.get('SARA_checkpoint')) if a['compressor']=='C3' else None)
                        result=generation_cache.get(generation_key);hit=result is not None;begin=time.perf_counter()
                        if result is None:
                            if a['compressor']=='C3' and not smoke:
                                sara_guard(reader,example['dataset'])
                                if model:model.close();model=None
                                if sara is None:
                                    from src.week10.root_cause.runtime import SaraReader
                                    sara=SaraReader(dict(checkpoint=cfg['sara_checkpoint']))
                                if a['granularity']=='NONE':raw=sara.generate_text(example['question'],'')
                                else:
                                    pool=[dict(text=u['source_text']) for u in selection['latent_pool']]
                                    raw=sara.generate(example['question'],pool,cfg['sara_k'],np.asarray(selection['latent_vectors']))
                                result=dict(prediction=raw['content'],prompt_tokens=raw['prompt_tokens'],generated_tokens=raw['generated_tokens'],generation_latency=raw['latency'],
                                    evidence_tokens_local=len(sara.tokenizer.encode(selection['text'],add_special_tokens=False)))
                            else:
                                if sara:sara.close();sara=None
                                if model is None:model=FixtureReader(cfg,reader) if smoke else Reader(cfg,reader)
                                result=model.generate(example['question'],selection['text'],example['task_type'])
                            generation_cache.put(generation_key,result)
                        baseline=dict(evaluate(result['prediction'],example),prediction=result['prediction'],reader_variant=variant) if a['granularity']=='NONE' else baselines[basekey]
                        saved=row(cfg,phase,example,a,selection,result,baseline,run_identity)
                        saved.update(strategy=a['strategy'] or 'FIXED',physical_lm_calls=0 if hit else 1,generation_cache_hit=hit,
                            model_hash=generation_key['sara_checkpoint_hash'] if a['compressor']=='C3' else generation_key['model_hash'],
                            timing_scope='Cached original selection/embedding and generation measurements; total end-to-end is unmeasured',
                            compression_ratio=selection['token_counts'][reader]/max(1,selection.get('raw_candidate_tokens',selection['token_counts'])[reader]),
                            allocation_tokens=dict(Counter({kind:sum(u['token_count'] for u in selection['units'] if u['unit_type']==kind) for kind in ['PASSAGE','SENTENCE','PROPOSITION']})))
                        saved['allocation_tokens']['TOKEN']=sum(len(toks['Qwen'].encode(f['text'],add_special_tokens=False)) for f in selection.get('token_tail_fragments',[])) if a['strategy']=='H4' else selection['token_counts']['Qwen'] if a['granularity']=='TOKEN' else 0
                        if selection.get('source_metrics_valid') is False:
                            saved.update(any_gold_recall=None,all_gold_recall=None,evidence_coverage=None,source_metrics_note='Unaligned token/latent output: input recall is not retained-output recall')
                        store.put(key,saved)
                        completed_total+=1
                        if a['granularity']=='NONE':baselines[basekey]=saved
                        times.append(time.perf_counter()-begin)
                        postfix=dict(total_queries=len(examples),completed=completed_total,remaining=len(actions)-completed_total,elapsed=round(time.perf_counter()-started),
                            recent_seconds_per_answer=round(float(np.mean(times[-20:])),3),configuration=a['action_id'])
                        if not smoke:
                            import torch
                            postfix['GPU_GiB']=round(torch.cuda.memory_allocated()/1024**3,2)
                        bar.set_postfix(postfix,refresh=False)
                finally:
                    if model:model.close()
                    if sara:sara.close()
            atomic(root/'status.json',dict(synthetic=smoke,executable_complete=len(store.rows())==len(actions),
                full_condition_surface_complete=not pending,completed=len(store.rows()),expected=len(actions),pending=sorted(set(pending)),
                selection_hash=sel['hash'],source_result_hash=digest(store.rows())))
            return report(phase,store.rows(),cfg,root,synthetic=smoke)
        finally:lab.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--phase',choices=list(DIRECTORIES),required=True)
    p.add_argument('--full',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--limit',type=int,default=3);p.add_argument('--review-reason');p.add_argument('--reviewed-config-json')
    args=p.parse_args();cfg=load(args.config)
    if args.review_reason:
        print(review(cfg,args.phase,json.loads(args.reviewed_config_json or '{}'),args.review_reason));return
    execute(args.phase,cfg,full=args.full,explicit=args.execute,smoke=args.smoke,limit=args.limit)


if __name__=='__main__':main()
