"""One frozen producer per process; dry-run cannot read any TEST records."""
import argparse
import contextlib
import gc
import importlib
import json
import os
import time
from pathlib import Path
from . import adapters
from .history import historical, recovery
from .storage import Shards, pair_membership
from .materialize import ensure_shared, dump_lines, lines, record_view, claim_required
from ..final_h import protocol as p

ORDER=['D0','semantic_k4','semantic_k2','semantic_k5','semantic_k6','semantic_k8',
       'BM25_k8','D1',p.PRIMARY,p.SAFE,p.VERIFY,'original_SARA_G0']


def dry_run(root,system):
    """Import/check metadata references only. No TEST byte reads, model load or generation."""
    p.branch_guard();root=Path(root)
    registry=p.read(root/'producer_registry.json')
    p.require(system in registry['systems'],'Missing required producer')
    entry=registry['systems'][system]
    p.require(entry['producer_id']==system and entry['gpu'] == (system not in ('semantic_k4',p.VERIFY)), 'Producer metadata differs')
    p.require(set(registry['systems'])==set(p.REQUIRED),'All 12 producer contracts required')
    recovery()
    for module in ('src.week9.final_h05.adapters','src.week9.final_h05.materialize',
                   'src.week9.backend','src.week2.precompute_dev_embeddings',
                   'src.week9.senttopk_e3.runner','src.week9.utilization_f.runner'):
        importlib.import_module(module)
    from .literal import bm25_prompt
    bm25_prompt()
    with historical():pass
    checkpoint=p.ROOT/'Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42'
    for name in ('adapter_model.safetensors','sara_extras.safetensors','tokenizer.model'):
        p.require((checkpoint/name).is_file(),'Checkpoint dependency missing: '+name)
    p.require(p.read(p.G_ROOT/'final_fit/feature_schema.json')['feature_count']==32,'G2 schema differs')
    p.require(p.read(p.G_ROOT/'final_fit/threshold.json')['threshold']==.70,'G2 threshold differs')
    p.require(entry['source_hashes'] and entry['resume']=='atomic query shards; exact freeze/config/source identity','Missing provenance/resume contract')
    return dict(producer_id=system,imports_valid=True,checkpoint_references_valid=True,schema_valid=True,
        deterministic_command=entry['command'],test_records_opened=False,actual_lm_calls=0)


class World:
    def __init__(self,root,freeze,system,stack):
        import torch
        self.root=Path(root);self.freeze=freeze;self.system=system;self.stack=stack
        self.records,path=ensure_shared(root,freeze)
        self.embeddings=torch.load(path,map_location='cpu',weights_only=False)['embeddings']
        p.require(set(self.embeddings)=={str(r['question_id']) for r in self.records},'SFR/query membership differs')
        p.require(all(tuple(x.shape)==(10,4096) and torch.isfinite(x).all() for x in self.embeddings.values()),'Real finite 10x4096 SFR vectors required; no fallback')
        self.identity=dict(freeze_digest=freeze['manifest_digest'],producer=system,method=freeze['producer_contracts'][system])
        self.shards=Shards(self.root/'producer_cache'/system,self.identity)
        self.backend=None;self._scorer=None;self._static_scorer=None;self._encoder=None;self._calls=None;self._ext=None;self._sentence=None
        import yaml
        self.cfg=yaml.safe_load((p.ROOT/'configs/week9/w9f.yaml').read_text())
        self.historical_modules=None;self.controller=None;self.w8_model=None;self.w8_tokenizer=None
        self.pipeline=None;self.schema=None;self.actual_lm_calls=0
    def saved(self,system,q):
        path=self.root/'producer_cache'/system
        store=Shards(path,dict(freeze_digest=self.freeze['manifest_digest'],producer=system,method=self.freeze['producer_contracts'][system]))
        row=store.get(q);p.require(row is not None,'Upstream query incomplete: '+system+'/'+str(q));return row
    def scorer(self):
        if self._scorer is None:
            from src.week8.semantic import SemanticScorer
            self._scorer=SemanticScorer(device='cpu')
        return self._scorer
    def static_scorer(self):
        if self._static_scorer is None:
            from src.week6.models import PretrainedSemanticScorer
            self._static_scorer=PretrainedSemanticScorer(device='cpu')
        return self._static_scorer
    def encoder(self):
        if self._encoder is None:
            from ..retrieval import LocalEncoder
            self._encoder=LocalEncoder(device='cpu')
        return self._encoder
    def generator(self):
        if self.backend is None:
            p.require(self.w8_model is None,'Never load two large generator instances')
            if self.system=='D0' or self.system.startswith('semantic_k') or self.system=='BM25_k8':
                from .historical_backend import HistoricalBackend
                self.backend=HistoricalBackend(self.cfg)
                p.require(str(self.backend.model.dtype)=='torch.bfloat16','Historical first-pass precision changed')
            else:
                from ..backend import SaraBackend
                self.backend=SaraBackend(self.cfg)
                p.require(self.backend.identity()==self.freeze['model'],'Canonical revision generator identity changed')
            actual_generate=self.backend.generate
            def counted(*args,**kwargs):
                self.actual_lm_calls+=1
                return actual_generate(*args,**kwargs)
            self.backend.generate=counted
        return self.backend
    def calls(self):
        from ..caching import RowCache
        if self._calls is None:self._calls=self.stack.enter_context(RowCache(self.root/'producer_cache'/self.system/'calls.jsonl',self.identity,True))
        return self._calls
    def extractions(self):
        from ..caching import RowCache
        if self._ext is None:self._ext=self.stack.enter_context(RowCache(self.root/'producer_cache'/self.system/'extractions.jsonl',self.identity,True))
        return self._ext
    def sentence_caches(self):
        if self._sentence is None:
            from ..caching import RowCache
            from ..retrieval_cache import RetrievalCache
            folder=self.root/'sentence_cache';i=self.identity
            self._sentence=[self.stack.enter_context(RowCache(folder/'inventory.jsonl',i,True)),
                self.stack.enter_context(RowCache(folder/'scores.jsonl',i,True)),
                self.stack.enter_context(RetrievalCache(folder/'sentences.sqlite3',i,True)),
                self.stack.enter_context(RetrievalCache(folder/'queries.sqlite3',i,True))]
        return self._sentence
    def pool(self,record,d0):
        from ..runner import locked_test_pool
        from .materialize import TEST_RAW
        return locked_test_pool(self.cfg,dict(a=p.read(p.ROOT/'results/week9/frozen/w9a_frozen_config.json')),
            record_view(record,d0),dict(raw=TEST_RAW),self.encoder())
    def verifier(self):
        if self.pipeline is None:
            import joblib
            self.pipeline=joblib.load(p.G_ROOT/'final_fit/logistic_pipeline.joblib')
            self.schema=p.read(p.G_ROOT/'final_fit/feature_schema.json')['feature_columns']
        return self.pipeline,self.schema
    def w8(self,record,d0):
        if self.historical_modules is None:
            p.require(self.backend is None,'W8 process must not retain canonical Mistral')
            self.historical_modules=self.stack.enter_context(historical())
            import src
            sara=str(p.ROOT/'Baselines/SARA-main/src')
            if sara not in src.__path__:src.__path__.append(sara)
            from src.model.loader import load_sara_for_eval
            self.w8_model,self.w8_tokenizer=load_sara_for_eval(str(p.ROOT/self.cfg['checkpoint']),device='cuda')
            actual_generate=self.w8_model.generate
            def counted(*args,**kwargs):
                self.actual_lm_calls+=1
                return actual_generate(*args,**kwargs)
            self.w8_model.generate=counted
            generation,_,_=self.historical_modules
            self.controller=generation.FrozenRuntimeController(p.read(p.ROOT/'results/week8/w8e_controller/frozen_controller.json'))
            _=self.controller.scorer.encoder
        return adapters.w8(record,d0,self.embeddings[str(record['question_id'])],self.w8_model,self.w8_tokenizer,
            self.controller,self.scorer(),self.historical_modules)
    def close(self):
        self.backend=None;self.w8_model=None;self.controller=None;self._encoder=None;self._scorer=None;gc.collect()
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()


def dispatch(world,system,record):
    q=str(record['question_id'])
    if system=='D0':return adapters.static(record,world.embeddings[q],world.generator(),world.static_scorer(),system)
    d0=world.saved('D0',q)
    if system=='semantic_k4':return dict(d0,system=system,alias_of='D0',conceptual_lm_calls=1)
    if system.startswith('semantic_k') or system=='BM25_k8':
        return adapters.static(record,world.embeddings[q],world.generator(),world.static_scorer(),system,d0)
    if system=='D1':
        return adapters.raw_intervention(record,d0,world.pool(record,d0),world.encoder(),world.generator(),world.calls(),
            world.embeddings,world.cfg,world.extractions())
    if system==p.PRIMARY:
        return adapters.primary(record,d0,world.saved('D1',q),world.generator(),world.calls(),world.embeddings,
            world.encoder(),world.freeze['encoder'],world.sentence_caches())
    if system==p.SAFE:
        return adapters.safe(record,d0,world.saved(p.PRIMARY,q),world.generator(),world.calls(),world.embeddings)
    if system==p.VERIFY:
        return adapters.verify(d0,world.saved(p.PRIMARY,q),*world.verifier())
    p.require(system=='original_SARA_G0','Unknown producer; fallback forbidden')
    return world.w8(record,d0)


def execute(root,system):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from .readiness import validate_v2
    freeze=validate_v2(root);claim_required(root,freeze)
    root=Path(root);entry=p.read(root/'producer_registry.json')['systems'][system]
    path=p.resolve(entry['result_path']);receipt_path=p.resolve(entry['receipt_path'])
    if receipt_path.exists():
        r=p.read(receipt_path)
        p.require(r['freeze_digest']==freeze['manifest_digest'] and r['predictions_sha256']==p.file_hash(path), 'Completed producer output changed')
        return
    import torch,random
    import numpy as np
    torch.set_num_threads(4);torch.manual_seed(42);random.seed(42);np.random.seed(42)
    with contextlib.ExitStack() as stack:
        world=World(root,freeze,system,stack)
        try:
            result=[];started=time.perf_counter();cache_hits=0
            for index,record in enumerate(world.records,1):
                q=str(record['question_id']);row=world.shards.get(q)
                if row is None:
                    before=world.actual_lm_calls
                    row=dispatch(world,system,record)
                    row['origin_physical_lm_calls']=world.actual_lm_calls-before
                    if 'row_content_hash' in row:
                        row.pop('row_content_hash');row['row_content_hash']=p.digest(row)
                    for field in ('evidence_tokens','retrieval_query_time','selection_latency','extraction_latency','revision_latency','verifier_latency','measured_end_to_end_latency'):
                        row.setdefault(field,None)
                    row=world.shards.put(q,row)
                else:cache_hits+=1
                result.append(row)
                if index%10==0:print(f'H2 {system} saved {index}/{len(world.records)}; elapsed={time.perf_counter()-started:.0f}s',flush=True)
            expected=[dict(question_id=str(r['question_id']),paper_id=str(r['paper_id']),split='test') for r in world.records]
            pair_membership(result,expected);dump_lines(path,result)
            spec=p.read(root/'test_materialization_spec.json')['systems'][system]
            p.write_once(receipt_path,dict(complete=True,system=system,split='test',freeze_digest=freeze['manifest_digest'],
                method_config=spec['method_config'],producer_source_hashes=spec['producer_source_hashes'],
                predictions_sha256=p.file_hash(path),membership_hash=p.digest(expected),
                actual_new_lm_calls_this_invocation=world.actual_lm_calls,completed_query_shard_hits=cache_hits,
                conceptual_costs_preserved=True,actual_new_runs='Query shards preserve conceptual costs on cache hits'))
        finally:world.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--system',required=True,choices=ORDER)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--dry-run',action='store_true');mode.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if args.dry_run:print(json.dumps(dry_run(p.resolve(args.root),args.system)))
    else:execute(p.resolve(args.root),args.system)


if __name__=='__main__':main()
