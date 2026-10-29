"""Separate QLoRA intervention; tiny CPU smoke is explicitly synthetic."""
from pathlib import Path
import json,copy
from .protocol import ROOT,root,identity,immutable,atomic,digest,sha

def tiny_smoke(cfg):
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from peft import LoraConfig,get_peft_model
    torch.manual_seed(42);torch.set_num_threads(2)
    conf=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=128)
    model=get_peft_model(Qwen3ForCausalLM(conf),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    old={n:p.detach().clone() for n,p in model.named_parameters()};opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01);losses=[]
    for _ in range(2):
        ids=torch.tensor([[1,2,3,4,5,6]]);labels=ids.clone();labels[:,:3]=-100;loss=model(input_ids=ids,labels=labels).loss;loss.backward();opt.step();opt.zero_grad();losses.append(float(loss.detach()))
    base_unchanged=all(torch.equal(old[n],p) for n,p in model.named_parameters() if not p.requires_grad);adapter_changed=any(not torch.equal(old[n],p) for n,p in model.named_parameters() if p.requires_grad)
    assert base_unchanged and adapter_changed
    result=dict(synthetic=True,real_Qwen_weights_loaded=False,steps=2,losses=losses,base_unchanged=base_unchanged,adapter_changed=adapter_changed,full_training_started=False)
    atomic(root(cfg)/'smoke'/'tiny_lora.json',result);return result

def training_groups(examples,dataset):
    """Normalized duplicate Hotpot questions share a split group, irrespective of ID."""
    from .data import qhash
    for example in examples:
        if dataset=='HotpotQA':example['cluster_id']='question:'+qhash(example['question'])
    return examples

def corpus_sources(cfg):
    if cfg.get('split_protocol'):
        from .splits import load_examples
        return [dict(example=e,partition='train' if role=='TRAIN' else 'validation') for role in ('TRAIN','VAL') for e in load_examples(cfg,role)]
    from .data import development_examples,qhash
    from src.week11.datasets import hotpot,scifact,read_jsonl
    from src.week11.qasper_adapter import load as qload
    manifests=json.loads((root(cfg)/'protocol'/'cohorts.json').read_text());records=[]
    # Exclude ALL historical project DEV papers from QASPER training, not only sampled 50.
    olddev=[json.loads(x) for x in (ROOT/cfg['datasets']['QASPER']['path']).read_text().splitlines() if x.strip()];oldpapers={str(r['example_id']) for r in olddev}
    dev=development_examples(cfg);blocked_q={qhash(e['question']) for e in dev}
    for group in manifests['confirmation'].values():blocked_q.update(x['question_hash'] for x in group)
    for dataset,spec in manifests['training'].items():
        p=Path(spec['path'])
        if not p.exists():raise FileNotFoundError(f'{dataset} TRAIN missing: {p}; run scripts/week12/prepare_training_data.py explicitly')
        if dataset=='QASPER':
            ds=copy.deepcopy(cfg['datasets']['QASPER']);ds.update(path=str(p),split='fit');xs=qload(ds);xs=[e for e in xs if e['paper_id'] not in oldpapers]
        elif dataset=='HotpotQA':xs=hotpot(json.loads(p.read_text()),'fit')
        else:
            corpus=read_jsonl(ROOT/cfg['datasets']['SciFact']['corpus_path']);native=read_jsonl(p);xs=scifact(native,corpus,'fit')
            # Block all papers cited by development or reserved confirmation claims.
            claims=read_jsonl(ROOT/cfg['datasets']['SciFact']['path']);blocked_ids={e['query_id'] for e in dev if e['dataset']=='SciFact'}|{e['query_id'] for e in manifests['confirmation']['SciFact']};blocked_docs={str(d) for r in claims if str(r['id']) in blocked_ids for d in r.get('cited_doc_ids',[])}
            changed=True
            while changed:
                old_count=len(blocked_docs)
                for r in native:
                    docs=set(map(str,r.get('cited_doc_ids',[])))
                    if docs & blocked_docs:blocked_docs.update(docs)
                changed=len(blocked_docs)!=old_count
            bad={str(r['id']) for r in native if any(str(d) in blocked_docs for d in r.get('cited_doc_ids',[]))};xs=[e for e in xs if e['query_id'] not in bad]
            # Connected document components ensure paper-group-disjoint train/val.
            from .data import components
            groups=components([dict(query_id=str(r['id']),split_documents=list(map(str,r.get('cited_doc_ids',[])))) for r in native]);
            for e in xs:e['cluster_id']=groups[e['query_id']]
        xs=training_groups(xs,dataset)
        xs=[e for e in xs if qhash(e['question']) not in blocked_q];xs.sort(key=lambda e:digest([dataset,e['cluster_id'],e['query_id'],cfg['seed']]))
        groups=sorted({e['cluster_id'] for e in xs},key=lambda g:digest([g,42]));val_groups=set(groups[:max(1,len(groups)//5)])
        train=[e for e in xs if e['cluster_id'] not in val_groups][:cfg['training_queries_per_dataset']];valid=[e for e in xs if e['cluster_id'] in val_groups][:max(10,cfg['training_queries_per_dataset']//10)]
        if not train or not valid:raise ValueError('No disjoint usable train/validation group: '+dataset)
        records.extend(dict(example=e,partition=part) for part,group in [('train',train),('validation',valid)] for e in group)
    return records

def prepare_corpus(cfg):
    if cfg.get('split_protocol'):
        from .training_v2 import prepare_corpus as prepare
        return prepare(cfg)
    from transformers import AutoTokenizer
    from .laboratory import Lab
    from .algorithm import choose,validate_spans
    from .reader import prompt_ids
    tok=AutoTokenizer.from_pretrained(cfg['models']['Qwen']['id'],revision=cfg['models']['Qwen']['revision'],local_files_only=True);lab=Lab(cfg,tok);out=root(cfg)/'lora'/'corpus';out.mkdir(parents=True,exist_ok=True);rows=[]
    sources=corpus_sources(cfg)
    from .protocol import Store
    source_specs=json.loads((root(cfg)/'protocol/cohorts.json').read_text())['training'];shards=Store(out/'shards',identity(cfg,dict(training_sources={d:sha(r['path']) for d,r in source_specs.items()})))
    manifest=[dict(dataset=r['example']['dataset'],query_id=r['example']['query_id'],cluster_id=r['example']['cluster_id'],partition=r['partition']) for r in sources];immutable(out/'membership.json',manifest)
    try:
        for item in sources:
            e=item['example']
            if all(shards.get([e['dataset'],e['query_id'],mode,b]) is not None for mode in cfg['lora']['modes'] for b in cfg['budgets']):continue
            pool,sent,props,ps,ss,meta=lab.prepare(e)
            for mode in cfg['lora']['modes']:
                for budget in cfg['budgets']:
                    key=[e['dataset'],e['query_id'],mode,budget]
                    if shards.get(key) is not None:continue
                    selection=choose(pool,sent,props,ps,ss,tok,budget,mode);validate_spans(selection['units'],e['inference']['documents']);prompt=prompt_ids(tok,e['question'],e['task_type'],selection['evidence_ids'],cfg['labels']);target=e['evaluation']['gold_label'] if e['task_type']=='verification' else str(e['evaluation']['gold_answer'][0]);target_ids=tok.encode(target,add_special_tokens=False)+[tok.eos_token_id]
                    if len(prompt)+len(target_ids)>cfg['lora']['max_length']:raise ValueError('Training sequence exceeds preregistered cap; no silent truncation')
                    shards.put(key,dict(dataset=e['dataset'],query_id=e['query_id'],cluster_id=e['cluster_id'],partition=item['partition'],mode=mode,budget=budget,input_ids=prompt+target_ids,labels=[-100]*len(prompt)+target_ids,attention_mask=[1]*(len(prompt)+len(target_ids)),target=target,evidence_hash=selection['evidence_hash']))
    finally:lab.close()
    rows=shards.rows();immutable(out/'rows.json',rows);return rows

def train(cfg,explicit=False):
    if not explicit:raise ValueError('Full LoRA training needs manual explicit opt-in')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Trainer,TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint
    from peft import prepare_model_for_kbit_training,get_peft_model,LoraConfig
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(cfg['seed'])
    rows=json.loads((root(cfg)/'lora/corpus/rows.json').read_text());train_rows=[r for r in rows if r['partition']=='train'];val_rows=[r for r in rows if r['partition']=='validation']
    assert not {(r['dataset'],r['cluster_id']) for r in train_rows}&{(r['dataset'],r['cluster_id']) for r in val_rows}
    if cfg.get('split_protocol'):
        from .training_v2 import validate_rows
        validate_rows(cfg,rows)
    out=root(cfg)/'lora';ident=identity(cfg,dict(corpus_hash=digest(rows),reader_precision='NF4 both base and adapted in 2x2'));immutable(out/'train_identity.json',ident)
    if (out/'complete.json').exists():return json.loads((out/'complete.json').read_text())
    spec=cfg['models']['Qwen'];quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True,quantization_config=quant,device_map={'':0},attn_implementation='sdpa');model=prepare_model_for_kbit_training(model);model=get_peft_model(model,LoraConfig(r=cfg['lora']['r'],lora_alpha=cfg['lora']['alpha'],lora_dropout=cfg['lora']['dropout'],target_modules=cfg['lora']['target_modules'],task_type='CAUSAL_LM'));model.config.use_cache=False
    tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True);tok.pad_token=tok.eos_token
    from .scheduling import full_coverage,CoverageLedger
    indices,schedule=full_coverage(train_rows,cfg['lora']['gradient_accumulation_steps'],cfg['seed'])
    loss_weights=schedule['effective_loss_weights']
    ledger=CoverageLedger(out/'coverage_events.jsonl',[(r['dataset'],r['query_id']) for r in train_rows],loss_weights)
    def collate(batch):
        width=max(len(x['input_ids']) for x in batch)
        data={k:torch.tensor([x[k]+[(-100 if k=='labels' else 0 if k=='attention_mask' else tok.pad_token_id)]*(width-len(x[k])) for x in batch]) for k in ['input_ids','labels','attention_mask']}
        data.update(_datasets=[x['dataset'] for x in batch],_queries=[x['query_id'] for x in batch],_weights=torch.tensor([loss_weights[x['dataset']] for x in batch],dtype=torch.float32))
        return data
    class BalancedTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            ds=inputs.pop('_datasets');qs=inputs.pop('_queries');weights=inputs.pop('_weights')
            outputs=model(**inputs)
            if len(ds)!=1:raise ValueError('Frozen scheduler uses the registered batch_size=1, accumulation=16')
            loss=outputs.loss*weights[0]
            if model.training and torch.is_grad_enabled():ledger.forward(ds,qs,weights.detach().cpu().tolist())
            return (loss,outputs) if return_outputs else loss
        def get_train_dataloader(self):
            from collections import Counter
            from .training_v2 import balancing
            indices,plan=balancing(train_rows,cfg['lora']['gradient_accumulation_steps'],cfg['seed'])
            atomic(out/'sampling_plan.json',plan)
            sampler=indices
            return self.accelerator.prepare(torch.utils.data.DataLoader(train_rows,batch_size=self.args.per_device_train_batch_size,sampler=sampler,collate_fn=collate))
        def evaluate(self,eval_dataset=None,ignore_keys=None,metric_key_prefix='eval'):
            if not cfg.get('split_protocol'):return super().evaluate(eval_dataset,ignore_keys,metric_key_prefix)
            from .training_v2 import validation_metrics
            result=validation_metrics(self.model,tok,val_rows,cfg)
            atomic(out/'validation_metrics'/('step_'+str(self.state.global_step)+'.json'),dict(step=self.state.global_step,**result))
            metrics={'eval_loss':result['eval_loss']};self.log(metrics);self.control=self.callback_handler.on_evaluate(self.args,self.state,self.control,metrics);return metrics
    args=TrainingArguments(output_dir=str(out/'checkpoints'),num_train_epochs=cfg['lora']['epochs'],per_device_train_batch_size=cfg['lora']['batch_size'],per_device_eval_batch_size=1,gradient_accumulation_steps=cfg['lora']['gradient_accumulation_steps'],learning_rate=cfg['lora']['learning_rate'],eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,save_total_limit=2,logging_steps=10,bf16=True,gradient_checkpointing=True,report_to=[],remove_unused_columns=False,seed=cfg['seed'],data_seed=cfg['seed'])
    trainer=BalancedTrainer(model=model,args=args,train_dataset=train_rows,eval_dataset=val_rows,data_collator=collate)
    trainer.model_accepts_loss_kwargs=False
    from transformers import TrainerCallback
    class CoverageCallback(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            ledger.step(state.global_step,not trainer.accelerator.optimizer_step_was_skipped)
            return control
    trainer.add_callback(CoverageCallback())
    last=get_last_checkpoint(str(out/'checkpoints')) if (out/'checkpoints').exists() else None;trainer.train(resume_from_checkpoint=last);trainer.save_model(str(out/'adapter'));tok.save_pretrained(out/'adapter');atomic(out/'training_history.json',dict(log_history=trainer.state.log_history,chosen_checkpoint=trainer.state.best_model_checkpoint,objective=cfg['lora'].get('checkpoint_objective'),stopping_criterion='fixed one epoch; no confirmation access',completed_optimizer_steps=trainer.state.global_step))
    actual=ledger.summary()
    if any(actual['missing_train_examples'].values()):raise ValueError('Full training ended without consuming every eligible TRAIN example')
    atomic(out/'sampling_actual.json',dict(actual,complete=True,completed_optimizer_steps=trainer.state.global_step,epochs_completed=trainer.state.epoch))
    result=dict(complete=True,identity_hash=digest(ident),adapter_hash=digest({p.name:sha(p) for p in (out/'adapter').glob('*') if p.is_file()}));immutable(out/'complete.json',result);return result
