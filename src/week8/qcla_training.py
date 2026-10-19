"""Canonical semantic-k4 QCLA training with frozen SARA and real conditioning.

One example is one query. Only adapter weights receive gradients. Gold answers
are supervised targets, never adapter inputs. QCLA-2 conditions on the saved
unadapted k4 draft. No validation-dependent checkpoint or epoch selection.
"""
from __future__ import annotations
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.week8.generator_features import canonical_k4_context
from src.week6.evaluation import compute_qa_answer_metrics


def finite_vector(value, shape, name):
    if value is None or tuple(value.shape) != tuple(shape) or not torch.isfinite(value).all():
        raise ValueError(f'Missing or invalid measured {name}: expected {shape}')
    return value.detach().float().cpu()


class QCLATrainingDataset(Dataset):
    def __init__(self, records, sfr_embeddings_by_qid, query_embeddings_by_qid=None,
                 draft_embeddings_by_qid=None, max_samples=None, *, routing_by_qid=None):
        self.records = records[:max_samples] if max_samples is not None else records
        self.sfr_embeddings = sfr_embeddings_by_qid
        self.query_embeddings = query_embeddings_by_qid or {}
        self.draft_embeddings = draft_embeddings_by_qid or {}
        self.routing = routing_by_qid or {}
        for r in self.records:
            q = str(r['question_id'])
            if q not in self.routing: raise ValueError(f'{q}: missing canonical semantic routing')
            finite_vector(self.sfr_embeddings.get(q), (10,4096), 'passage embeddings')
            finite_vector(self.query_embeddings.get(q), (4096,), 'query embedding')
            finite_vector(self.draft_embeddings.get(q), (4096,), 'frozen draft embedding')
            g = self.routing[q]
            if str(g.iloc[0].paper_id) != str(r['paper_id']): raise ValueError('Paper identity differs')
            if str(g.iloc[0].question).strip() != str(r['question']).strip(): raise ValueError('Question differs')

    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        r = self.records[idx]; q = str(r['question_id'])
        return dict(question_id=q, record=r, passage_embeddings=self.sfr_embeddings[q].float(),
                    query_embedding=self.query_embeddings[q].float(), draft_embedding=self.draft_embeddings[q].float(),
                    routing=self.routing[q])


def single_query_collate(batch):
    if len(batch) != 1: raise ValueError('QCLA requires one complete query per batch')
    return batch[0]


def prepare_example(batch, tokenizer, adapter, device):
    r = batch['record']
    _, prompt, _, compressed, embeds, full = canonical_k4_context(
        batch['routing'], r, batch['passage_embeddings'], tokenizer)
    # Cached SFR values are bfloat16; train adapter operations in float32.
    transformed = adapter(embeds.to(device=device,dtype=torch.float32),
        e_q=batch['query_embedding'].to(device=device,dtype=torch.float32),
        e_y=batch['draft_embedding'].to(device=device,dtype=torch.float32))
    return prompt, transformed, full, compressed


def gold_target_inputs(tokenizer, prompt, record, device, max_target_tokens=128):
    gold = record.get('gold_answers')
    if not isinstance(gold,list) or not gold or not isinstance(gold[0],str) or not gold[0].strip():
        raise ValueError('Missing real gold-answer target; no empty or fabricated target')
    prefix = tokenizer(prompt,return_tensors='pt')['input_ids']
    # Preserve the canonical generation prefix exactly; never append another Answer marker/BOS.
    target = tokenizer(' '+gold[0],add_special_tokens=False)['input_ids'][:max_target_tokens-1]
    target = torch.tensor([target+[tokenizer.eos_token_id]],dtype=torch.long)
    ids = torch.cat([prefix,target],1).to(device)
    return ids, target.to(device)


def answer_loss(model, ids, targets, embeds):
    # Compute vocabulary logits only at answer-predicting positions, rather than
    # retaining prompt-length vocabulary tensors. The causal shift is explicit.
    n = targets.shape[1]
    result = model(input_ids=ids,attention_mask=torch.ones_like(ids),
        retrieval_embeds=embeds.to(dtype=model.dtype),use_cache=False,logits_to_keep=n+1)
    logits = result.logits[:,-n-1:-1,:].float()
    loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1))
    if not torch.isfinite(loss): raise ValueError('Nonfinite measured answer loss')
    return loss


def freeze_generator(model, training=False):
    for p in model.parameters(): p.requires_grad_(False)
    if training:
        # HF checkpointing is active in train mode; disable frozen-generator
        # dropout so adaptation does not change the generation environment.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout): module.eval()
    else: model.eval()


def train_qcla_epoch(adapter,sara_model,sara_tokenizer,dataloader,optimizer,
                     scaler=None,gradient_accumulation_steps=16,device='cuda',
                     candidate_indices=None,use_amp=False,max_steps=None):
    if candidate_indices is not None: raise ValueError('Hardcoded candidate allocation is unsupported')
    if scaler is not None or use_amp: raise ValueError('Use float32 adapter operations and SARA bfloat16; no external autocast/scaler')
    if gradient_accumulation_steps < 1: raise ValueError('Positive accumulation count required')
    freeze_generator(sara_model,True); adapter.train(); optimizer.zero_grad(set_to_none=True)
    total=0.; count=0; pending=0; gradient_verified=False
    for batch in dataloader:
        prompt, embeds, _, _ = prepare_example(batch,sara_tokenizer,adapter,device)
        ids,targets = gold_target_inputs(sara_tokenizer,prompt,batch['record'],device)
        loss = answer_loss(sara_model,ids,targets,embeds)
        # Sum gradients first; normalize by actual group size, including the last partial group.
        loss.backward(); total+=float(loss.detach()); count+=1; pending+=1
        del loss, embeds, ids, targets
        at_end=(count==len(dataloader)) or (max_steps is not None and count>=max_steps)
        if pending==gradient_accumulation_steps or at_end:
            gradients=[p.grad for p in adapter.parameters() if p.requires_grad and p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise ValueError('Adapter gradients missing or nonfinite')
            if not gradient_verified:
                if not any(torch.count_nonzero(g).item() for g in gradients): raise ValueError('No nonzero adapter gradient through frozen SARA')
                gradient_verified=True
            for g in gradients:g.div_(pending)
            torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.)
            optimizer.step();optimizer.zero_grad(set_to_none=True);pending=0
        if max_steps is not None and count>=max_steps: break
    if not count:raise ValueError('No measured training queries')
    if any(p.grad is not None for p in sara_model.parameters()):raise ValueError('Frozen SARA received parameter gradients')
    return total/count


@torch.no_grad()
def evaluate_qcla(adapter,sara_model,sara_tokenizer,dataloader,device='cuda',
                  candidate_indices=None,use_amp=False,max_steps=None,*,generate=False):
    if candidate_indices is not None:raise ValueError('Use canonical saved routing')
    freeze_generator(sara_model);adapter.eval();rows=[]
    for batch in dataloader:
        r=batch['record'];prompt,embeds,full,compressed=prepare_example(batch,sara_tokenizer,adapter,device)
        ids,targets=gold_target_inputs(sara_tokenizer,prompt,r,device)
        nll=float(answer_loss(sara_model,ids,targets,embeds))
        row=dict(question_id=str(r['question_id']),paper_id=str(r['paper_id']),answer_nll=nll,
                 full_indices=full,compressed_indices=compressed,target_tokens=targets.numel())
        if generate:
            prefix=sara_tokenizer(prompt,return_tensors='pt')['input_ids'].to(device)
            start=time.perf_counter()
            output=sara_model.generate(input_ids=prefix,attention_mask=torch.ones_like(prefix),
                retrieval_embeds=embeds.to(dtype=sara_model.dtype),max_new_tokens=128,do_sample=False,
                pad_token_id=sara_tokenizer.pad_token_id,eos_token_id=sara_tokenizer.eos_token_id)
            if str(device).startswith('cuda'):torch.cuda.synchronize()
            duration=time.perf_counter()-start
            prediction=sara_tokenizer.decode(output[0],skip_special_tokens=True).strip()
            metrics=compute_qa_answer_metrics(prediction,r['gold_answers'],r.get('gold_answers_reformatted',r['gold_answers']))
            row.update(prediction=prediction,token_f1=metrics['token_f1'],
                generation_latency_sec=duration,prompt_tokens=prefix.shape[1],
                generated_tokens=len(output[0]),baseline_draft=str(batch['routing'].iloc[0].draft_answer))
        rows.append(row)
        if max_steps is not None and len(rows)>=max_steps:break
    if not rows:raise ValueError('No measured evaluation queries')
    return rows if generate else sum(r['answer_nll'] for r in rows)/len(rows)
