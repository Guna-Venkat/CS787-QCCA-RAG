"""Additive rank-8 LoRA on frozen SARA, preserving its original PEFT adapter.

The original q/k/v/o projections (including trained SARA LoRA) remain intact.
Only new residual factors are trainable. They are saved independently and need
the original SARA checkpoint when reloaded. No merging or nested PEFT wrappers.
"""
from __future__ import annotations
import math
import torch
from tqdm.auto import tqdm
from src.week8.qcla_training import answer_loss


class DeltaLoRALinear(torch.nn.Module):
    def __init__(self,original,r=8,alpha=16,dropout=.05):
        super().__init__();self.original=original;self.scale=alpha/r
        self.in_features=original.in_features;self.out_features=original.out_features
        device=next(original.parameters()).device
        self.w8g_A=torch.nn.Parameter(torch.empty(r,self.in_features,device=device,dtype=torch.float32))
        self.w8g_B=torch.nn.Parameter(torch.zeros(self.out_features,r,device=device,dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.w8g_A,a=math.sqrt(5))
        self.w8g_dropout=torch.nn.Dropout(dropout)
        for p in original.parameters():p.requires_grad_(False)
    def forward(self,x,*args,**kwargs):
        baseline=self.original(x,*args,**kwargs)
        low=torch.nn.functional.linear(self.w8g_dropout(x.float()),self.w8g_A)
        delta=torch.nn.functional.linear(low,self.w8g_B)*self.scale
        return baseline+delta.to(baseline.dtype)


def setup_lora_generator(base_model,r=8,lora_alpha=16,lora_dropout=.05,target_modules=None):
    targets=set(target_modules or ['q_proj','k_proj','v_proj','o_proj'])
    if any(isinstance(m,DeltaLoRALinear) for m in base_model.modules()):raise ValueError('W8-G adapter already attached; reload original SARA for each variant')
    for p in base_model.parameters():p.requires_grad_(False);p.grad=None
    candidates=[(name,module) for name,module in base_model.named_modules()
        if name.rsplit('.',1)[-1] in targets and '.self_attn.' in name]
    if not candidates:raise ValueError('No real SARA attention projections found')
    for name,module in candidates:
        parent_name,leaf=name.rsplit('.',1);parent=base_model.get_submodule(parent_name)
        setattr(parent,leaf,DeltaLoRALinear(module,r,lora_alpha,lora_dropout))
    base_model._w8g_projection_names=[name for name,_ in candidates]
    trainable=[name for name,p in base_model.named_parameters() if p.requires_grad]
    if not trainable or any(not name.endswith(('w8g_A','w8g_B')) for name in trainable):
        raise ValueError('Only new W8-G residual factors may be trained')
    return base_model


def adapter_state(model):
    return {name:p.detach().cpu().clone() for name,p in model.named_parameters() if name.endswith(('w8g_A','w8g_B'))}


def load_adapter_state(model,state):
    params=dict(model.named_parameters());expected=set(adapter_state(model))
    if set(state)!=expected:raise ValueError('W8-G checkpoint architecture mismatch')
    with torch.no_grad():
        for name,value in state.items():
            if params[name].shape!=value.shape or not torch.isfinite(value).all():raise ValueError('Invalid adapter factor')
            params[name].copy_(value.to(params[name].device))


def training_mode(model):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if hasattr(model,'enable_input_require_grads'):model.enable_input_require_grads()
    model.train()
    # Only the new adapter's dropout is active during training.
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.eval()
    for module in model.modules():
        if isinstance(module,DeltaLoRALinear):module.w8g_dropout.train()


def compressed_embeddings(batch,sfr_embeddings_by_qid,model,device):
    q=str(batch['question_id']);source=sfr_embeddings_by_qid.get(q)
    indices=batch['compressed_indices']
    if source is None or source.shape!=(10,4096) or not torch.isfinite(source).all():raise ValueError(f'{q}: missing finite cached SFR embeddings')
    if indices!=sorted(set(indices)) or not set(indices)<=set(range(10)):raise ValueError('Invalid compressed-slot identities')
    return source[indices].to(device=device,dtype=model.dtype)  # flat slots x 4096


def train_generator_lora_epoch(sara_model,dataloader,optimizer,sfr_embeddings_by_qid,
        scaler=None,gradient_accumulation_steps=16,device='cuda',use_amp=False,max_steps=None):
    if scaler is not None or use_amp:raise ValueError('W8-G uses float32 residual factors with frozen bfloat16 SARA')
    if gradient_accumulation_steps<1:raise ValueError('Positive accumulation required')
    training_mode(sara_model);optimizer.zero_grad(set_to_none=True)
    total=0.;raw_total=0.;count=0;pending=0;verified=False
    trainable=[p for p in sara_model.parameters() if p.requires_grad]
    for batch in dataloader:
        embeds=compressed_embeddings(batch,sfr_embeddings_by_qid,sara_model,device)
        loss=answer_loss(sara_model,batch['input_ids'].to(device),batch['targets'].to(device),embeds)
        weight=float(batch['weight'])
        if not math.isfinite(weight) or weight<=0:raise ValueError('Invalid FIT-only sample weight')
        weighted=loss*weight;weighted.backward()
        total+=float(weighted.detach());raw_total+=float(loss.detach());count+=1;pending+=1
        del loss,weighted,embeds
        at_end=count==len(dataloader) or (max_steps is not None and count>=max_steps)
        if pending==gradient_accumulation_steps or at_end:
            grads=[p.grad for p in trainable if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('Missing or nonfinite W8-G gradients')
            if not verified:
                if not any(torch.count_nonzero(g).item() for g in grads):raise ValueError('No nonzero W8-G adapter gradients')
                verified=True
            for g in grads:g.div_(pending)
            torch.nn.utils.clip_grad_norm_(trainable,1.)
            optimizer.step();optimizer.zero_grad(set_to_none=True);pending=0
        if max_steps is not None and count>=max_steps:break
    if not count:raise ValueError('No real FIT examples')
    if any(p.grad is not None for p in sara_model.parameters() if not p.requires_grad):raise ValueError('Frozen original SARA received gradients')
    return {'query_mean_weighted_nll':total/count,'query_mean_unweighted_nll':raw_total/count,'queries':count,
        'only_new_adapter_gradients_verified':verified}


@torch.no_grad()
def evaluate_generator_lora(sara_model,dataloader,sfr_embeddings_by_qid,device='cuda',use_amp=False,max_steps=None):
    """Unweighted diagnostic NLL; generation-based evaluation lives in the runner."""
    sara_model.eval();values=[]
    for batch in dataloader:
        embeds=compressed_embeddings(batch,sfr_embeddings_by_qid,sara_model,device)
        values.append(float(answer_loss(sara_model,batch['input_ids'].to(device),batch['targets'].to(device),embeds)))
        if max_steps is not None and len(values)>=max_steps:break
    if not values:raise ValueError('No evaluation inputs')
    return sum(values)/len(values)
