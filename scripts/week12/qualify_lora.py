"""One synthetic optimizer step on pinned Qwen NF4; no real training data or adapter saved."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.environ.setdefault('HF_HUB_OFFLINE','1')
from src.week12.protocol import load,root,atomic,identity

def main():
 import torch,gc
 from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
 from peft import prepare_model_for_kbit_training,get_peft_model,LoraConfig
 cfg=load();spec=cfg['models']['Qwen'];torch.manual_seed(42)
 tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True)
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
 model=AutoModelForCausalLM.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True,device_map={'':0},quantization_config=quant,attn_implementation='sdpa');model=prepare_model_for_kbit_training(model)
 model=get_peft_model(model,LoraConfig(r=cfg['lora']['r'],lora_alpha=cfg['lora']['alpha'],lora_dropout=cfg['lora']['dropout'],target_modules=cfg['lora']['target_modules'],task_type='CAUSAL_LM'));model.config.use_cache=False
 ids=tok.encode('Synthetic fixture. Context: a toy signal is red. Question: what color? Answer: red',add_special_tokens=False);data=torch.tensor([ids],device='cuda');labels=data.clone();labels[:,:-2]=-100
 before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad};base={n:p.detach().flatten()[:8].clone() for n,p in model.named_parameters() if not p.requires_grad}
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0002)
 from torch.nn.attention import sdpa_kernel,SDPBackend
 with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):loss=model(input_ids=data,attention_mask=torch.ones_like(data),labels=labels).loss;loss.backward();opt.step()
 changed=any(not torch.equal(before[n],p) for n,p in model.named_parameters() if p.requires_grad);base_ok=all(torch.equal(base[n],p.detach().flatten()[:8]) for n,p in model.named_parameters() if not p.requires_grad)
 assert changed and base_ok and torch.isfinite(loss)
 result=dict(status='PASS',synthetic_data_only=True,optimizer_steps=1,model=spec,precision='NF4/bfloat16',loss=float(loss.detach()),adapter_changed=changed,base_parameter_samples_unchanged=base_ok,adapter_saved=False,full_training_started=False,identity=identity(cfg),peak_vram_bytes=torch.cuda.max_memory_allocated())
 atomic(root(cfg)/'qualification/real_qlora_smoke.json',result);print({k:v for k,v in result.items() if k!='identity'})
 del model,opt;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
