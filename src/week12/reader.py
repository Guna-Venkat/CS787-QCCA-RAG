"""Qwen token-ID block reader. Same evidence IDs are used in paired LoRA arms."""
import time
from .protocol import digest
LABELS=('SUPPORT','REFUTE','NOT_ENOUGH_INFO')
def parse_label(text,labels=LABELS):return text.strip() if text.strip() in labels else 'INVALID_OUTPUT'
def framing(question,task,labels=LABELS):
    instruction=('Classify the claim using the context. Return exactly one allowed label: '+', '.join(labels)+'. Return no explanation.') if task=='verification' else 'Answer the question in a single phrase or short sentence using the context if provided.'
    return instruction+'\nContext:\n','\n'+('Claim' if task=='verification' else 'Question')+': '+question

def prompt_ids(tok,question,task,evidence_ids,labels=LABELS):
    pre,post=framing(question,task,labels);marker='W12_EVIDENCE_INSERTION_6a4b9c'
    chat=tok.apply_chat_template([dict(role='user',content=pre+marker+post)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    if chat.count(marker)!=1:raise ValueError('Prompt marker ambiguous')
    a,b=chat.split(marker);return tok.encode(a,add_special_tokens=False)+list(evidence_ids)+tok.encode(b,add_special_tokens=False)

class Reader:
    def __init__(self,cfg,variant='BASE_FP16',adapter=None):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
        from .protocol import sha
        self.adapter_hash=digest({p.name:sha(p) for p in adapter.glob('*') if p.is_file()}) if adapter is not None else None
        self.cfg=cfg;self.variant=variant;spec=cfg['models']['Qwen'];self.tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True)
        opts=dict(local_files_only=True,revision=spec['revision'],device_map={'':0},attn_implementation='sdpa')
        if variant in ('BASE_NF4','LORA_NF4'):opts['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
        else:opts['torch_dtype']=torch.float16
        self.model=AutoModelForCausalLM.from_pretrained(spec['id'],**opts)
        if variant=='LORA_NF4':
            if adapter is None:raise ValueError('Qualified adapter required')
            from peft import PeftModel
            self.model=PeftModel.from_pretrained(self.model,str(adapter))
        self.model.eval()
    def generate(self,example,selection):
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        ids=prompt_ids(self.tok,example['question'],example['task_type'],selection['evidence_ids'],self.cfg['labels']);cap=self.cfg['generation']['max_new_tokens_verification' if example['task_type']=='verification' else 'max_new_tokens_qa']
        if len(ids)+cap>self.cfg['generation']['max_context']:raise ValueError('Context overflow: no silent truncation')
        torch.manual_seed(self.cfg['seed']);torch.cuda.manual_seed_all(self.cfg['seed']);data=torch.tensor([ids],device=self.model.device);mask=torch.ones_like(data);t=time.perf_counter()
        with torch.inference_mode(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):out=self.model.generate(input_ids=data,attention_mask=mask,max_new_tokens=cap,do_sample=False,pad_token_id=self.tok.eos_token_id)
        torch.cuda.synchronize();new=out[0,len(ids):];raw=self.tok.decode(new,skip_special_tokens=True).strip()
        return dict(raw_prediction=raw,prediction=parse_label(raw,self.cfg['labels']) if example['task_type']=='verification' else raw,prompt_tokens=len(ids),generated_tokens=len(new),generation_seconds=time.perf_counter()-t,prompt_hash=digest(ids),generation_cap_reached=len(new)>=cap,reader=self.variant,model_hash=digest([self.cfg['models']['Qwen'],self.variant,self.adapter_hash]))
    def close(self):
        import torch,gc
        self.model=None;gc.collect();torch.cuda.empty_cache()
