"""One reader at a time. Synthetic tests never instantiate these classes."""
from .storage import ROOT

def require_memory(weight_gb):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    free,total=torch.cuda.mem_get_info()
    required=weight_gb*1e9+4*1024**3
    if free<required:raise MemoryError(f'Insufficient free GPU memory before model load: {free/1e9:.1f}GB free, estimated need {required/1e9:.1f}GB')


class SaraReader:
    def __init__(self,cfg):
        from src.week9.backend import SaraBackend
        require_memory(14.5)
        self.backend=SaraBackend(dict(checkpoint=cfg['checkpoint'],device='cuda',max_new_tokens=128))
        self.tokenizer=self.backend.tokenizer
    def generate(self,question,ranked,k,vectors=None):
        from src.week6.routing import build_routed_sara_prompt
        import torch
        passages=[dict(text=c['text']) for c in ranked]
        if vectors is None:
            # Identical reader weights. No latent vectors in full-text control.
            passages=passages[:k]; selected=list(range(len(passages)))
        else: selected=list(range(min(k,len(passages))))
        prompt,_,heavy=build_routed_sara_prompt(question,passages,selected,self.tokenizer)
        emb=None if vectors is None else torch.tensor(vectors[heavy],dtype=torch.float16,device='cuda') if heavy else None
        result=self.backend.generate(prompt,128,emb)
        from .storage import digest
        return dict(result,prompt_hash=digest(prompt),lm_calls=1,cumulative_prompt_tokens=result['prompt_tokens'])
    def generate_text(self,question,text):
        from .storage import digest
        user='Answer my questions based on the given context.\n---\n## Context\n'+text+'\n---\n## Your Task\nAnswer the following question in a succinct manner. Use a single phrase or a short sentence if possible.\nQuestion: '+question+'\nYour Answer:'
        prompt=self.tokenizer.apply_chat_template([dict(role='user',content=user)],tokenize=False,add_generation_prompt=True)
        result=self.backend.generate(prompt,128,None)
        return dict(result,prompt_hash=digest(prompt),lm_calls=1,cumulative_prompt_tokens=result['prompt_tokens'])

    def close(self):
        import gc,torch
        self.backend.model=None; gc.collect();torch.cuda.empty_cache()

class TextReader:
    def __init__(self,spec):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        self.spec=spec
        require_memory(16.4 if 'Qwen3' in spec['id'] else 16.1 if 'Llama' in spec['id'] else 14.5)
        self.tokenizer=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True,
            torch_dtype=torch.float16,device_map={'':0},attn_implementation='sdpa').eval()
    def generate(self,question,text,seed=None):
        import torch,time
        from .storage import digest
        user='Answer the question in a single phrase or short sentence using the context if provided.\nContext:\n'+text+'\nQuestion: '+question
        kwargs={'enable_thinking':False} if 'Qwen3-' in self.spec['id'] else {}
        prompt=self.tokenizer.apply_chat_template([dict(role='user',content=user)],tokenize=False,add_generation_prompt=True,**kwargs)
        inputs=self.tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to('cuda');n=inputs.input_ids.shape[-1]
        if n+128>8192: raise ValueError('Preregistered 8192-token limit exceeded; no silent truncation')
        if seed is not None: torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        options=dict(do_sample=False) if seed is None else dict(do_sample=True,temperature=.7,top_p=.9)
        started=time.perf_counter()
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with torch.inference_mode(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            out=self.model.generate(**inputs,max_new_tokens=128,**options,pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize();ids=out[0,n:];text=self.tokenizer.decode(ids,skip_special_tokens=True).strip()
        return dict(content=text,prompt_hash=digest(prompt),prompt_tokens=n,cumulative_prompt_tokens=n,generated_tokens=len(ids),
            latency=time.perf_counter()-started,lm_calls=1,model=self.spec,precision='fp16',seed=seed)
    def close(self):
        import gc,torch
        self.model=None;gc.collect();torch.cuda.empty_cache()
