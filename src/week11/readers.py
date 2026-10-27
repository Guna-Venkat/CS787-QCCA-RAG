import gc
import time
from .prompts import user_prompt
from .cache import digest
from .budgets import FixtureTokenizer


def tokenizers(cfg,synthetic=False):
    if synthetic:return {k:FixtureTokenizer() for k in cfg['readers']}
    from transformers import AutoTokenizer
    return {k:AutoTokenizer.from_pretrained(cfg['models'][k]['id'],revision=cfg['models'][k]['revision'],local_files_only=True)
            for k in cfg['readers']}


class Reader:
    def __init__(self,cfg,name):
        if name not in cfg['readers']:raise ValueError('Unsupported reader')
        from src.week10.root_cause.runtime import TextReader
        self.runtime=TextReader(cfg['models'][name]);self.name=name;self.cfg=cfg
    def generate(self,question,evidence,task_type):
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        tok=self.runtime.tokenizer
        user=user_prompt(question,evidence,task_type)
        kwargs={'enable_thinking':False} if self.name=='Qwen' else {}
        prompt=tok.apply_chat_template([dict(role='user',content=user)],tokenize=False,add_generation_prompt=True,**kwargs)
        data=tok(prompt,return_tensors='pt',add_special_tokens=False).to('cuda');n=data.input_ids.shape[-1]
        if n+self.cfg['generation']['max_new_tokens']>self.cfg['generation']['max_context']:
            raise ValueError('Reader context overflow; FULL is not unlimited, no truncation permitted')
        torch.manual_seed(self.cfg['seed']);torch.cuda.manual_seed_all(self.cfg['seed']);started=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            output=self.runtime.model.generate(**data,max_new_tokens=self.cfg['generation']['max_new_tokens'],
                do_sample=False,pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize();ids=output[0,n:]
        return dict(prediction=tok.decode(ids,skip_special_tokens=True).strip(),prompt_tokens=n,generated_tokens=len(ids),
            generation_latency=time.perf_counter()-started,prompt_hash=digest(prompt),peak_vram=torch.cuda.max_memory_allocated())
    def close(self):self.runtime.close()


class FixtureReader:
    def __init__(self,cfg,name):self.name=name;self.cfg=cfg
    def generate(self,question,evidence,task_type):
        # Fabricated response independent of gold; never write outside smoke.
        prediction='SUPPORT' if task_type=='verification' and 'supports' in evidence else 'NOT_ENOUGH_INFO' if task_type=='verification' else (evidence.split('.')[0] if evidence else 'unknown')
        return dict(prediction=prediction,prompt_tokens=len(FixtureTokenizer().encode(user_prompt(question,evidence,task_type))),
            generated_tokens=len(prediction.split()),generation_latency=0.,peak_vram=None,synthetic=True)
    def close(self):pass
