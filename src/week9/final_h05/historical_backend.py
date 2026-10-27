"""Unchanged historical SARA first-pass/static precision and generate interface.

W7/W8 first passes loaded load_sara_for_eval with its BF16 default. E3/F0's
later FP16 efficient-SDPA revision backend remains separate and unchanged.
"""
import time
from ..final_h import protocol as p


class HistoricalBackend:
    def __init__(self,cfg):
        import src
        path=str(p.ROOT/'Baselines/SARA-main/src')
        if path not in src.__path__:src.__path__.append(path)
        from src.model.loader import load_sara_for_eval
        self.model,self.tokenizer=load_sara_for_eval(str(p.ROOT/cfg['checkpoint']),device=cfg['device'])
        self.device=cfg['device'];self.answer_max_tokens=128;self.cfg=cfg
    def identity(self):
        from ..backend import checkpoint_hashes
        if not hasattr(self,'_identity'):
            self._identity=dict(model_id='mistralai/Mistral-7B-Instruct-v0.2',model_revision=p.digest(checkpoint_hashes(self.cfg)),
                dtype=str(self.model.dtype),quantization='none',interface='trained_SARA',attention_backend='historical_default_sdpa')
        return self._identity
    def generate(self,prompt,max_tokens,retrieval_embeds):
        import torch
        inputs=self.tokenizer(prompt,return_tensors='pt').to(self.device)
        if str(self.device).startswith('cuda'):torch.cuda.synchronize()
        started=time.perf_counter()
        with torch.no_grad():
            output=self.model.generate(**inputs,retrieval_embeds=retrieval_embeds,max_new_tokens=max_tokens,
                do_sample=False,pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.tokenizer.eos_token_id)
        if str(self.device).startswith('cuda'):torch.cuda.synchronize()
        ids=output[0].tolist()
        return dict(content=self.tokenizer.decode(ids,skip_special_tokens=True).strip(),generated_token_ids=ids,
            generated_tokens=len(ids),prompt_tokens=inputs.input_ids.shape[-1],latency=time.perf_counter()-started,**self.identity())
