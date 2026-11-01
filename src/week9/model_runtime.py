"""One textual model at a time, FP16 on Turing, all generated tokens counted."""
from __future__ import annotations
import gc,re,time
from contextlib import contextmanager
from .model_registry import get_model,assert_projector_compatible,runtime_identity
from .provenance import digest


def parse_final(raw,reasoning=False):
    had_reasoning='<think>' in raw or '</think>' in raw
    if '</think>' in raw:final=raw.rsplit('</think>',1)[1].strip()
    elif '<think>' in raw:final=''  # truncated reasoning is not a final answer
    elif reasoning:final=''  # no identifiable boundary: do not score the reasoning as an answer
    else:final=raw.strip()
    return final,had_reasoning


def memory_status(allocated,limit_bytes=20_000_000_000):
    return 'PASS' if allocated<limit_bytes else 'FAIL'


class TextRuntime:
    _owner=None
    def __init__(self,model_id,revision,precision='fp16',*,cache_dir=None,thinking=False,device='cuda',max_prompt_tokens=4096):
        self.identity=runtime_identity(model_id,revision,precision,thinking)
        self.spec=get_model(model_id);self.device=device;self.model=None;self.tokenizer=None
        self.cache_dir=cache_dir;self.max_prompt_tokens=max_prompt_tokens
        assert_projector_compatible(model_id,'text')
    def load(self):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
        from packaging.version import Version
        import transformers
        if Version(transformers.__version__)<Version(self.spec.transformers_min):raise RuntimeError('Transformers too old for '+self.spec.model_id)
        if TextRuntime._owner not in (None,self):raise RuntimeError('Release the previous model before loading another')
        if not torch.cuda.is_available() or self.device!='cuda':raise RuntimeError('Representative VRAM smoke requires one CUDA GPU')
        TextRuntime._owner=self
        try:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            options=dict(revision=self.identity['model_revision'],cache_dir=self.cache_dir,local_files_only=True,trust_remote_code=False)
            self.tokenizer=AutoTokenizer.from_pretrained(self.spec.model_id,**options)
            if not self.tokenizer.chat_template:raise ValueError('Official tokenizer has no chat template')
            quant=None;p=self.identity['precision']
            if p=='int8':quant=BitsAndBytesConfig(load_in_8bit=True)
            if p in ('nf4','nf4_fp32'):quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float32 if p=='nf4_fp32' else torch.float16,bnb_4bit_use_double_quant=True)
            cls=AutoModelForCausalLM
            if self.spec.key=='M5':
                from transformers import Gemma3ForConditionalGeneration
                cls=Gemma3ForConditionalGeneration
            started=time.perf_counter()
            self.model=cls.from_pretrained(self.spec.model_id,torch_dtype=torch.float32 if p=='nf4_fp32' else torch.float16,device_map={'':0},attn_implementation='eager',quantization_config=quant,**options).eval()
            self.load_seconds=time.perf_counter()-started
            self.load_peak_allocated=torch.cuda.max_memory_allocated();self.load_peak_reserved=torch.cuda.max_memory_reserved()
            if memory_status(self.load_peak_allocated)=='FAIL':raise MemoryError('Model loading exceeded the strict <20 GB allocated ceiling')
            return self
        except (RuntimeError,ValueError,MemoryError,ImportError,OSError):
            self.close()
            raise
    def render(self,messages):
        if self.tokenizer is None:raise RuntimeError('Tokenizer unavailable')
        kwargs={'enable_thinking':self.identity['thinking']} if self.spec.key=='M4' else {}
        # Same semantic user prompt; official template controls special tokens only.
        return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**kwargs)
    def generate(self,prompt,max_tokens=128,retrieval_embeds=None):
        import torch
        if retrieval_embeds is not None:raise ValueError('Text models cannot receive SARA projector/latent embeddings')
        if self.model is None:raise RuntimeError('Load one textual runtime first')
        inputs=self.tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to(self.device)
        n=inputs.input_ids.shape[1]
        if n>self.max_prompt_tokens or n+max_tokens>self.spec.context_length:raise ValueError('Prompt/context budget exceeded; no silent truncation')
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        with torch.inference_mode():
            out=self.model.generate(**inputs,do_sample=False,max_new_tokens=max_tokens,use_cache=True,return_dict_in_generate=True,output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        torch.cuda.synchronize();latency=time.perf_counter()-started
        validate_scores(out.scores)
        ids=out.sequences[0,n:].tolist();eos=self.model.generation_config.eos_token_id
        ends=[eos] if isinstance(eos,int) else (eos or [])
        if ends:
            end=next((i for i,x in enumerate(ids) if x in ends),None)
            if end is not None:ids=ids[:end+1]
        raw=self.tokenizer.decode(ids,skip_special_tokens=True).strip()
        # Qwen3 officially supports no-thinking; DeepSeek has no equivalent switch.
        reasoning=self.spec.reasoning and not (self.spec.key=='M4' and not self.identity['thinking'])
        final,thinking=parse_final(raw,reasoning)
        allocated=max(self.load_peak_allocated,torch.cuda.max_memory_allocated());reserved=max(self.load_peak_reserved,torch.cuda.max_memory_reserved())
        result=dict(generation_scores_valid=True,content=final,raw_output=raw,raw_output_publication_safe=False,unexpected_reasoning=thinking,
            empty_output=not bool(final),truncated=bool(ids) and ids[-1] not in ends,
            prompt_tokens=n,generated_tokens=len(ids),total_tokens=n+len(ids),input_characters=len(prompt),input_words=len(prompt.split()),
            generated_token_ids=ids,latency=latency,terminated=bool(ids) and ids[-1] in ends,
            peak_memory_allocated=allocated,peak_memory_reserved=reserved,vram_status=memory_status(allocated),
            gpu=torch.cuda.get_device_name(0),compute_capability=list(torch.cuda.get_device_capability(0)),load_seconds=self.load_seconds,**self.identity)
        if result['vram_status']=='FAIL':raise MemoryError('Representative run exceeded <20 GB peak allocated VRAM')
        return result
    def close(self):
        import torch
        self.model=None;self.tokenizer=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        if TextRuntime._owner is self:TextRuntime._owner=None
    def __enter__(self):return self.load()
    def __exit__(self,*args):self.close()


def validate_scores(scores):
    import torch
    for score in scores:
        if torch.isnan(score).any() or torch.isposinf(score).any() or not torch.isfinite(score).any():
            raise RuntimeError('Numerically invalid generation scores; do not interpret this precision as a quality result')
