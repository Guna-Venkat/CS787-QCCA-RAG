"""Lazy local-only resources; CPU FP16 and GPU sequential execution."""
import threading,gc,time
from pathlib import Path
from demo.constants import ROOT

class ModelUnavailable(RuntimeError):pass

def configuration():
    import json
    return json.loads((ROOT/'results/week12/qbgf/v3/protocol/protocol_config.json').read_text())

def local_model(spec):
    path=Path.home()/'.cache/huggingface/hub'/('models--'+spec['id'].replace('/','--'))/'snapshots'/spec['revision']
    weights=list(path.glob('*.safetensors'))+list(path.glob('pytorch_model*.bin'))
    return path.is_dir() and bool(weights) and all(p.exists() for p in weights)

def cpu_available_mib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):return int(line.split()[1])/1024
    return 0

def inventory():
    import subprocess
    c=configuration();free=None
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
        if r.returncode==0:free=int(r.stdout.splitlines()[0])
    except (OSError,ValueError,IndexError,subprocess.TimeoutExpired):pass
    return dict(sfr_local=local_model(c['models']['SFR']),qwen_local=local_model(c['models']['Qwen']),gpu_free_mib=free,cpu_available_mib=cpu_available_mib())

class LiveResources:
    def __init__(self,device='cpu'):
        if device not in ('cpu','cuda'):raise ValueError('Expected cpu or cuda')
        self.lock=threading.RLock();self.cfg=configuration();self.tok=None;self.encoder=None;self.reader=None;self.vectors={};self.device=device
    def set_device(self,device):
        if device not in ('cpu','cuda'):raise ValueError('Expected cpu or cuda')
        with self.lock:
            if device!=self.device:self.release_gpu();self.device=device;self.vectors.clear()
    def _cpu_memory(self):
        available=cpu_available_mib()
        if available<24000:raise ModelUnavailable(f'CPU FP16 loading needs at least 24,000 MiB available RAM; {available:.0f} MiB available.')
        import torch,os
        torch.set_num_threads(min(8,os.cpu_count() or 1))
    def tokenizer(self):
        if self.tok is None:
            from transformers import AutoTokenizer
            spec=self.cfg['models']['Qwen']
            bundled=ROOT/'demo/assets/live_tokenizer';source=bundled/'source.json'
            if source.exists():
                import json,hashlib
                meta=json.loads(source.read_text())
                if meta['model']!=spec:raise ModelUnavailable('Bundled tokenizer pin differs from the live reader.')
                for name,expected in meta['files'].items():
                    if hashlib.sha256((bundled/name).read_bytes()).hexdigest()!=expected:raise ModelUnavailable('Bundled tokenizer integrity check failed.')
                self.tok=AutoTokenizer.from_pretrained(bundled,local_files_only=True)
            else:
                self.tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True)
        return self.tok
    def release_gpu(self):
        # Explicit CPU transfer frees parameter storage even if framework hooks retain
        # a model reference. Deleting the wrapper alone is insufficient on some versions.
        encoder,reader=self.encoder,self.reader
        self.encoder=None;self.reader=None
        if encoder is not None and self.device=='cuda':encoder.to('cpu')
        if reader is not None:
            model=reader.model;reader.model=None
            if model is not None and self.device=='cuda':model.to('cpu')
            del model
        del encoder,reader
        gc.collect()
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            clear_workspaces=getattr(torch._C,'_cuda_clearCublasWorkspaces',None)
            if clear_workspaces is not None:clear_workspaces()
            torch.cuda.empty_cache()
        # Return freed CPU tensor arenas to the OS where glibc provides this API.
        # File-backed checkpoint pages may remain reclaimable filesystem cache.
        try:
            import ctypes
            ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError,OSError):pass
    def _memory(self,mib):
        import torch
        if not torch.cuda.is_available():raise ModelUnavailable('CUDA reader/encoder is unavailable; BM25 context-only mode still works on CPU.')
        # Torch's allocator can reuse its own cached blocks although nvidia-smi
        # reports those bytes as occupied. Do not reject a safe same-process handoff.
        reusable=max(0,torch.cuda.memory_reserved()-torch.cuda.memory_allocated())
        available=(torch.cuda.mem_get_info()[0]+reusable)/1024**2
        if available<mib:raise ModelUnavailable(f'Insufficient available GPU memory ({available:.0f} MiB). Finish other GPU work or use BM25 context-only mode.')
    def available_memory_mib(self):
        import torch
        if not torch.cuda.is_initialized():return inventory()['gpu_free_mib']
        return (torch.cuda.mem_get_info()[0]+max(0,torch.cuda.memory_reserved()-torch.cuda.memory_allocated()))/1024**2
    def embed(self,texts,role):
        import numpy as np
        from .engine import digest
        keys=[digest([self.cfg['models']['SFR'],self.device,'float16',role,t]) for t in texts]
        missing=[(k,t) for k,t in zip(keys,texts) if k not in self.vectors]
        if missing:
            if self.encoder is None:
                self.release_gpu()
                if self.device=='cpu':self._cpu_memory()
                else:self._memory(16000)
                if not local_model(self.cfg['models']['SFR']):raise ModelUnavailable('Pinned SFR checkpoint is not available locally; no download was attempted.')
                import torch
                from sentence_transformers import SentenceTransformer
                spec=self.cfg['models']['SFR'];self.encoder=SentenceTransformer(spec['id'],revision=spec['revision'],device=self.device,local_files_only=True,model_kwargs={'torch_dtype':torch.float16,'attn_implementation':'sdpa'})
            from src.week10.root_cause.encoding import SFR_INSTRUCTION
            tok=self.encoder.tokenizer;width=self.encoder.max_seq_length-2
            for key,text in missing:
                value=(SFR_INSTRUCTION+text) if role=='query' else text;ids=tok.encode(value,add_special_tokens=False)
                windows=[tok.decode(ids[i:i+width],skip_special_tokens=True) for i in range(0,len(ids),width)] or ['']
                v=np.asarray(self.encoder.encode(windows,batch_size=1,normalize_embeddings=False,show_progress_bar=False),dtype=np.float32).mean(0)
                if not np.isfinite(v).all():raise ModelUnavailable('Nonfinite SFR embedding.')
                self.vectors[key]=v
            if len(self.vectors)>1000:
                # Retain this request; avoid unbounded long-running shared resource caches.
                self.vectors={k:self.vectors[k] for k in keys}
        return np.asarray([self.vectors[k] for k in keys])
    def generate(self,query,selection):
        # CPU stages never coexist: discard encoder before loading the reader.
        # GPU execution retains the previous lazy reader behavior.
        if self.device=='cpu':
            self.release_gpu();self._cpu_memory()
            if not local_model(self.cfg['models']['Qwen']):raise ModelUnavailable('Pinned Qwen checkpoint is not available locally; no download was attempted.')
            try:
                self.reader=CPUReader(self.cfg,self.tokenizer())
                return self.reader.generate(dict(question=query,task_type='freeform_qa'),selection)
            finally:self.release_gpu()
        if self.reader is None:
            self.release_gpu();self._memory(18000)
            if not local_model(self.cfg['models']['Qwen']):raise ModelUnavailable('Pinned Qwen checkpoint is not available locally; no download was attempted.')
            from src.week12.reader import Reader
            self.reader=Reader(self.cfg,'BASE_FP16')
        return self.reader.generate(dict(question=query,task_type='freeform_qa'),selection)

class CPUReader:
    """Demo-only CPU backend, retaining the frozen reader's prompt/decoding/parser.

    CPU SDPA uses a supported backend instead of the CUDA-only efficient backend.
    FP16 numerical differences are possible; these outputs are not study scores.
    """
    def __init__(self,cfg,tok):
        import torch
        from transformers import AutoModelForCausalLM
        self.cfg=cfg;self.tok=tok;spec=cfg['models']['Qwen']
        self.model=AutoModelForCausalLM.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True,device_map={'':'cpu'},torch_dtype=torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True)
        self.model.eval()
    def generate(self,example,selection):
        import torch
        from src.week12.reader import prompt_ids,parse_label
        from src.week12.protocol import digest
        ids=prompt_ids(self.tok,example['question'],example['task_type'],selection['evidence_ids'],self.cfg['labels'])
        cap=self.cfg['generation']['max_new_tokens_verification' if example['task_type']=='verification' else 'max_new_tokens_qa']
        if len(ids)+cap>self.cfg['generation']['max_context']:raise ValueError('Context overflow: no silent truncation')
        torch.manual_seed(self.cfg['seed']);data=torch.tensor([ids],device='cpu');t=time.perf_counter()
        with torch.inference_mode():out=self.model.generate(input_ids=data,attention_mask=torch.ones_like(data),max_new_tokens=cap,do_sample=False,pad_token_id=self.tok.eos_token_id)
        new=out[0,len(ids):];raw=self.tok.decode(new,skip_special_tokens=True).strip()
        return dict(raw_prediction=raw,prediction=parse_label(raw,self.cfg['labels']) if example['task_type']=='verification' else raw,prompt_tokens=len(ids),generated_tokens=len(new),generation_seconds=time.perf_counter()-t,prompt_hash=digest(ids),generation_cap_reached=len(new)>=cap,reader='BASE_FP16_CPU',model_hash=digest([self.cfg['models']['Qwen'],'BASE_FP16',None]),device='cpu',dtype='float16')
