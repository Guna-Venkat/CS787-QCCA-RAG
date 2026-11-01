"""Local deterministic SARA backend and independently resumable LM calls."""
from __future__ import annotations
import time
from contextlib import nullcontext
import json
from pathlib import Path
from .provenance import ROOT, digest, file_hash
from .data import resolve
from .caching import RowCache


def checkpoint_hashes(cfg):
    directory = resolve(cfg['checkpoint'])
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    paths = sorted(p for p in directory.iterdir() if p.is_file() and
        (p.suffix in ('.json', '.safetensors', '.model') or p.name == 'chat_template.jinja'))
    if not any(p.suffix == '.safetensors' for p in paths):
        raise FileNotFoundError('Checkpoint weights missing')
    return {str(p): file_hash(p) for p in paths}


def load_tokenizer(cfg):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(resolve(cfg['checkpoint'])), local_files_only=True, use_fast=True)


class SaraBackend:
    def __init__(self, cfg):
        from .model_registry import CANONICAL,assert_projector_compatible
        assert_projector_compatible(cfg.get('first_pass_sara_model',CANONICAL),'sara_latent')
        actual=json.loads((resolve(cfg['checkpoint'])/'adapter_config.json').read_text())['base_model_name_or_path']
        assert_projector_compatible(actual,'sara_latent')
        self.cfg=cfg
        import src
        path = str(ROOT / 'Baselines/SARA-main/src')
        if path not in src.__path__:
            src.__path__.append(path)
        from src.model.loader import load_sara_for_eval
        self.device = cfg['device']
        import torch
        # TITAN RTX cannot use the efficient SDPA kernel with BF16. Explicit
        # FP16 keeps the trained SARA checkpoint and avoids quadratic math SDPA.
        self.model, self.tokenizer = load_sara_for_eval(str(resolve(cfg['checkpoint'])),
            device=self.device, dtype=torch.float16, attn_implementation='sdpa')
        self.model.eval()
        self.answer_max_tokens = cfg['max_new_tokens']

    def identity(self):
        from .model_registry import CANONICAL
        if not hasattr(self,'_identity'):
            self._identity=dict(model_id=CANONICAL,model_revision=digest(checkpoint_hashes(self.cfg)),dtype=str(getattr(self.model,'dtype','unknown')),quantization='none',interface='trained_SARA',attention_backend='efficient_sdpa' if str(self.device).startswith('cuda') else 'cpu_sdpa')
        return self._identity

    def render(self, messages):
        # Mistral templates often reject system roles: preserve content and order.
        content = '\n\n'.join(m['content'] for m in messages)
        return self.tokenizer.apply_chat_template([dict(role='user', content=content)], tokenize=False, add_generation_prompt=True)

    def generate(self, prompt, max_tokens, retrieval_embeds=None):
        import torch
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        n = inputs.input_ids.shape[-1]
        if n + max_tokens > self.model.config.max_position_embeddings:
            raise ValueError('Prompt exceeds context capacity; no silent truncation')
        kwargs = {} if retrieval_embeds is None else {'retrieval_embeds': retrieval_embeds}
        if torch.cuda.is_available() and str(self.device).startswith('cuda'):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        # Fail explicitly if this hardware/runtime cannot provide efficient
        # attention, rather than silently allocating a quadratic math matrix.
        from torch.nn.attention import sdpa_kernel, SDPBackend
        attention = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) if str(self.device).startswith('cuda') else nullcontext()
        with torch.inference_mode(), attention:
            output = self.model.generate(**inputs, **kwargs, do_sample=False, max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id)
        if torch.cuda.is_available() and str(self.device).startswith('cuda'):
            torch.cuda.synchronize()
        # Local XMistral.generate uses inputs_embeds and returns ONLY new tokens.
        # Slicing by prompt length would silently turn every answer into empty text.
        ids = output[0].tolist()
        meta=self.identity() if hasattr(self,'cfg') else {}
        gpu=str(self.device).startswith('cuda') and torch.cuda.is_available()
        return dict(content=self.tokenizer.decode(ids, skip_special_tokens=True).strip(),
            prompt_tokens=n, generated_tokens=len(ids), generated_token_ids=ids,input_characters=len(prompt),input_words=len(prompt.split()),
            gpu=torch.cuda.get_device_name(0) if gpu else None,
            peak_memory_allocated=torch.cuda.max_memory_allocated() if gpu else None,
            peak_memory_reserved=torch.cuda.max_memory_reserved() if gpu else None,
            latency=time.perf_counter()-started,**meta)


class CachedClient:
    def __init__(self, backend, cache, inference, cost, stage, *, context=None, retrieval_embeds=None, evidence_ids=(), revision_mode=None):
        self.backend, self.cache, self.inference = backend, cache, inference
        self.cost, self.stage = cost, stage
        self.context, self.retrieval_embeds = context, retrieval_embeds
        self.answer_max_tokens = backend.answer_max_tokens
        self.latency = 0.
        self.evidence_ids=list(evidence_ids);self.revision_mode=revision_mode

    def prepare_prompt(self, messages):
        if hasattr(self.backend,'set_stage'):self.backend.set_stage(self.stage)
        prompt = self.backend.render(messages)
        if self.context is not None:
            # Insert evidence instruction inside original chat envelope. The original
            # k4 context and compressed-slot identity are otherwise unchanged.
            marker = '[/INST]'
            if marker not in self.context:
                raise ValueError('Expected canonical Mistral instruction envelope')
            prompt = self.context.rsplit(marker, 1)[0] + '\n\n' + '\n\n'.join(m['content'] for m in messages) + '\n' + marker
        return prompt

    def chat(self, messages, max_tokens=256, **kwargs):
        prompt = self.prepare_prompt(messages)
        model=self.backend.identity() if hasattr(self.backend,'identity') else {}
        identity = dict(prompt_hash=digest(prompt),decoding_hash=digest(dict(max_tokens=max_tokens,do_sample=False)),
            task=self.stage,revision_mode=self.revision_mode,evidence_ids=self.evidence_ids,model=model,config_hash=self.cache.stamp)
        key = self.cache.key(self.inference['question_id'], self.inference['paper_id'], [digest(identity)], 'lm_call')
        result = self.cache.rows.get(key)
        if result is None:
            result = self.backend.generate(prompt, max_tokens, self.retrieval_embeds)
            result = self.cache.put(key, dict(result, question_id=self.inference['question_id'],
                paper_id=self.inference['paper_id'], prompt_sha256=digest(prompt), stage=self.stage,call_identity=identity))
        self.cost.add_call(self.stage, result)
        self.latency += result['latency']
        return result
