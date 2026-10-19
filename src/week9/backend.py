"""Local deterministic SARA backend and independently resumable LM calls."""
from __future__ import annotations
import time
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
        import src
        path = str(ROOT / 'Baselines/SARA-main/src')
        if path not in src.__path__:
            src.__path__.append(path)
        from src.model.loader import load_sara_for_eval
        self.device = cfg['device']
        self.model, self.tokenizer = load_sara_for_eval(str(resolve(cfg['checkpoint'])), device=self.device)
        self.model.eval()
        self.answer_max_tokens = cfg['max_new_tokens']

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
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs, do_sample=False, max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id)
        if torch.cuda.is_available() and str(self.device).startswith('cuda'):
            torch.cuda.synchronize()
        # Local XMistral.generate uses inputs_embeds and returns ONLY new tokens.
        # Slicing by prompt length would silently turn every answer into empty text.
        ids = output[0].tolist()
        return dict(content=self.tokenizer.decode(ids, skip_special_tokens=True).strip(),
            prompt_tokens=n, generated_tokens=len(ids), generated_token_ids=ids,
            latency=time.perf_counter()-started)


class CachedClient:
    def __init__(self, backend, cache, inference, cost, stage, *, context=None, retrieval_embeds=None):
        self.backend, self.cache, self.inference = backend, cache, inference
        self.cost, self.stage = cost, stage
        self.context, self.retrieval_embeds = context, retrieval_embeds
        self.answer_max_tokens = backend.answer_max_tokens
        self.latency = 0.

    def prepare_prompt(self, messages):
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
        identity = [digest(prompt), str(max_tokens), self.stage]
        key = self.cache.key(self.inference['question_id'], self.inference['paper_id'], identity, 'lm_call')
        result = self.cache.rows.get(key)
        if result is None:
            result = self.backend.generate(prompt, max_tokens, self.retrieval_embeds)
            result = self.cache.put(key, dict(result, question_id=self.inference['question_id'],
                paper_id=self.inference['paper_id'], prompt_sha256=digest(prompt), stage=self.stage))
        self.cost.add_call(self.stage, result)
        self.latency += result['latency']
        return result
