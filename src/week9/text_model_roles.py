"""Optional frozen-panel textual extraction/revision; canonical SARA draft stays fixed."""
import json
from .model_registry import CANONICAL,get_model,runtime_identity,assert_projector_compatible
from .model_runtime import TextRuntime
from .provenance import digest
from .data import resolve


class TextRoleBackend:
    text_only=True
    def __init__(self,cfg):
        from .backend import load_tokenizer
        self.cfg=cfg;self.device=cfg['device'];self.answer_max_tokens=cfg['max_new_tokens']
        roles=cfg['textual_model_roles'];panel=json.loads(resolve(roles['frozen_panel']).read_text())
        if panel.get('panel_hash')!=digest({k:v for k,v in panel.items() if k!='panel_hash'}) or not panel.get('frozen') or panel.get('selection_split')!='fit':raise ValueError('Invalid FIT-frozen model panel')
        self.panel_hash=panel['panel_hash'];self.allowed={r['model_id']:r for r in panel['model_panel']}
        if roles.get('first_pass_sara_model',CANONICAL)!=CANONICAL:raise ValueError('Alternative first-pass projector requires a new experiment')
        self.roles={k:roles[k] for k in ('extractor_model','revision_model')}
        for model in self.roles.values():
            if model not in self.allowed:raise ValueError('Role model is outside the frozen FIT panel')
            assert_projector_compatible(model,'text')
        self.canonical_tokenizer=load_tokenizer(cfg);self.tokenizers={};self.runtime=None;self.stage='revision'
    def set_stage(self,stage):self.stage=stage
    def identity(self):
        model=self.roles['revision_model' if self.stage=='revision' else 'extractor_model'];choice=self.allowed[model]
        return dict(runtime_identity(model,choice['revision'],choice['precision'],False),interface='text_only',panel_hash=self.panel_hash)
    @property
    def tokenizer(self):
        from transformers import AutoTokenizer
        identity=self.identity();model=identity['model_id']
        if model not in self.tokenizers:
            self.tokenizers[model]=AutoTokenizer.from_pretrained(model,revision=identity['model_revision'],local_files_only=True,cache_dir=self.cfg.get('model_cache_dir'))
        return self.tokenizers[model]
    def render(self,messages):
        identity=self.identity();kwargs={'enable_thinking':False} if get_model(identity['model_id']).key=='M4' else {}
        return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**kwargs)
    def generate(self,prompt,max_tokens,retrieval_embeds=None):
        if retrieval_embeds is not None:raise ValueError('Textual role cannot receive SARA latents')
        identity=self.identity()
        if self.runtime is None or self.runtime.identity['model_id']!=identity['model_id']:
            self.close()
            self.runtime=TextRuntime(identity['model_id'],identity['model_revision'],identity['precision'],cache_dir=self.cfg.get('model_cache_dir'),max_prompt_tokens=self.cfg.get('text_max_prompt_tokens',4096))
            self.runtime.load()
        return self.runtime.generate(prompt,max_tokens)
    def close(self):
        if self.runtime is not None:self.runtime.close();self.runtime=None
