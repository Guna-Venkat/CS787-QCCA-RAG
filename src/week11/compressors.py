import importlib.util
from .budgets import token_counts


def availability(cfg):
    return dict(C0='IMPLEMENTED',C1='QUALIFIED_ADAPTER' if cfg['token_compressor'].get('qualified') else 'PENDING_QUALIFICATION' if importlib.util.find_spec('llmlingua') else 'PENDING_DEPENDENCY',
                C2='IMPLEMENTED_QUERY_ONLY_EXACT_SENTENCE_TOPK',C3='IMPLEMENTED_QASPER_MISTRAL_ONLY',
                C4='FUTURE_WORK_FAITHFUL_SELECOM_NOT_IMPLEMENTED')


class LLMLingua2:
    def __init__(self,spec):
        self.spec=spec;self.model=None
    def compress(self,text,budget,question,tokenizers):
        if budget==0 or not text:return dict(text='',source_support=1.,notes='Empty/zero-budget token control')
        if budget=='FULL':return dict(text=text,source_support=1.,notes='FULL token control is identity; no token removal')
        if not self.spec.get('qualified') or not self.spec.get('revision'):
            raise RuntimeError('LLMLingua-2 pending explicit pinned-model feasibility qualification')
        if self.model is None:
            from huggingface_hub import snapshot_download
            from llmlingua import PromptCompressor
            path=snapshot_download(self.spec['id'],revision=self.spec['revision'],local_files_only=True)
            self.model=PromptCompressor(model_name=path,use_llmlingua2=True,device_map='cpu')
        result=self.model.compress_prompt([text],instruction='',question=question,target_token=budget,
            condition_compare=True,force_tokens=['\n'])
        compressed=result['compressed_prompt']
        if max(token_counts(compressed,tokenizers).values())>budget:
            # A compressor uses its own tokenizer: never pretend this meets another budget.
            raise ValueError('LLMLingua-2 output exceeds shared reader budget; no silent truncation')
        return dict(text=compressed,source_support=None,notes='Token deletion output; exact contiguous-span support is not claimed',
                    compressor_metadata={k:v for k,v in result.items() if k!='compressed_prompt'})


def sara_guard(reader,dataset):
    if reader!='Mistral' or dataset!='QASPER':
        raise ValueError('MISTRAL_ONLY_LATENT_BASELINE supports existing QASPER SARA checkpoint only')
