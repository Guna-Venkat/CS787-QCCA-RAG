"""All online calls counted; unknown measurements remain null."""
from dataclasses import dataclass, asdict, field


@dataclass
class QueryCost:
    initial_draft_prompt_tokens: int = 0
    initial_draft_generated_tokens: int = 0
    retrieval_query_time: float | None = None
    selection_latency: float | None = None
    extraction_latency: float | None = None
    revision_latency: float | None = None
    selector_prompt_tokens: int = 0
    selector_generated_tokens: int = 0
    extraction_prompt_tokens: int = 0
    extraction_generated_tokens: int = 0
    revision_prompt_tokens: int = 0
    revision_generated_tokens: int = 0
    num_lm_calls: int = 1
    final_prompt_tokens: int = 0
    index_time_embedding_cost: float | None = None
    index_time_synopsis_cost: float = 0.0
    measured_end_to_end_latency: float | None = None

    model_calls: list = field(default_factory=list)

    def add_call(self, stage, result):
        self.model_calls.append(dict(stage=stage,**{k:result[k] for k in ('model_id','model_revision','dtype','quantization','gpu','peak_memory_allocated','peak_memory_reserved','input_characters','input_words','prompt_tokens','generated_tokens','latency') if k in result}))
        prefix = {'selector': 'selector', 'extraction': 'extraction', 'revision': 'revision'}[stage]
        setattr(self, prefix + '_prompt_tokens', getattr(self, prefix + '_prompt_tokens') + result['prompt_tokens'])
        setattr(self, prefix + '_generated_tokens', getattr(self, prefix + '_generated_tokens') + result['generated_tokens'])
        self.num_lm_calls += 1
        if stage == 'revision':
            self.final_prompt_tokens = result['prompt_tokens']

    def to_dict(self):
        r = asdict(self)
        for name in ('prompt', 'generated'):
            r['total_query_' + name + '_tokens'] = sum(r[p + '_' + name + '_tokens']
                for p in ('initial_draft', 'selector', 'extraction', 'revision'))
        r['tokenizer_note']='Stage token counts are model-dependent; model_calls includes source character/word lengths when measured.'
        r['latency_note'] = 'Stage times exclude cached initial draft; end-to-end latency unmeasured.'
        return r
