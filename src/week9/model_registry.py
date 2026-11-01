"""Official textual generators; the trained SARA latent space is not portable."""
from dataclasses import dataclass,asdict

CANONICAL='mistralai/Mistral-7B-Instruct-v0.2'

@dataclass(frozen=True)
class ModelSpec:
    key:str
    model_id:str
    family:str
    parameters_b:float
    hidden_size:int
    context_length:int
    license:str
    gated:bool=False
    reasoning:bool=False
    secondary:bool=False
    transformers_min:str='4.43.0'
    architecture:str='AutoModelForCausalLM'
    native_dtype:str='bfloat16'
    weight_bytes_estimate:int=0
    roles:tuple=('extractor_model','revision_model')
    def to_dict(self):return dict(asdict(self),roles=list(self.roles))

MODELS=[
 ModelSpec('M0',CANONICAL,'Mistral',7.24,4096,32768,'Apache-2.0',native_dtype='float16',weight_bytes_estimate=14480000000,roles=('first_pass_sara_model','extractor_model','revision_model')),
 ModelSpec('M1','meta-llama/Llama-3.1-8B-Instruct','Llama',8.03,4096,131072,'Llama-3.1',gated=True,weight_bytes_estimate=16060000000),
 ModelSpec('M2','Qwen/Qwen2.5-7B-Instruct','Qwen2',7.61,3584,32768,'Apache-2.0',transformers_min='4.37.0',weight_bytes_estimate=15220000000),
 ModelSpec('M3','deepseek-ai/DeepSeek-R1-Distill-Qwen-7B','Qwen2-distill',7.61,3584,131072,'MIT (upstream Qwen Apache-2.0)',reasoning=True,weight_bytes_estimate=15220000000),
 ModelSpec('M4','Qwen/Qwen3-8B','Qwen3',8.2,4096,40960,'Apache-2.0',reasoning=True,secondary=True,transformers_min='4.51.0',weight_bytes_estimate=16400000000),
 ModelSpec('M5','google/gemma-3-12b-it','Gemma3',12.,3840,131072,'Gemma terms',gated=True,secondary=True,transformers_min='4.50.0',architecture='Gemma3ForConditionalGeneration',weight_bytes_estimate=24400000000),
]


def get_model(value):
    for model in MODELS:
        if value in (model.key,model.model_id):return model
    raise ValueError('Unknown/unofficial model: '+value)


def assert_projector_compatible(model_id,interface='text'):
    get_model(model_id)
    if interface=='sara_latent' and model_id!=CANONICAL:
        raise ValueError('Existing SARA projector is Mistral-specific. SAME HIDDEN DIMENSION != COMPATIBLE LATENT SPACE')
    if interface not in ('text','sara_latent'):raise ValueError('Unknown model interface')


def runtime_identity(model_id,revision,precision='fp16',thinking=False):
    from .provenance import digest
    get_model(model_id)
    if not revision or revision=='main':raise ValueError('Pin an immutable model revision from the cache manifest')
    if precision not in ('fp16','int8','nf4','nf4_fp32'):raise ValueError('Unsupported precision')
    return dict(model_id=model_id,model_revision=revision,dtype='float32' if precision=='nf4_fp32' else 'float16',quantization={'fp16':'none','int8':'bitsandbytes-LLM.int8','nf4':'bitsandbytes-NF4-double-quant','nf4_fp32':'bitsandbytes-NF4-double-quant-FP32-compute'}[precision],precision=precision,thinking=thinking)
