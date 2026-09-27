"""Training sanity test (release plan §2/§4): on a SARA batch, LoRA AND projector get gradients.

Verifies, with a tiny CPU model:
- loaded class is XMistralForCausalLM with a projector;
- <COMPRESS> positions are replaced by projected embeds in the forward;
- LoRA params receive nonzero gradients;
- projector params receive nonzero gradients (compressed evidence enabled);
- the optimizer param groups contain both LoRA and projector params;
- a plain-Mistral load would be caught (fail-loud contract).

Run: ``python -m pytest tests/test_train_sara_sanity.py -q``
"""

import torch
from peft import LoraConfig, TaskType, get_peft_model

from src.model.xMistral import XMistralConfig, XMistralForCausalLM

COMPRESS_ID, PAD_ID = 301, 300
HIDDEN, RETR_DIM, VOCAB = 64, 16, 320


def _cfg():
    return XMistralConfig(
        hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=VOCAB, max_position_embeddings=128,
        projector_type="mlp2x_gelu", retriever_hidden_size=RETR_DIM, tie_word_embeddings=False,
    )


def test_lora_and_projector_get_gradients():
    torch.manual_seed(0)
    model = XMistralForCausalLM(_cfg())
    model.compress_token_id = COMPRESS_ID
    assert isinstance(model, XMistralForCausalLM) and hasattr(model, "projector")

    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.0,
                      target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora)
    model.get_base_model().projector.requires_grad_(True)  # projector trains alongside LoRA
    model.train()

    # Optimizer param groups must contain BOTH lora and projector params.
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    proj_params = [p for n, p in model.named_parameters() if "projector" in n and p.requires_grad]
    assert lora_params, "no trainable LoRA params"
    assert proj_params, "no trainable projector params"
    opt = torch.optim.AdamW([{"params": lora_params}, {"params": proj_params, "lr": 1e-3}], lr=1e-4)

    # SARA batch: 2 examples with 1 and 2 <COMPRESS> tokens; supervise the last 2 tokens.
    input_ids = torch.tensor([[7, COMPRESS_ID, 8, 9, 5], [COMPRESS_ID, COMPRESS_ID, 8, 9, 5]])
    attn = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :3] = -100  # only supervise the tail
    embeds = torch.randn(3, RETR_DIM)  # 1 + 2, example-major

    out = model(input_ids=input_ids, attention_mask=attn, retrieval_embeds=embeds,
                per_example_counts=[1, 2], labels=labels)
    assert out.loss is not None and torch.isfinite(out.loss)
    opt.zero_grad()
    out.loss.backward()

    lora_grad = max((p.grad.abs().max().item() for p in lora_params if p.grad is not None), default=0.0)
    proj_grad = max((p.grad.abs().max().item() for p in proj_params if p.grad is not None), default=0.0)
    assert lora_grad > 0, "LoRA params received zero gradient"
    assert proj_grad > 0, "projector params received zero gradient (compressed evidence not flowing?)"


def test_plain_mistral_would_be_caught():
    """A config with retriever_hidden_size<=0 yields no projector -> SARA forward must fail loudly."""
    cfg = _cfg()
    cfg.retriever_hidden_size = 0
    model = XMistralForCausalLM(cfg)
    model.compress_token_id = COMPRESS_ID
    assert not hasattr(model, "projector")
    input_ids = torch.tensor([[7, COMPRESS_ID, 9]])
    try:
        model.prepare_inputs_embeds(input_ids, torch.randn(1, RETR_DIM), per_example_counts=[1])
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "expected a loud failure when retrieval_embeds given but no projector exists"
