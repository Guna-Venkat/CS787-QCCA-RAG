"""Checkpoint round-trip test (release plan §8): a SARA adapter must reload numerically exactly.

Verifies, with a tiny CPU model, that after save -> fresh build -> load:
- LoRA params match;
- ALL projector params match (i.e. modules_to_save=["projector"] actually persists/restores it);
- added-token input-embedding rows match;
- added-token lm_head rows match (untied);
- logits on a fixed batched input (with retrieval_embeds) match.

Run: ``python -m pytest tests/test_checkpoint_roundtrip.py -q``
"""

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from src.model.loader import load_sara_extras, save_sara_extras
from src.model.xMistral import XMistralConfig, XMistralForCausalLM

COMPRESS_ID, PAD_ID = 301, 300
HIDDEN, RETR_DIM, VOCAB = 64, 16, 320


def _cfg():
    return XMistralConfig(
        hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=VOCAB, max_position_embeddings=128,
        projector_type="mlp2x_gelu", retriever_hidden_size=RETR_DIM, tie_word_embeddings=False,
    )


def _wrap(model):
    # Projector is persisted explicitly (save_sara_extras), NOT via PEFT modules_to_save.
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.0,
                     target_modules=["q_proj", "v_proj"])
    peft = get_peft_model(model, cfg)
    peft.get_base_model().projector.requires_grad_(True)  # projector trains alongside LoRA
    return peft


def test_checkpoint_roundtrip_numerical(tmp_path):
    torch.manual_seed(0)
    base = XMistralForCausalLM(_cfg()).eval()
    base.compress_token_id = COMPRESS_ID
    # Snapshot the frozen public-base weights so the reload starts from the SAME base (emulates
    # from_pretrained(PUBLIC_BASE) being identical across processes).
    base_snapshot = {k: v.clone() for k, v in base.state_dict().items()}
    peft = _wrap(base)

    # Perturb projector, LoRA, and added-token rows to distinctive nonzero values.
    with torch.no_grad():
        for n, p in peft.named_parameters():
            if "projector" in n:
                p.add_(torch.randn_like(p) * 0.1)
            if "lora_" in n:
                p.add_(torch.randn_like(p) * 0.1)
        embed = peft.get_base_model().get_input_embeddings().weight.data
        embed[[PAD_ID, COMPRESS_ID]] = torch.randn(2, HIDDEN)
        head = peft.get_base_model().get_output_embeddings().weight.data
        head[[PAD_ID, COMPRESS_ID]] = torch.randn(2, HIDDEN)

    out_dir = str(tmp_path / "adapter")
    peft.save_pretrained(out_dir)
    save_sara_extras(peft, [PAD_ID, COMPRESS_ID], out_dir, RETR_DIM)

    # Fixed input with retrieval_embeds for a logits comparison.
    input_ids = torch.tensor([[7, COMPRESS_ID, 8, COMPRESS_ID, 9]])
    embeds = torch.randn(2, RETR_DIM)
    peft.eval()
    with torch.no_grad():
        ref_logits = peft(input_ids=input_ids, retrieval_embeds=embeds, per_example_counts=[2]).logits

    # Fresh-process-style reload: same public base weights, fresh adapter load.
    base2 = XMistralForCausalLM(_cfg()).eval()
    base2.load_state_dict(base_snapshot, strict=True)
    base2.compress_token_id = COMPRESS_ID
    peft2 = PeftModel.from_pretrained(base2, out_dir)
    load_sara_extras(peft2, out_dir)
    peft2.eval()

    # Projector: every param numerically equal (explicit persistence, not modules_to_save).
    proj1 = dict(peft.get_base_model().projector.state_dict())
    proj2 = dict(peft2.get_base_model().projector.state_dict())
    assert proj1, "no projector params found"
    for key, p in proj1.items():
        assert key in proj2, f"projector param {key} missing after reload"
        assert torch.allclose(p, proj2[key], atol=1e-6), f"projector mismatch at {key}"

    # Added-token rows numerically equal.
    e1 = peft.get_base_model().get_input_embeddings().weight.data[[PAD_ID, COMPRESS_ID]]
    e2 = peft2.get_base_model().get_input_embeddings().weight.data[[PAD_ID, COMPRESS_ID]]
    assert torch.allclose(e1, e2, atol=1e-6)
    h1 = peft.get_base_model().get_output_embeddings().weight.data[[PAD_ID, COMPRESS_ID]]
    h2 = peft2.get_base_model().get_output_embeddings().weight.data[[PAD_ID, COMPRESS_ID]]
    assert torch.allclose(h1, h2, atol=1e-6)

    # Logits on the fixed input match.
    with torch.no_grad():
        new_logits = peft2(input_ids=input_ids, retrieval_embeds=embeds, per_example_counts=[2]).logits
    assert torch.allclose(ref_logits, new_logits, atol=1e-5), \
        f"max logit diff {(ref_logits - new_logits).abs().max().item()}"
