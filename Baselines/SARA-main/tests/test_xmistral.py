"""Correctness tests for the modernized XMistral SARA model (§1 of the release plan).

Run: ``python -m pytest tests/test_xmistral.py -q`` (CPU, tiny random model).

Covers: (1) two examples with different #compressed docs; (2) left padding; (3) one example with no
compressed docs; (4) mismatched compress/embedding counts -> raises; (5) batched == single greedy;
(6) decoding when generate() receives projected inputs_embeds.
"""

import pytest
import torch

from src.model.xMistral import XMistralConfig, XMistralForCausalLM, extract_generated_text

COMPRESS_ID = 300
HIDDEN = 64
RETR_DIM = 16
VOCAB = 320  # COMPRESS_ID (300) and pad (301) are inside the vocab


def _tiny_model():
    torch.manual_seed(0)
    cfg = XMistralConfig(
        hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
        max_position_embeddings=128, projector_type="mlp2x_gelu", retriever_hidden_size=RETR_DIM,
    )
    model = XMistralForCausalLM(cfg).eval()
    model.compress_token_id = COMPRESS_ID
    return model


def _row(prefix_len, n_compress, pad_left, total_len):
    """Build one left-padded row: [pad...] [tokens] [<COMPRESS> * n] [tokens]."""
    body = [7] * prefix_len + [COMPRESS_ID] * n_compress + [8, 9]
    pad = [301] * (total_len - len(body))
    ids = (pad + body) if pad_left else (body + pad)
    mask = [0] * len(pad) + [1] * len(body)
    if not pad_left:
        mask = [1] * len(body) + [0] * len(pad)
    return ids, mask


def test_two_examples_different_compress_counts():
    model = _tiny_model()
    # example 0 has 1 compress token, example 1 has 3 -> 4 embeds total, example-major order
    ids0, m0 = _row(prefix_len=2, n_compress=1, pad_left=True, total_len=9)
    ids1, m1 = _row(prefix_len=1, n_compress=3, pad_left=True, total_len=9)
    input_ids = torch.tensor([ids0, ids1])
    attn = torch.tensor([m0, m1])
    embeds = torch.randn(4, RETR_DIM)
    out = model.prepare_inputs_embeds(input_ids, embeds, per_example_counts=[1, 3])
    assert out.shape == (2, 9, HIDDEN)
    # the compress positions must equal projected embeds (example-major, doc-order)
    proj = model.projector(embeds.to(out.dtype))
    mask = input_ids == COMPRESS_ID
    assert torch.allclose(out[mask], proj, atol=1e-5)


def test_left_padding_does_not_shift_injection():
    model = _tiny_model()
    ids, m = _row(prefix_len=2, n_compress=2, pad_left=True, total_len=10)
    input_ids = torch.tensor([ids])
    embeds = torch.randn(2, RETR_DIM)
    out = model.prepare_inputs_embeds(input_ids, embeds, per_example_counts=[2])
    proj = model.projector(embeds.to(out.dtype))
    assert torch.allclose(out[input_ids == COMPRESS_ID], proj, atol=1e-5)


def test_example_with_no_compressed_docs():
    model = _tiny_model()
    # one example, zero compress tokens, retrieval_embeds=None -> plain embedding
    input_ids = torch.tensor([[7, 7, 8, 9]])
    out = model.prepare_inputs_embeds(input_ids, None)
    assert torch.allclose(out, model.model.embed_tokens(input_ids))


def test_mismatched_counts_raise():
    model = _tiny_model()
    input_ids = torch.tensor([[7, COMPRESS_ID, COMPRESS_ID, 9]])  # 2 compress tokens
    # global mismatch
    with pytest.raises(ValueError):
        model.prepare_inputs_embeds(input_ids, torch.randn(3, RETR_DIM), per_example_counts=[2])
    # per-example mismatch
    with pytest.raises(ValueError):
        model.prepare_inputs_embeds(input_ids, torch.randn(2, RETR_DIM), per_example_counts=[1])
    # wrong embedding dim
    with pytest.raises(ValueError):
        model.prepare_inputs_embeds(input_ids, torch.randn(2, RETR_DIM + 1), per_example_counts=[2])


def test_batched_equals_single_greedy():
    model = _tiny_model()
    ids0, m0 = _row(prefix_len=3, n_compress=1, pad_left=True, total_len=9)
    ids1, m1 = _row(prefix_len=1, n_compress=2, pad_left=True, total_len=9)
    embeds0 = torch.randn(1, RETR_DIM)
    embeds1 = torch.randn(2, RETR_DIM)
    gen_kw = dict(max_new_tokens=5, do_sample=False, num_beams=1)

    batched = model.generate(
        input_ids=torch.tensor([ids0, ids1]), attention_mask=torch.tensor([m0, m1]),
        retrieval_embeds=torch.cat([embeds0, embeds1], dim=0), per_example_counts=[1, 2], **gen_kw,
    )
    single0 = model.generate(input_ids=torch.tensor([ids0]), attention_mask=torch.tensor([m0]),
                             retrieval_embeds=embeds0, per_example_counts=[1], **gen_kw)
    single1 = model.generate(input_ids=torch.tensor([ids1]), attention_mask=torch.tensor([m1]),
                             retrieval_embeds=embeds1, per_example_counts=[2], **gen_kw)
    assert torch.equal(batched[0], single0[0])
    assert torch.equal(batched[1], single1[0])


def test_generate_returns_only_new_tokens_and_decodes():
    model = _tiny_model()
    ids, m = _row(prefix_len=3, n_compress=2, pad_left=True, total_len=10)
    out = model.generate(input_ids=torch.tensor([ids]), attention_mask=torch.tensor([m]),
                         retrieval_embeds=torch.randn(2, RETR_DIM), per_example_counts=[2],
                         max_new_tokens=6, do_sample=False, num_beams=1)
    # generate-from-inputs_embeds returns ONLY new tokens on this transformers version
    assert out.shape[1] == 6

    class _Tok:
        def batch_decode(self, ids, **kw):
            return [" ".join(map(str, row.tolist())) for row in ids]

    text = extract_generated_text(_Tok(), out)
    assert isinstance(text, list) and len(text) == 1 and isinstance(text[0], str)
